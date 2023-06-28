/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_

#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

namespace embedding {
template<typename K, typename V>
class DramLevelDBStore : public MultiTierStorage<K, V> {
 public:
  DramLevelDBStore(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc, const std::string& name)
      : dram_feat_desc_(feat_desc),
        MultiTierStorage<K, V>(sc, name) {
    dram_ = new DramStorage<K, V>(sc, feat_desc);
    leveldb_ = new LevelDBStore<K, V>(sc, feat_desc);
  }

  ~DramLevelDBStore() override {
    MultiTierStorage<K, V>::DeleteFromEvictionManager();
    delete dram_;
    delete leveldb_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DramLevelDBStore);

  Status Get(K key, void** value_ptr) override {
    Status s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = leveldb_->Get(key, value_ptr);
    if (s.ok()) {
      s = dram_->TryInsert(key, *value_ptr);
      if (s.ok()) {
        return s;
      }
      leveldb_->DestroyValuePtr(*value_ptr);
      return dram_->Get(key, value_ptr);
    }
    return s;
  }

  void Insert(K key, void* value_ptr) override {
    LOG(FATAL)<<"Unsupport Insert(K, void*) in DramLevelDBStore.";
  }

  void Insert(K key, void** value_ptr,
              size_t alloc_len) override {
    dram_->Insert(key, value_ptr, alloc_len);
  }

  Status GetOrCreate(K key, void** value_ptr,
      size_t size) override {
    Status s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = leveldb_->Get(key, value_ptr);
    if (s.ok()) {
      s = dram_->TryInsert(key, *value_ptr);
      if (s.ok()) {
        return s;
      }
      leveldb_->DestroyValuePtr(*value_ptr);
      return dram_->Get(key, value_ptr);
    }
    dram_->Insert(key, value_ptr, size);
    return Status::OK();
  }
 
  Status Remove(K key) override {
    dram_->Remove(key);
    leveldb_->Remove(key);
    return Status::OK();
  }

  bool IsUseHbm() override {
    return false;
  }

  bool IsSingleHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    /*The return value is set to false temporarily,
      because the corresponding interface is not implemented.*/
    return false;
  }

  int64 Size() const override {
    int64 total_size = dram_->Size();
    total_size += leveldb_->Size();
    return total_size;
  }

  int64 Size(int level) const override {
    if (level == 0) {
      return dram_->Size();
    } else if (level == 1) {
      return leveldb_->Size();
    } else {
      return -1;
    }
  }

  int LookupTier(K key) const override {
    Status s = dram_->Contains(key);
    if (s.ok())
      return 0;
    s = leveldb_->Contains(key);
    if (s.ok())
      return 1;
    return -1;
  }

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    std::vector<K> key_list, tmp_leveldb_key_list;
    std::vector<void*> value_ptr_list, tmp_leveldb_value_list;
    TF_CHECK_OK(dram_->GetSnapshot(&key_list, &value_ptr_list));

    TF_CHECK_OK(leveldb_->GetSnapshot(
        &tmp_leveldb_key_list, &tmp_leveldb_value_list));

    for (int64 i = 0; i < tmp_leveldb_value_list.size(); i++) {
      tmp_leveldb_value_list[i] =
          (void*)((int64)tmp_leveldb_value_list[i] | (1L << 48));
    }

    std::vector<K> leveldb_key_list;
    for (int64 i = 0; i < tmp_leveldb_key_list.size(); i++) {
      Status s = dram_->Contains(tmp_leveldb_key_list[i]);
      if (!s.ok()) {
        key_list.emplace_back(tmp_leveldb_key_list[i]);
        leveldb_key_list.emplace_back(tmp_leveldb_key_list[i]);
        value_ptr_list.emplace_back(tmp_leveldb_value_list[i]);
      }
    }

    ValueIterator<V>* value_iter =
        leveldb_->GetValueIterator(
            leveldb_key_list, emb_config.emb_index, value_len);

    {
      mutex_lock l(*(leveldb_->get_mutex()));
      std::vector<FeatureDescriptor<V>*> feat_desc_list(2);
      HbmMultiTierFeatureDescriptor<V> hbm_feat_desc(nullptr, 1, true, true);
      feat_desc_list[0] = dram_feat_desc_;
      feat_desc_list[1] = &hbm_feat_desc;
      TF_CHECK_OK((Storage<K, V>::SaveToCheckpoint(
          tensor_name, writer,
          emb_config,
          value_len, default_value,
          key_list,
          value_ptr_list,
          feat_desc_list,
          value_iter)));
    }

    for (auto it: tmp_leveldb_value_list) {
      cpu_allocator()->DeallocateRaw((void*)((int64)it & 0x1ffffffffffff));
    }
    delete value_iter;

    return Status::OK();
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    void* value_ptr;
    for (int64 i = 0; i < evict_size; ++i) {
      if (dram_->Get(evict_ids[i], &value_ptr).ok()) {
        TF_CHECK_OK(leveldb_->Commit(evict_ids[i], value_ptr));
        TF_CHECK_OK(dram_->Remove(evict_ids[i]));
        dram_->DestroyValuePtr(value_ptr);
      }
    }
    return Status::OK();
  }

  Status EvictionWithDelayedDestroy(K* evict_ids, int64 evict_size) override {
    mutex_lock l(*(dram_->get_mutex()));
    mutex_lock l1(*(leveldb_->get_mutex()));
    MultiTierStorage<K, V>::ReleaseInvalidValuePtr(dram_->feature_descriptor());
    void* value_ptr = nullptr;
    for (int64 i = 0; i < evict_size; ++i) {
      if (dram_->Get(evict_ids[i], &value_ptr).ok()) {
        TF_CHECK_OK(leveldb_->Commit(evict_ids[i], value_ptr));
        TF_CHECK_OK(dram_->Remove(evict_ids[i]));
        MultiTierStorage<K, V>::KeepInvalidValuePtr(value_ptr);
      }
    }
    return Status::OK();
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    leveldb_->SetTotalDims(total_dims);
  }

 private:
  DramStorage<K, V>* dram_;
  LevelDBStore<K, V>* leveldb_;
  FeatureDescriptor<V>* dram_feat_desc_ = nullptr;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_
