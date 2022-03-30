#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {

template <class V>
class ValuePtr;

template <class K, class V>
class SSDKV : public KVInterface<K, V> {
 public:
  SSDKV(std::string path) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map.max_load_factor(0.8);
    hash_map.set_empty_key(-1);
    hash_map.set_deleted_key(-2);
    current_version = 0;
    current_offset = 0;
    buffer_size = 4 * 1024;  // Write 4MB at once.
    buffer_cur = 0;
    emb_files.push_back(EmbFile(path_, current_version));

    counter_ = new SizeCounter<K>(8);
    total_app_count = 0;
    new_value_ptr_fn_ = [](size_t size) {
      return new NormalContiguousValuePtr<V>(ev_allocator(), size);
    };
    compaction_ration = 2;  // unique key 到达存储的 key 的compaction_ration倍时
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
    val_len = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
    buffer_size = std::max(buffer_size, val_len);
    write_buffer = new char[buffer_size];
    unsigned int max_key_count = 1 + int(buffer_size / val_len);
    key_buffer = new K[max_key_count];
    max_app_count = 10 * 1024 * 1024 / val_len;  // 10MB TODO:
  }

  ~SSDKV() {
    if (buffer_cur > 0) {
      emb_files[current_version].fs.write(write_buffer, buffer_cur * val_len);
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur = 0;
    }
    for (int i = 0; i < emb_files.size(); i++) {
      emb_files[i].fs.close();
    }
    delete[] write_buffer;
    delete[] key_buffer;
  }

  Status UpdateFlushStatus() {
    for (int i = 0; i < buffer_cur; ++i) {
      auto iter = hash_map.find(key_buffer[i]);
      if (iter == hash_map.end()) {
        return errors::NotFound("Unable to find Key: ", key_buffer[i],
                                " in SSDKV.");
      } else {
        iter->second.flushed = true;
      }
    }
    return Status::OK();
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    SingleThreadDynamicCompaction();
    spin_rd_lock l(mu);
    auto iter = hash_map.find(key);
    if (iter == hash_map.end()) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDKV.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      EmbPosition posi = iter->second;
      if (posi.flushed) {
        emb_files[posi.version].fs.seekg(posi.offset, std::ios::beg);
        emb_files[posi.version].fs.read((char*)(val->GetPtr()), val_len);
      } else {
        memcpy((char*)val->GetPtr(), write_buffer + posi.buffer_offset,
               val_len);
      }
      *value_ptr = val;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    SingleThreadDynamicCompaction();
    spin_wr_lock l(mu);
    auto iter = hash_map.find(key);
    if (iter == hash_map.end()) {
      if (buffer_cur * val_len + val_len > buffer_size) {
        emb_files[current_version].fs.write(write_buffer, buffer_cur * val_len);
        emb_files[current_version].app_count += buffer_cur;
        if (emb_files[current_version].app_count >= max_app_count) {
          ++current_version;
          current_offset = 0;
          emb_files.push_back(EmbFile(path_, current_version));
        }
        TF_CHECK_OK(UpdateFlushStatus());
        buffer_cur = 0;
      }
      hash_map[key] = EmbPosition(current_offset, current_version,
                                  buffer_cur * val_len, false);
      current_offset += val_len;
      memcpy(write_buffer + buffer_cur * val_len, (char*)value_ptr->GetPtr(),
             val_len);
      key_buffer[buffer_cur] = key;
      ++buffer_cur;
      counter_->add(key, 1);
      total_app_count++;
      return Status::OK();
    } else {
      return errors::AlreadyExists("already exists Key: ", key, " in SSDKV.");
    }
  }

  Status BatchInsert(std::vector<K> keys,
                     std::vector<ValuePtr<V>*> value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(std::vector<K> keys,
                     std::vector<ValuePtr<V>*> value_ptrs) {
    SingleThreadDynamicCompaction();
    spin_wr_lock l(mu);
    total_app_count += keys.size();
    for (int i = 0; i < keys.size(); i++) {
      //====[
      if (buffer_cur * val_len + val_len > buffer_size) {
        emb_files[current_version].fs.write(write_buffer, buffer_cur * val_len);
        emb_files[current_version].app_count += buffer_cur;
        if (emb_files[current_version].app_count >= max_app_count) {
          ++current_version;
          current_offset = 0;
          emb_files.push_back(EmbFile(path_, current_version));//EmbFile
        }
        TF_CHECK_OK(UpdateFlushStatus());
        buffer_cur = 0;
      }
      //====]
      hash_map[keys[i]] = EmbPosition(current_offset, current_version,
                                      buffer_cur * val_len, false);
      current_offset += val_len;
      memcpy(write_buffer + buffer_cur * val_len,
             (char*)value_ptrs[i]->GetPtr(), val_len);
      key_buffer[buffer_cur] = keys[i];
      ++buffer_cur;
      delete value_ptrs[i];
    }
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    SingleThreadDynamicCompaction();
    spin_wr_lock l(mu);
    total_app_count++;
    if (buffer_cur * val_len + val_len > buffer_size) {
      emb_files[current_version].fs.write(write_buffer, buffer_cur * val_len);
      emb_files[current_version].app_count += buffer_cur;
      if (emb_files[current_version].app_count >= max_app_count) {
        ++current_version;
        current_offset = 0;
        emb_files.push_back(EmbFile(path_, current_version));
      }
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur = 0;
    }
    hash_map[key] = EmbPosition(current_offset, current_version,
                                buffer_cur * val_len, false);
    current_offset += val_len;
    memcpy(write_buffer + buffer_cur * val_len, (char*)value_ptr->GetPtr(),
           val_len);
    key_buffer[buffer_cur] = key;
    ++buffer_cur;
    delete value_ptr;
    return Status::OK();
  }

  Status Remove(K key) {
    counter_->sub(key, 1);
    spin_wr_lock l(mu);
    if (hash_map.erase(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ", key, " in SSDKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) {
    spin_rd_lock l(mu);
    for (const auto it : hash_map) {
      key_list->push_back(it.first);
      EmbPosition posi = it.second;
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      if (posi.flushed) {
        emb_files[posi.version].fs.seekg(posi.offset, std::ios::beg);
        emb_files[posi.version].fs.read((char*)(val->GetPtr()), val_len);
      } else {
        memcpy((char*)val->GetPtr(), write_buffer + posi.buffer_offset,
               val_len);
      }
      value_ptr_list->push_back(val);
    }
    return Status::OK();
  }

  int64 Size() const { return hash_map.size(); }

  void FreeValuePtr(ValuePtr<V>* value_ptr) { delete value_ptr; }

  std::string DebugString() const {
    return strings::StrCat("counter_->size(): ", counter_->size(),
                           "total_app_count->size(): ", total_app_count);
  }

 private:
  void SingleThreadDynamicCompaction() {
    return;//策略
    spin_wr_lock l(mu);
    if (hash_map.size() * compaction_ration < total_app_count) {
      emb_files[current_version].fs.write(write_buffer, buffer_cur * val_len);
      emb_files[current_version].app_count += buffer_cur;
      TF_CHECK_OK(UpdateFlushStatus());
      size_t save_version = current_version;
      ++current_version;  // 无论如何都+1
      current_offset = 0;
      emb_files.push_back(EmbFile(path_, current_version));
      buffer_cur = 0;
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      total_app_count = hash_map.size();  // important
      for (const auto it : hash_map) {
        EmbPosition posi = it.second;
        if (!posi.flushed) {
          LOG(INFO) << "BUG!!!!!!!";
          posi.Print();
        }
        emb_files[posi.version].fs.seekg(posi.offset, std::ios::beg);
        emb_files[posi.version].fs.read((char*)(val->GetPtr()), val_len);

        if (buffer_cur * val_len + val_len > buffer_size) {
          emb_files[current_version].fs.write(write_buffer,
                                              buffer_cur * val_len);
          emb_files[current_version].app_count += buffer_cur;
          if (emb_files[current_version].app_count >= max_app_count) {
            ++current_version;
            current_offset = 0;
            emb_files.push_back(EmbFile(path_, current_version));
          }
          TF_CHECK_OK(UpdateFlushStatus());
          buffer_cur = 0;
        }
        hash_map[it.first] = EmbPosition(current_offset, current_version,
                                         buffer_cur * val_len, false);
        current_offset += val_len;
        memcpy(write_buffer + buffer_cur * val_len, (char*)val->GetPtr(),
               val_len);
        key_buffer[buffer_cur] = it.first;
        ++buffer_cur;
      }
      // remove file
      for (int i = 0; i <= save_version; ++i) {
        std::remove(emb_files[i].filepath.c_str());
      }
    }
  }

 private:
 //局部性
  size_t val_len;
  char* write_buffer;
  K* key_buffer;
  size_t buffer_size;
  size_t buffer_cur;
  SizeCounter<K>* counter_;
  size_t total_app_count;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
  int total_dims_;

  mutable easy_spinrwlock_t mu = EASY_SPINRWLOCK_INITIALIZER;
  class EmbPosition {
   public:
    size_t offset;         // 在文件中的偏移
    size_t version;        // 存储在哪个文件中
    size_t buffer_offset;  // 在buffer中的偏移
    bool flushed;          //是否刷到了磁盘上
    EmbPosition(size_t o, size_t v, size_t bo, bool f)
        : offset(o), version(v), buffer_offset(bo), flushed(f) {}
    EmbPosition()
        : offset(-1), version(-1), buffer_offset(-1), flushed(false) {}
    void Print() {
      LOG(INFO) << "EmbPosition: "
                << "offset= " << offset << ", version= " << version
                << ", buffer_offset= " << buffer_offset
                << ", flushed= " << flushed;
    }
  };
  struct EmbFile {
    std::fstream fs;
    bool active;
    size_t app_count;
    size_t version;
    std::string filepath;
    EmbFile(std::string path_, size_t ver) {
      version = ver;
      std::stringstream ss;
      ss << std::setw(4) << std::setfill('0') << ver << ".emb";
      filepath = path_ + ss.str();
      fs = std::fstream(filepath, std::ios::app | std::ios::in | std::ios::out |
                                      std::ios::binary);
      active = true;
      CHECK(fs.good());
      app_count = 0;
    }
  };
  float compaction_ration;
  size_t max_app_count;
  google::dense_hash_map<K, EmbPosition> hash_map;
  std::vector<EmbFile> emb_files;
  size_t current_version;
  size_t current_offset;
};

}  // namespace tensorflow

#endif TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
