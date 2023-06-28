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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NORMAL_FEATURE_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NORMAL_FEATURE_DESCRIPTOR_H_
#include "tensorflow/core/framework/embedding/feature_descriptor.h"
#include "tensorflow/core/framework/embedding/hbm_multi_tier_feature_descriptor.h"

namespace tensorflow {
namespace embedding {
template<class V>
class NormalFeatureDescriptor: public FeatureDescriptor<V> {
 public:
  NormalFeatureDescriptor(Allocator* alloc, int64 slot_num,
                          bool need_record_freq,
                          bool need_record_version)
      : alloc_bytes_(0),
        alloc_(alloc),
        FeatureDescriptor<V>(slot_num,
                             need_record_freq,
                             need_record_version) {}
  
  NormalFeatureDescriptor(NormalFeatureDescriptor<V>* feat_desc)
      : alloc_bytes_(feat_desc->alloc_bytes_),
        alloc_(feat_desc->alloc_),
        FeatureDescriptor<V>(feat_desc) {}
  
  NormalFeatureDescriptor(HbmMultiTierFeatureDescriptor<V>* feat_desc)
      : alloc_bytes_(0),
        alloc_(feat_desc->dram_alloc_),
        FeatureDescriptor<V>(feat_desc) {
    alloc_bytes_ = feat_desc->dram_alloc_bytes_ + 
                   feat_desc->hbm_alloc_bytes_ -
                   sizeof(V*);
  }

  ~NormalFeatureDescriptor() {}
  
  void InitSlotInfo(int emb_index, int64 embedding_dim,
                    const std::pair<V*, int64>& default_value) override {
    bool is_compute_alloc_bytes = FeatureDescriptor<V>::SetEmbeddingInfo(
        emb_index, embedding_dim, default_value);
    if (is_compute_alloc_bytes) {
      FeatureDescriptor<V>::ComputeAllocBytes(&alloc_bytes_);
      FeatureDescriptor<V>::CreateFreqAndVersionDescriptor(&alloc_bytes_);
    }
  }

  V* GetEmbedding(void *val, int emb_index) override {
    return reinterpret_cast<V*>(val)
        + FeatureDescriptor<V>::slot_infos_[emb_index].embedding_offset;
  }

  void* Allocate() override {
    void* val = alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment, alloc_bytes_);
    FeatureDescriptor<V>::InitFreqAndVersion(val);
    return val;
  }

  void Deallocate(void* val) override {
    alloc_->DeallocateRaw(val);
  }

  void SetDefaultValue(void* val, int64 index) override {
    for (int i = 0; i < FeatureDescriptor<V>::slot_infos_.size(); i++) {
      V* val_ptr = GetEmbedding(val, i);
      FeatureDescriptor<V>::SetDefaultValue((void*)val_ptr, i, index);
    }
  }

  void SetAllocator(Allocator* alloc) override {
    alloc_ = alloc;
  }

  int data_bytes() override {
    return alloc_bytes_;
  }

 private:
  int alloc_bytes_;
  Allocator* alloc_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NORMAL_FEATURE_DESCRIPTOR_H_