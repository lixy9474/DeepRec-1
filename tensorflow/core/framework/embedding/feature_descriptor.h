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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

class BaseFreqDescriptor {
 public:
  virtual int64 GetFreq(void* value_ptr) = 0;
  virtual void AddFreq(void* value_ptr, int64 freq) {}
  virtual void SetFreq(void* value_ptr, int64 freq) {}
};

class FreqDescriptor: public BaseFreqDescriptor {
 public:
  explicit FreqDescriptor(int offset_byte)
      : offset_byte_(offset_byte) {}

  int64 GetFreq(void* value_ptr) override {
    return *(int64*)(value_ptr + offset_byte_);
  }

  void AddFreq(void* value_ptr, int64 freq) override {
    *(int64*)(value_ptr + offset_byte_) += freq;
  }

  void SetFreq(void* value_ptr, int64 freq) override {
    *(int64*)(value_ptr + offset_byte_) = freq;
  }

 private:
  int offset_byte_;
};

class NonFreqDescriptor: public BaseFreqDescriptor {
 public:
  int64 GetFreq(void* value_ptr) override {
    LOG(FATAL)<<"Can not get freq from NonFreqCounter.";
  }
};

class BaseVersionDescriptor {
 public:
  virtual int64 GetVersion(void* value_ptr) = 0;
  virtual void UpdateVersion(void* value_ptr, int64 version) {}
};

class VersionDescriptor: public BaseVersionDescriptor {
 public:
  explicit VersionDescriptor(int offset_byte)
      : offset_byte_(offset_byte) {}
  
  int64 GetVersion(void* value_ptr) override {
    return *(int64*)(value_ptr + offset_byte_);
  }

  void UpdateVersion(void* value_ptr, int64 version) override {
    *(int64*)(value_ptr + offset_byte_) = version;
  }
 private:
  int offset_byte_;
};

class NonVersionDescriptor: public BaseVersionDescriptor {
 public:
  int64 GetVersion(void* value_ptr) override {
    LOG(FATAL)<<"Can not get version from NonFreqCounter.";
  }
};


template <class V>
class FeatureDescriptor {
 public:
  FeatureDescriptor() {}
  virtual ~FeatureDescriptor() {}

  virtual void SetEmbeddingDim(int emb_index, int64 embedding_dim) = 0;
  virtual V* GetEmbedding(void *val, int emb_index) = 0;
  virtual void* Allocate() = 0;
  virtual void Deallocate(void* val) = 0;
  virtual void SetAllocator(Allocator* alloc) = 0;
  virtual void SetDefaultValue(void* val, int64 emb_index, 
                               V* default_value_ptr,
                               int64 default_value_len) = 0;
  virtual int64 GetFreq(void* val) = 0;
  virtual int64 GetVersion(void* val) = 0;
  virtual void UpdateVersion(void* val, int64 version) = 0;
  virtual void SetFreq(void* val, int64 freq) = 0;
  virtual void AddFreq(void* val, int64 freq) = 0;
  virtual int data_bytes() = 0;
};

template<class V>
class NormalFeatureDescriptor: public FeatureDescriptor<V> {
 public:
  NormalFeatureDescriptor(Allocator* alloc, int64 slot_num,
                          bool need_record_freq,
                          bool need_record_version)
      : alloc_bytes_(0),
        alloc_(alloc) {
    embedding_offsets_.resize(slot_num);
    embedding_dims_.resize(slot_num);
    for (int i = 0; i < embedding_offsets_.size(); i++) {
      embedding_offsets_[i] = EMPTY_OFFSET_VALUE;
    }

    if (!need_record_freq) {
      freq_desc_.reset(new NonFreqDescriptor());
    }
    if (!need_record_version) {
      version_desc_.reset(new NonVersionDescriptor());
    }
  }
  ~NormalFeatureDescriptor() {}
  
  void SetEmbeddingDim(int emb_index, int64 embedding_dim) override {
    bool is_aligned = true;
    TF_CHECK_OK(ReadBoolFromEnvVar("EV_DATA_ALIGNED", true,
        &is_aligned));
    if (is_aligned) {
      embedding_dim = ComputeAlignedDim(embedding_dim);
    }

    //Avoid parallel consitency issue
    __sync_bool_compare_and_swap(
        &embedding_offsets_[emb_index], EMPTY_OFFSET_VALUE,  embedding_dim);
    embedding_dims_[emb_index] = embedding_dim;
    LOG(INFO)<<embedding_offsets_[emb_index]<<", "<<embedding_dims_[emb_index];
    //Check whether all offsets are set
    for (int i = 0; i < embedding_offsets_.size(); i++) {
      if (embedding_offsets_[i] == EMPTY_OFFSET_VALUE) {
        return;
      }
    }

    ComputeEmbeddingOffsets();
    ComputeAllocSize();
    if (!freq_desc_) {
      freq_desc_.reset(new FreqDescriptor(alloc_bytes_));
      alloc_bytes_ += sizeof(int64);
    }
    if (!version_desc_) {
      version_desc_.reset(new VersionDescriptor(alloc_bytes_));
      alloc_bytes_ += sizeof(int64);
    }    
  }

  V* GetEmbedding(void *val, int emb_index) override {
    return reinterpret_cast<V*>(val) + embedding_offsets_[emb_index];
  }

  void* Allocate() override {
    void* val = alloc_->AllocateRaw(Allocator::kAllocatorAlignment, alloc_bytes_);
    freq_desc_->SetFreq(val, 0);
    LOG(INFO)<<"Alloc bytes: "<<alloc_bytes_;
    return val;
  }

  void Deallocate(void* val) override {
    alloc_->DeallocateRaw(val);
  }

  void SetDefaultValue(void* val, int64 emb_index, 
                       V* default_value_ptr,
                       int64 default_value_len) override {
    V* val_ptr = GetEmbedding(val, emb_index);
    memcpy(val_ptr, default_value_ptr, default_value_len * sizeof(V));
  }

  void SetAllocator(Allocator* alloc) override {
    alloc_ = alloc;
  }

  void CopyData(NormalFeatureDescriptor<V>* src_feat_desc,
                void* src_value_ptr,
                void* dst_value_ptr) {
    for (int i = 0; i < embedding_offsets_.size(); i++) {
      memcpy(dst_value_ptr,
             src_value_ptr,
             alloc_bytes_);
    }
  };

  int64 GetFreq(void* val) override {
    return freq_desc_->GetFreq(val);
  }

  int64 GetVersion(void* val) override {
    return version_desc_->GetVersion(val);
  }

  void SetFreq(void* val, int64 freq) override {
    freq_desc_->SetFreq(val, freq);
  }

  void UpdateVersion(void* val, int64 version) override {
    version_desc_->UpdateVersion(val, version);
  }

  void AddFreq(void* val, int64 freq) override {
    freq_desc_->AddFreq(val, freq);
  }

  int data_bytes() override {
    return alloc_bytes_;
  }

 private:
  int64 ComputeAlignedDim(int64 embedding_dim) {
    int padding_bytes =
        ALIGN_BYTES - embedding_dim * sizeof(V) % ALIGN_BYTES;
    if (padding_bytes == ALIGN_BYTES) {
      return embedding_dim;
    } else {
      return embedding_dim + padding_bytes / sizeof(V);
    }
  }
   
  void ComputeAllocSize() {
   for(auto dim: embedding_dims_) {
     alloc_bytes_ += dim * sizeof(V);
   }
  }

  void ComputeEmbeddingOffsets() {
    for (int i = embedding_offsets_.size() - 1 ; i >= 0; i--) {
      embedding_offsets_[i] = 0;
      for (int j = 0; j < i; j++) {
        embedding_offsets_[i] += embedding_offsets_[j]; 
      }
    }
  }

 private:
  int alloc_bytes_;
  Allocator* alloc_;
  std::vector<int> embedding_dims_;
  std::vector<int> embedding_offsets_;
  const int EMPTY_OFFSET_VALUE= -1;
  const int ALIGN_BYTES = 16;
  std::unique_ptr<BaseFreqDescriptor> freq_desc_;
  std::unique_ptr<BaseVersionDescriptor> version_desc_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_