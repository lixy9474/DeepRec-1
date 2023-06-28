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
  virtual BaseFreqDescriptor* Clone() {};
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

  BaseFreqDescriptor* Clone() override {
    return new FreqDescriptor(offset_byte_);
  }
  
 private:
  int offset_byte_;
};

class NonFreqDescriptor: public BaseFreqDescriptor {
 public:
  int64 GetFreq(void* value_ptr) override {
    LOG(FATAL)<<"Can not get freq from NonFreqCounter.";
  }

  BaseFreqDescriptor* Clone() override {
    return new NonFreqDescriptor();
  }
};

class BaseVersionDescriptor {
 public:
  virtual int64 GetVersion(void* value_ptr) = 0;
  virtual void UpdateVersion(void* value_ptr, int64 version) {}
  virtual BaseVersionDescriptor* Clone() = 0;
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

  BaseVersionDescriptor* Clone() override {
    return new VersionDescriptor(offset_byte_);
  }

 private:
  int offset_byte_;
};

class NonVersionDescriptor: public BaseVersionDescriptor {
 public:
  int64 GetVersion(void* value_ptr) override {
    LOG(FATAL)<<"Can not get version from NonFreqCounter.";
  }

  BaseVersionDescriptor* Clone() override {
    return new NonVersionDescriptor();
  }
};

struct SlotInfo {
  int embedding_dim;
  int embedding_offset;
  void* default_value;
  int64 default_value_dim;
  int default_value_len;
};

template <class V>
class FeatureDescriptor {
 public:
  FeatureDescriptor(int64 slot_num,
                    bool need_record_freq,
                    bool need_record_version) {
    slot_infos_.resize(slot_num);
    for (int i = 0; i < slot_infos_.size(); i++) {
      slot_infos_[i].embedding_offset = EMPTY_OFFSET_VALUE;
    }

    if (!need_record_freq) {
      freq_desc_.reset(new NonFreqDescriptor());
    }
    if (!need_record_version) {
      version_desc_.reset(new NonVersionDescriptor());
    }                
  }

  FeatureDescriptor(FeatureDescriptor<V>* feat_desc) {
    slot_infos_ = feat_desc->slot_infos_;
    freq_desc_.reset(
        feat_desc->freq_desc_->Clone());
    version_desc_.reset(
        feat_desc->version_desc_->Clone());
  }

  virtual ~FeatureDescriptor() {}

  virtual void InitSlotInfo(int emb_index, int64 embedding_dim,
      const std::pair<V*, int64>& default_value) = 0;
  virtual V* GetEmbedding(void *val, int emb_index) = 0;
  virtual void* Allocate() = 0;
  virtual void Deallocate(void* val) = 0;
  virtual void SetAllocator(Allocator* alloc) = 0;
  virtual void SetDefaultValue(void* val, int64 key) = 0;
  virtual int data_bytes() = 0;

  inline int64 GetFreq(void* val) {
    return freq_desc_->GetFreq(val);
  }

  inline int64 GetVersion(void* val) {
    return version_desc_->GetVersion(val);
  }

  inline void SetFreq(void* val, int64 freq) {
    freq_desc_->SetFreq(val, freq);
  }

  inline void UpdateVersion(void* val, int64 version) {
    version_desc_->UpdateVersion(val, version);
  }

  inline void AddFreq(void* val, int64 freq) {
    freq_desc_->AddFreq(val, freq);
  }

  inline int total_dim() {
    int64 slot_num = slot_infos_.size();
    return slot_infos_[-1].embedding_offset + slot_infos_[-1].embedding_dim;
  }

 protected:
  bool SetEmbeddingInfo(int emb_index, int64 embedding_dim,
                    const std::pair<V*, int64>& default_value) {
    slot_infos_[emb_index].default_value = default_value.first;
    slot_infos_[emb_index].default_value_dim = default_value.second;
    slot_infos_[emb_index].default_value_len = embedding_dim;

    bool is_aligned = true;
    TF_CHECK_OK(ReadBoolFromEnvVar("EV_DATA_ALIGNED", true,
        &is_aligned));
    if (is_aligned) {
      embedding_dim = ComputeAlignedDim(embedding_dim);
    }

    //Avoid parallel consitency issue
    __sync_bool_compare_and_swap(
        &slot_infos_[emb_index].embedding_offset,
        EMPTY_OFFSET_VALUE, embedding_dim);
    slot_infos_[emb_index].embedding_dim = embedding_dim;
    //Check whether all offsets are set
    for (int i = 0; i < slot_infos_.size(); i++) {
      if (slot_infos_[i].embedding_offset == EMPTY_OFFSET_VALUE) {
        return false;
      }
    }

    ComputeEmbeddingOffsets();
    return true;
  }

  void ComputeAllocBytes(int* alloc_bytes) {
    for(auto slot_info: slot_infos_) {
      *alloc_bytes += slot_info.embedding_dim * sizeof(V);
    }
  }

  void CreateFreqAndVersionDescriptor(int* alloc_bytes) {
    if (!freq_desc_) {
      freq_desc_.reset(new FreqDescriptor(*alloc_bytes));
      *alloc_bytes += sizeof(int64);
    }
    if (!version_desc_) {
      version_desc_.reset(new VersionDescriptor(*alloc_bytes));
      *alloc_bytes += sizeof(int64);
    }
  }

  void InitFreqAndVersion(void* val) {
    freq_desc_->SetFreq(val, 0);
    version_desc_->UpdateVersion(val, -1);
  }

  V* GetDefaultValuePtr(int64 emb_index, int64 key) {
    V* default_value_base = (V*)slot_infos_[emb_index].default_value;
    int64 default_value_offset =
        (key % slot_infos_[emb_index].default_value_dim) *
        slot_infos_[emb_index].default_value_len;
    return default_value_base + default_value_offset;
  }

  void SetDefaultValue(void* val, int64 emb_index, int64 key) {
    memcpy(val,
           GetDefaultValuePtr(emb_index, key),
           slot_infos_[emb_index].default_value_len * sizeof(V));
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

  void ComputeEmbeddingOffsets() {
    for (int i = slot_infos_.size() - 1 ; i >= 0; i--) {
      slot_infos_[i].embedding_offset = 0;
      for (int j = 0; j < i; j++) {
        slot_infos_[i].embedding_offset += slot_infos_[j].embedding_offset;
      }
    }
  }

 protected:
  const int EMPTY_OFFSET_VALUE= -1;
  const int ALIGN_BYTES = 16;
  std::vector<SlotInfo> slot_infos_;
  std::unique_ptr<BaseFreqDescriptor> freq_desc_;
  std::unique_ptr<BaseVersionDescriptor> version_desc_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_