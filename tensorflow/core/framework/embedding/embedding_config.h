/* Copyright 2019 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_

#include <cmath>
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/shrink_config.h"
#include "tensorflow/core/framework/embedding/filter_config.h"


namespace tensorflow {
class IndexConfig {
 public:
  IndexConfig(int emb_index, int primary_emb_index)
      :emb_index_(emb_index), primary_emb_index_(primary_emb_index) {}
  ~IndexConfig() {}

  bool IsPrimary() const {
    return (emb_index_ == primary_emb_index_);
  }

  int emb_index() const {
    return emb_index_; 
  }

  int primary_emb_index() const {
    return primary_emb_index_;
  }
 private:
  int emb_index_;
  int primary_emb_index_;
};

class LayoutConfig {
 public:
  LayoutConfig(int block_num, int slot_num, const std::string& layout)
      :block_num_(block_num), slot_num_(slot_num), normal_fix_flag_(0) {
    if (layout == "normal_contiguous" ||
        layout == "normal_contiguous_gpu") {
      normal_fix_flag_ = 1;
    }
  }
  ~LayoutConfig() {}

  int block_num() const {
    return block_num_;
  }

  int slot_num() const {
    return slot_num_;
  }

  int total_slot_num() {
    return block_num_ * (1 + slot_num_);
  }

  int64 total_alloc_num(int64 alloc_len) {
    return block_num_ *
        (1 + (1 - normal_fix_flag_) * slot_num_) *
        (1 + normal_fix_flag_ * (alloc_len * (slot_num_ + 1) - 1));
  }

 private:
  int block_num_;
  int slot_num_;
  int normal_fix_flag_;
};

class RecordConfig {
 public:
  RecordConfig(bool is_record_freq, bool is_record_version)
      :is_record_freq_(is_record_freq),
       is_record_version_(is_record_version) {}
  ~RecordConfig() {}
  
  bool is_record_freq() const {
    return is_record_freq_;
  }

  bool is_record_version() const {
    return is_record_version_;
  }
  
 private:
  bool is_record_freq_;
  bool is_record_version_;
}

class EmbeddingConfig {
 public:
  EmbeddingConfig(int64 emb_index,
                  int64 primary_emb_index,
                  int64 block_num,
                  int slot_num,
                  const std::string& name = "",
                  int64 steps_to_live = 0,
                  int64 filter_freq = 0,
                  int64  = 999999,
                  float l2_weight_threshold = -1.0,
                  const std::string& layout = "normal",
                  int64 max_element_size = 0,
                  float false_positive_probability = -1.0,
                  DataType counter_type = DT_UINT64,
                  int64 default_value_dim = 4096,
                  float default_value_no_permission = .0,
                  bool record_freq =false,
                  bool record_version=false)
      :name(name),
       default_value_dim(default_value_dim),
       index_config_(emb_index, primary_emb_index),
       layout_config_(block_num, slot_num, layout),
       record_config_(record_freq, record_version) {
    if (filter_freq != 0) {
      if (max_element_size == 0 || false_positive_probability == -1.0) {
        filter_config_ = 
            CounterFilterConfig(filter_freq,
                                max_freq,
                                default_value_no_permission);
      } else {
        filter_config_ = 
            BloomFilterConfig(filter_freq,
                              max_freq,
                              max_element_size,
                              false_positive_probability,
                              default_value_no_permission,
                              counter_type);
      }
    } else {
      filter_config_ = NotFilterConfig();
    }


    if (steps_to_live != 0) {
      shrink_config_ = GlobalStepPolicyConfig(steps_to_live);
    } else if (l2_weight_threshold != -1.0) {
      shrink_config_ = L2WeightPolicyConfig(l2_weight_threshold);
    } else {
      shrink_config_ = NotShrinkConfig();
    }
  }

  const IndexConfig& index_config() {
    return index_config_;
  }

  const LayoutConfig& layout_config() {
    return layout_config_;
  }

  const RecordConfig& record_config() {
    return record_config_;
  }

  const FilterConfig& filter_config() {
    return filter_config_;
  }

  const ShrinkConfig&  shrink_config() {
    return shrink_config_;
  }

  int64 default_value_dim() {
    return default_value_dim_;
  }

  std::string DebugString() const {
    return strings::StrCat("opname: ", name,
                           " emb_index: ", index_config_.emb_index(),
                           " primary_emb_index: ",
                           index_config_.primary_emb_index(),
                           " block_num: ", layout_config_.block_num(),
                           " slot_num: ", layout_config_.slot_num());
  }
 private:
  std::string name;
  int64 default_value_dim_;
  IndexConfig index_config_;
  LayoutConfig layout_config_;
  ShrinkConfig shrink_config_;
  FilterConfig filter_config_;
  RecordConfig record_config_;  
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_

