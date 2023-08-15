#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_

#include <cmath>
#include "tensorflow/core/framework/embedding/config.pb.h"

namespace tensorflow {
struct EmbeddingConfig {
  int64 emb_index;
  int64 primary_emb_index;
  int64 block_num;
  int64 slot_num;
  std::string name;
  int64 steps_to_live;
  int64 filter_freq;
  int64 max_freq;
  float l2_weight_threshold;
  int64 default_value_dim;
  float default_value_no_permission;
  bool record_freq;
  bool record_version;
  bool is_inference;

  EmbeddingConfig(int64 emb_index = 0,
                  int64 primary_emb_index = 0,
                  int64 block_num = 1,
                  int slot_num = 0,
                  const std::string& name = "",
                  int64 steps_to_live = 0,
                  int64 filter_freq = 0,
                  int64 max_freq = 999999,
                  float l2_weight_threshold = -1.0,
                  int64 default_value_dim = 4096,
                  float default_value_no_permission = .0,
                  bool record_freq =false,
                  bool record_version=false,
                  bool is_inference=false):
      emb_index(emb_index),
      primary_emb_index(primary_emb_index),
      block_num(block_num),
      slot_num(slot_num),
      name(name),
      steps_to_live(steps_to_live),
      filter_freq(filter_freq),
      max_freq(max_freq),
      l2_weight_threshold(l2_weight_threshold),
      default_value_dim(default_value_dim),
      default_value_no_permission(default_value_no_permission),
      record_freq(record_freq),
      record_version(record_version),
      is_inference(is_inference) {}

  bool is_counter_filter(){
    return filter_freq != 0;
  }

  bool is_primary() const {
    return emb_index == primary_emb_index;
  }

  bool is_save_freq() const {
    return filter_freq != 0 || record_freq;
  }

  bool is_save_version() const {
    return steps_to_live != 0 || record_version;
  }

  int64 get_filter_freq() {
    return filter_freq;
  }

  std::string DebugString() const {
    return strings::StrCat("opname: ", name,
                           " emb_index: ", emb_index,
                           " primary_emb_index: ", primary_emb_index,
                           " block_num: ", block_num,
                           " slot_num: ", slot_num,
                           " steps_to_live: ", steps_to_live,
                           " filter_freq: ", filter_freq,
                           " max_freq: ", max_freq,
                           " l2_weight_threshold: ", l2_weight_threshold);
  }
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_

