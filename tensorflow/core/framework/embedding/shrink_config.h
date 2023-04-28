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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_CONFIG_H_
namespace tensorflow {
class ShrinkConfig {
 public:
  ShrinkConfig() {}
  ShrinkConfig(int64 steps_to_live) {}
  ShrinkConfig(float l2_weight_threshold) {}

  virtual ~ShrinkConfig() {}

  virtual int64 steps_to_live() = 0;
  virtual float l2_weight_threshold() = 0;
  virtual bool is_based_on_global_step() = 0;
  virtual bool is_based_on_l2_weight() = 0;
};

class GlobalStepPolicyConfig: public ShrinkConfig {
 public:
  GlobalStepPolicyConfig(int64 steps_to_live)
      :steps_to_live_(steps_to_live) {}
  
  int64 steps_to_live() const override {
    return steps_to_live_;
  }

  float l2_weight_threshold() const override {
    LOG(FATAL)<<"GlobalStep Shrink Policy is used,"
              <<" can't get the l2_weight_threshold.";
    return -1.0;
  }

  bool is_based_on_global_step() override {
    return true;
  }

  bool is_based_on_l2_weight() override {
    return false;
  }
 private:
  int64 steps_to_live_;
};

class L2WeightPolicyConfig: public ShrinkConfig {
 public:
  L2WeightPolicyConfig(float l2_weight_threshold)
      :l2_weight_threshold_(l2_weight_threshold) {}
  
  int64 steps_to_live() const override {
     LOG(FATAL)<<"L2Weight Shrink Policy is used,"
              <<" can't get the steps_to_live.";
    return 0;
  }

  float l2_weight_threshold() const override {
    return l2_weight_threshold_;
  }

  bool is_based_on_global_step() override {
    return false;
  }

  bool is_based_on_l2_weight() override {
    return true;
  }
 private:
  float l2_weight_threshold_;
};

class NotShrinkConfig: public ShrinkConfig {
 public:
  NotShrinkConifg() {}

  int64 steps_to_live() const override {
    LOG(FATAL)<<"Feature Eviction is not Enabled.";
    return 0;
  }

  float l2_weight_threshold() const override {
    LOG(FATAL)<<"Feature Eviction is not Enabled.";
    return -1.0;
  }

  bool is_based_on_global_step() override {
    return false;
  }

  bool is_based_on_l2_weight() override {
    return false;
  }
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_CONFIG_H_

