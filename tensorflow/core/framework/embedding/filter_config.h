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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_CONFIG_H_

#include <cmath>
#include "tensorflow/core/framework/embedding/config.pb.h"

namespace tensorflow {
class FilterConfig {
 public:
  FilterConfig() {}
	FilterConfig(int64 filter_freq,
	             int64 max_freq,
							 float default_value_no_permmsion) {}
	virtual ~FilterConfig() {}

	virtual int64 filter_freq() = 0;
	virtual float default_value_no_permission() = 0;
	virtual bool is_has_filter() = 0;
	virtual bool is_counter_filter() = 0;
};

class CounterFilterConfig: public FilterConfig {
 public:
  FilterConfig(int64 filter_freq,
	             int64 max_freq,
							 float default_value_no_permmsion)
		:filter_freq_(filter_freq),
		 max_freq_(max_freq),
		 default_value_no_permmsion_(default_value_no_permmsion) {}
	
	int64 filter_freq() const override {
		return filter_freq_;
	}

	float default_value_no_permission() const override {
		return default_value_no_permission_;
	}

	bool is_has_filter() override {
		return true;
	}

	bool is_counter_filter() override {
		return true;
	}
		 
 private:
  int64 filter_freq_;
	int64 max_freq_;
	float default_value_no_permission_;
};

class BloomFilterConfig: public FilterConfig {
 public:
  BloomFilterConfig(int64 filter_freq,
	                  int64 max_freq,
							      int64 max_element_size,
                    float false_positive_probability,
							      float default_value_no_permmsion,
							      DataType counter_type)
		  :filter_freq_(filter_freq),
		   max_freq_(max_freq),
			 kHashFunc_(0),
			 num_counter_(0),
		   default_value_no_permmsion_(default_value_no_permmsion),
		   counter_type_(counter_type) {
    kHashFunc_ = calc_num_hash_func(false_positive_probability);
    num_counter_ = calc_num_counter(
			  max_element_size, false_positive_probability);
	}
	~BloomFilterConfig() {}

	int64 filter_freq() const override {
		return filter_freq_;
	}

	float default_value_no_permission() const override {
		return default_value_no_permission_;
	}

	bool is_has_filter() override {
		return true;
	}

	bool is_counter_filter() override {
		return false;
	}

 private:

  int64 calc_num_hash_func(float false_positive_probability) {
    float loghpp = fabs(log(false_positive_probability)/log(2));
    return ceil(loghpp);
  }

	int64 calc_num_counter(int64 max_element_size,
                         float false_positive_probability) {
    float loghpp = fabs(log(false_positive_probability));
    float factor = log(2) * log(2);
    int64 num_bucket = ceil(loghpp / factor * max_element_size);
    if (num_bucket * sizeof(counter_type) > 10 * (1L << 30))
      LOG(WARNING)<<"The Size of BloomFilter is more than 10GB!";
    return num_bucket;
  }

  int64 filter_freq_;
	int64 max_freq_;
	int64 kHashFunc_;
  int64 num_counter_;
	float default_value_no_permission_;
	DataType counter_type_;
};

class NotFilterConfig: public FilterConfig {
 public:
  NotFilterConfig() {}
	~NotFilterConfig() {}

	int64 filter_freq() const override {
		LOG(FATAL)<<"Feature Filter is not enabled.";
		return -1;
	}

	float default_value_no_permission() const override {
		LOG(FATAL)<<"Feature Filter is not enabled.";
		return -1.0;
	}

	bool is_has_filter() override {
		return false;
	}

	bool is_counter_filter() override {
		return false;
	}
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_CONFIG_H_