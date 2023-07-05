/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_
#define TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/cache_factory.h"
#include "tensorflow/core/framework/embedding/feature_descriptor_factory.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template<class T>
class EVKeyDumpIterator: public  DumpIterator<T> {
 public:
  EVKeyDumpIterator(std::vector<T>& key_list):key_list_(key_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < key_list_.size();
  }

  T Next() {
    return key_list_[keys_idx_++];
  }

 private:
  int64 keys_idx_;
  std::vector<T>& key_list_;
};

template<class K, class T>
class EVValueDumpIterator: public  DumpIterator<T> {
 public:
  EVValueDumpIterator(EmbeddingVar<K, T>*& ev,
      std::vector<T* >& valueptr_list)
        : ev_(ev),
          valueptr_list_(valueptr_list) {
    keys_idx_ = 0;
    col_idx_ = 0;
  }

  bool HasNext() const {
    if (keys_idx_ < valueptr_list_.size()) {
      if (keys_idx_ < valueptr_list_.size() - 1)
        return true;
      else
        return col_idx_ < ev_->ValueLen();
    } else
      return false;
  }

  T Next() {
    if (col_idx_ >= ev_->ValueLen()) {
      keys_idx_++;
      col_idx_ = 0;
    }
    Eigen::array<Eigen::DenseIndex, 1> dims({ev_->ValueLen()});
    typename TTypes<T>::Flat value_flat =
      typename TTypes<T>::Flat(valueptr_list_[keys_idx_], dims);
    return value_flat(col_idx_++);
  }

 private:
  EmbeddingVar<K, T>* ev_;
  std::vector<T* >& valueptr_list_;
  int64 keys_idx_;
  int64 col_idx_;
};

template<class T>
class EVVersionDumpIterator: public  DumpIterator<T> {
 public:
  EVVersionDumpIterator(std::vector<T >& version_list)
      : version_list_(version_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < version_list_.size();
  }

  T Next() {
    return version_list_[keys_idx_++];
  }

 private:
  std::vector<T>& version_list_;
  int64 keys_idx_;
};

template<class T>
class EVFreqDumpIterator: public  DumpIterator<T> {
 public:
  EVFreqDumpIterator(std::vector<T>& freq_list) : freq_list_(freq_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < freq_list_.size();
  }

  T Next() {
    return freq_list_[keys_idx_++];
  }

 private:
  std::vector<T>& freq_list_;
  int64 keys_idx_;
};

template<class T>
class EVOffsetDumpIterator: public  DumpIterator<T> {
 public:
  EVOffsetDumpIterator(std::vector<T>& offset_list)
      : offset_list_(offset_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < offset_list_.size();
  }

  T Next() {
    return offset_list_[keys_idx_++];
  }

 private:
  std::vector<T>& offset_list_;
  int64 keys_idx_;
};

template <class K, class V>
Status GetInputEmbeddingVar(OpKernelContext* ctx, int input,
                            EmbeddingVar<K, V>** var) {
  if (LookupResource(ctx, HandleFromInput(ctx, input), var).ok()) {
    return Status::OK();
  } else {
    return errors::Internal("Invalid versioned variable reference.");
  }
}

namespace {
const static string part_str = "part_";
}

template<typename K, typename V>
Status DynamicRestoreValue(EmbeddingVar<K, V>* ev, BundleReader* reader,
    std::string name_string, int orig_partnum, const GPUDevice* device,
    int64 partition_id = 0, int64 partition_num = 1, bool reset_version = false) {
  string curr_partid_str = std::to_string(partition_id);
  bool filter_flag = true;
  embedding::BatchCache<K>* cache_for_restore_hbm = nullptr;
  if (ev->IsMultiLevel() && ev->IsUseHbm()) {
    auto cache_strategy = ev->storage()->CacheStrategy();
    cache_for_restore_hbm = embedding::CacheFactory::Create<K>(
        cache_strategy, "hbm_restore_cache for " + name_string);
  }
  for (int i = 0; i < orig_partnum; i++) {
    string part_id = std::to_string(i);
    string pre_subname =
      name_string.substr(0, name_string.find("part_"));
    string post_subname =
      name_string.substr(name_string.find("part_")
          + part_str.size() + curr_partid_str.size());
    string tensor_name =
      pre_subname + part_str + part_id + post_subname;

    string tensor_key = tensor_name + "-keys";
    string tensor_value = tensor_name + "-values";
    string tensor_version = tensor_name + "-versions";
    string tensor_freq = tensor_name + "-freqs";
    
    TensorShape key_shape, value_shape, version_shape, freq_shape;
    Status st = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupTensorShape(tensor_value, &value_shape);
    if (!st.ok()) {
      return st;
    }
    if (!reset_version) {
      st = reader->LookupTensorShape(tensor_version, &version_shape);
      if (!st.ok()) {
        return st;
      }
    }

    st = reader->LookupTensorShape(tensor_freq, &freq_shape);
    if (!st.ok()) {
      if (st.code() == error::NOT_FOUND) {
        freq_shape = version_shape;
      }else {
        return st;
      }
    }

    st = reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupHeader(tensor_value,
        sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
    if (!st.ok()) {
      return st;
    }
    if (!reset_version) {
      st = reader->LookupHeader(tensor_version,
          sizeof(int64) * version_shape.dim_size(0));
      if (!st.ok()) {
        return st;
      }
    }
    st = reader->LookupHeader(tensor_freq,
        sizeof(int64) * freq_shape.dim_size(0));
    if (!st.ok()) {
      if (st.code() == error::NOT_FOUND) {
        filter_flag = false;
      }else {
        return st;
      }
    }

    size_t buffer_size = 8 << 20;
    RestoreBuffer restore_buff;
    restore_buff.key_buffer = new char[buffer_size];
    restore_buff.value_buffer = new char[buffer_size];
    restore_buff.version_buffer = new char[buffer_size];
    restore_buff.freq_buffer = new char[buffer_size];
    int64 newDim = ev->ValueLen();
    size_t value_unit_bytes_new = sizeof(V) * newDim;
    int64 idx = 0;
    bool restore_customDim;
    TF_CHECK_OK(ReadBoolFromEnvVar(
                                  "TF_EV_RESTORE_CUSTOM_DIM", false, &restore_customDim));
    size_t key_bytes_read = 0;
    size_t value_bytes_read = 0;
    size_t version_bytes_read = 0;
    size_t freq_bytes_read = 0;
    int64 tot_key_num = key_shape.dim_size(0);
    size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);

    while(tot_key_num > 0) {
      size_t read_key_num = std::min(std::min(buffer_size / sizeof(K),
            buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
      read_key_num = std::min(read_key_num, buffer_size / value_unit_bytes_new);
      read_key_num = std::min((int64)read_key_num, tot_key_num);
      reader->LookupSegment(tensor_key, read_key_num * sizeof(K),
          restore_buff.key_buffer, key_bytes_read);
      reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes,
          restore_buff.value_buffer, value_bytes_read);
      if (!reset_version) {
        reader->LookupSegment(tensor_version, read_key_num * sizeof(int64),
            restore_buff.version_buffer, version_bytes_read);
        if (version_bytes_read == 0) {
          memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
        }
      } else {
        int64 *version_tmp = (int64*)restore_buff.version_buffer;
        memset(version_tmp, 0, read_key_num * sizeof(int64));
      }
      if (filter_flag) {
        reader->LookupSegment(tensor_freq, (read_key_num + 1)* sizeof(int64),
            restore_buff.freq_buffer, freq_bytes_read);
        if (freq_bytes_read == 0) {
          int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
          for (int64 i = 0; i < read_key_num; i++) {
            freq_tmp[i] = (ev->MinFreq() == 0) ? 1 : ev->MinFreq();
          }
        }
      }else {
        int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
        for (int64 i = 0; i < read_key_num; i++) {
          freq_tmp[i] = (ev->MinFreq() == 0) ? 1 : ev->MinFreq();
        }
      }

      if (key_bytes_read > 0) {
        read_key_num = key_bytes_read / sizeof(K);
        VLOG(2) << "repartition, read_key_num:" << read_key_num;
        if (restore_customDim && value_shape.dim_size(1) != newDim) {
          VLOG(2) << "restore, read_value_reshape dim: from "
                  << value_shape.dim_size(1) << " to " << newDim;
          if (read_key_num * value_unit_bytes != value_bytes_read) {
            return tensorflow::errors::FailedPrecondition(
                "Expected read_key_num * value_unit_bytes == value_bytes_read, "
                "but got read_key_num * value_unit_bytes != value_bytes_read!");
          }

	  std::unique_ptr<char[]> tmp_ptr(new char[buffer_size]);
          size_t read_once = std::min(value_unit_bytes, value_unit_bytes_new);
          for (int i = 0; i < read_key_num; ++i) {
            memcpy(tmp_ptr.get() + i * value_unit_bytes_new,
                   restore_buff.value_buffer + i * value_unit_bytes, read_once);
            if (value_shape.dim_size(1) >= newDim) continue;
            auto p = ev->GetDefaultValue(idx);
            ++idx;
            memcpy(tmp_ptr.get() + i * value_unit_bytes_new +
                       value_unit_bytes,
                   p + value_unit_bytes,
                   value_unit_bytes_new - value_unit_bytes);
          }
          auto tmp = tmp_ptr.release();
          tmp_ptr.reset(restore_buff.value_buffer);
          restore_buff.value_buffer = tmp;
        }
        st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
            partition_id, partition_num, false, device);
        if (cache_for_restore_hbm) {
          cache_for_restore_hbm->update(
              (K*)restore_buff.key_buffer, read_key_num,
              (int64*)restore_buff.version_buffer,
              (int64*)restore_buff.freq_buffer);
        }
        if (!st.ok()) {
          return st;
        }
        tot_key_num -= read_key_num;
      }
    }
  }
  if (cache_for_restore_hbm) {
    int64 cache_capacity = ev->CacheSize();
    int64 num_of_hbm_ids =
        std::min(cache_capacity, (int64)cache_for_restore_hbm->size());
    K* hbm_ids = new K[num_of_hbm_ids];
    int64* hbm_freqs = new int64[num_of_hbm_ids];
    int64* hbm_versions = nullptr;
    cache_for_restore_hbm->get_cached_ids(
        hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs);
    ev->ImportToHbm(hbm_ids, num_of_hbm_ids);
    ev->storage()->Schedule([ev, hbm_ids, num_of_hbm_ids,
                                     hbm_versions, hbm_freqs]() {
      embedding::BatchCache<K>* cache = ev->Cache();
      cache->update(hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs);
      delete[] hbm_ids;
      delete[] hbm_freqs;
    });
    delete cache_for_restore_hbm;
  }
  return Status::OK();
}

template<typename K, typename V>
Status EVRestoreNoPartition(EmbeddingVar<K, V>* ev, BundleReader* reader,
    std::string tensor_key, std::string tensor_value,
    std::string tensor_version, std::string tensor_freq,
    const GPUDevice* device,
    bool reset_version=false) {
  TensorShape key_shape;
  TensorShape value_shape;
  TensorShape version_shape;
  TensorShape freq_shape;
  TensorShape key_filter_shape;
  TensorShape version_filter_shape;
  TensorShape freq_filter_shape;

  Status st;
  reader->LookupTensorShape(tensor_key, &key_shape);
  reader->LookupTensorShape(tensor_value, &value_shape);
  reader->LookupTensorShape(tensor_version, &version_shape);
  st = reader->LookupTensorShape(tensor_freq, &freq_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      freq_shape = version_shape;
    }else {
      return st;
    }
  }
  st = reader->LookupTensorShape(tensor_key + "_filtered", &key_filter_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      key_filter_shape = key_shape;
    }else {
      return st;
    }
  }
  st = reader->LookupTensorShape(tensor_version + "_filtered",
      &version_filter_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      version_filter_shape = version_shape;
    }else {
      return st;
    }
  }
  st = reader->LookupTensorShape(tensor_freq + "_filtered",
      &freq_filter_shape);
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      freq_filter_shape = freq_shape;
    }else {
      return st;
    }
  }

  bool filter_flag = true;
  bool restore_filter_flag = true;
  st = reader->LookupHeader(tensor_key,
      sizeof(K) * key_shape.dim_size(0));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_value,
      sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_version,
      sizeof(int64) * version_shape.dim_size(0));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_freq,
      sizeof(int64) * freq_shape.dim_size(0));
  if (!st.ok()) {
    if (st.code() == error::NOT_FOUND) {
      filter_flag = false;
    }else {
      return st;
    }
  }
  st = reader->LookupHeader(tensor_key + "_filtered",
      sizeof(K) * key_filter_shape.dim_size(0));
  if (!st.ok()){
    if (st.code() == error::NOT_FOUND){
      restore_filter_flag=false;
    } else {
      return st;
    }
  }
  st = reader->LookupHeader(tensor_version + "_filtered",
      sizeof(K) * version_filter_shape.dim_size(0));
  if (!st.ok() && st.code() != error::NOT_FOUND){
    return st;
  }
  st = reader->LookupHeader(tensor_freq + "_filtered",
      sizeof(K) * freq_filter_shape.dim_size(0));
  if (!st.ok() && st.code() != error::NOT_FOUND){
    return st;
  }
  embedding::BatchCache<K>* cache_for_restore_hbm = nullptr;
  if (ev->IsMultiLevel() && ev->IsUseHbm()) {
    auto cache_strategy = ev->storage()->CacheStrategy();
    cache_for_restore_hbm = embedding::CacheFactory::Create<K>(
        cache_strategy, "hbm_restore_cache for " + tensor_key);
  }

  size_t buffer_size = 8 << 20;
  RestoreBuffer restore_buff;
  restore_buff.key_buffer = new char[buffer_size];
  restore_buff.value_buffer = new char[buffer_size];
  restore_buff.version_buffer = new char[buffer_size];
  restore_buff.freq_buffer = new char[buffer_size];
  int64 newDim = ev->ValueLen();
  size_t value_unit_bytes_new = sizeof(V) * newDim;
  int64 idx = 0;
  bool restore_customDim;
  TF_CHECK_OK(ReadBoolFromEnvVar(
                                "TF_EV_RESTORE_CUSTOM_DIM", false, &restore_customDim));
  size_t key_bytes_read = 0;
  size_t value_bytes_read = 0;
  size_t version_bytes_read = 0;
  size_t freq_bytes_read = 0;
  size_t key_filter_bytes_read = 0;
  size_t version_filter_bytes_read = 0;
  size_t freq_filter_bytes_read = 0;

  int64 tot_key_num = key_shape.dim_size(0);
  size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
  std::string key_str = "|";
  while(tot_key_num > 0) {
    size_t read_key_num = std::min(
        std::min(buffer_size / sizeof(K),
          buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
    read_key_num = std::min(read_key_num, buffer_size / value_unit_bytes_new);
    read_key_num = std::min((int64)read_key_num, tot_key_num);
    reader->LookupSegment(tensor_key, read_key_num * sizeof(K),
        restore_buff.key_buffer, key_bytes_read);
    reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes,
        restore_buff.value_buffer, value_bytes_read);
    if (!reset_version) {
      reader->LookupSegment(tensor_version, read_key_num * sizeof(int64),
          restore_buff.version_buffer, version_bytes_read);
      if (version_bytes_read == 0) {
          memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
      }
    } else {
      int64 *version_tmp = (int64*)restore_buff.version_buffer;
      memset(version_tmp, 0, read_key_num * sizeof(int64));
    }
    if (filter_flag) {
      reader->LookupSegment(tensor_freq, read_key_num * sizeof(int64),
          restore_buff.freq_buffer, freq_bytes_read);
      if (freq_bytes_read == 0) {
        int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
        for (int64 i = 0; i < read_key_num; i++) {
          freq_tmp[i] = (ev->MinFreq() == 0) ? 1 : ev->MinFreq();
        }
      }
    } else {
      int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
      for (int64 i = 0; i < read_key_num; i++) {
        freq_tmp[i] = (ev->MinFreq() == 0) ? 1 : ev->MinFreq();
      }
    }
    if (key_bytes_read > 0) {
      read_key_num = key_bytes_read / sizeof(K);
      VLOG(2) << "restore, read_key_num:" << read_key_num;

      if (restore_customDim && value_shape.dim_size(1) != newDim) {
        VLOG(2) << "restore, read_value_reshape dim: from "
                << value_shape.dim_size(1) << " to " << newDim;
        if (read_key_num * value_unit_bytes != value_bytes_read) {
          return tensorflow::errors::FailedPrecondition(
              "Expected read_key_num * value_unit_bytes == value_bytes_read, "
              "but got read_key_num * value_unit_bytes != value_bytes_read!");
        }

        std::unique_ptr<char[]> tmp_ptr(new char[buffer_size]);
        size_t read_once = std::min(value_unit_bytes, value_unit_bytes_new);
        for (int i = 0; i < read_key_num; ++i) {
          memcpy(tmp_ptr.get() + i * value_unit_bytes_new,
                 restore_buff.value_buffer + i * value_unit_bytes, read_once);
          if (value_shape.dim_size(1) >= newDim) continue;
          auto p = ev->GetDefaultValue(idx);
          ++idx;
          memcpy(tmp_ptr.get() + i * value_unit_bytes_new +
                     value_unit_bytes,
                 p + value_unit_bytes, value_unit_bytes_new - value_unit_bytes);
        }
        auto tmp = tmp_ptr.release();
        tmp_ptr.reset(restore_buff.value_buffer);
        restore_buff.value_buffer = tmp;
      }
      st = ev->Import(restore_buff, read_key_num, 1, 0, 1, false, device);
      if (cache_for_restore_hbm) {
        cache_for_restore_hbm->update(
            (K*)restore_buff.key_buffer, read_key_num,
            (int64*)restore_buff.version_buffer,
            (int64*)restore_buff.freq_buffer);
      }
      if (!st.ok())
        return st;
      tot_key_num -= read_key_num;
    }
  }
  
  if (restore_filter_flag) {
    int64 tot_key_filter_num = key_filter_shape.dim_size(0);
    while (tot_key_filter_num > 0) {
      size_t read_key_num = std::min(buffer_size / sizeof(K),
          buffer_size / sizeof(int64));
      read_key_num = std::min((int64)read_key_num, tot_key_filter_num);
      reader->LookupSegment(tensor_key + "_filtered",
          read_key_num * sizeof(K), restore_buff.key_buffer,
          key_filter_bytes_read);
      if (!reset_version) {
        reader->LookupSegment(tensor_version + "_filtered",
            read_key_num * sizeof(int64), restore_buff.version_buffer,
            version_filter_bytes_read);
      } else {
        int64 *version_tmp = (int64*)restore_buff.version_buffer;
        memset(version_tmp, 0, read_key_num * sizeof(int64));
      }
      reader->LookupSegment(tensor_freq + "_filtered",
          read_key_num * sizeof(int64), restore_buff.freq_buffer,
          freq_filter_bytes_read);
      if (key_filter_bytes_read > 0) {
        read_key_num = key_filter_bytes_read / sizeof(K);
        VLOG(2) << "restore, read_key_num:" << read_key_num;

        st = ev->Import(restore_buff, read_key_num, 1, 0, 1, true, device);
        if (cache_for_restore_hbm) {
          cache_for_restore_hbm->update(
              (K*)restore_buff.key_buffer, read_key_num,
              (int64*)restore_buff.version_buffer,
              (int64*)restore_buff.freq_buffer);
        }
        if (!st.ok())
          return st;
        tot_key_filter_num -= read_key_num;
      }
    }
  }

  if (cache_for_restore_hbm) {
    int64 cache_capacity = ev->CacheSize();
    int64 num_of_hbm_ids =
        std::min(cache_capacity, (int64)cache_for_restore_hbm->size());
    K* hbm_ids = new K[num_of_hbm_ids];
    int64* hbm_freqs = new int64[num_of_hbm_ids];
    int64* hbm_versions = nullptr;
    cache_for_restore_hbm->get_cached_ids(
        hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs);
    ev->ImportToHbm(hbm_ids, num_of_hbm_ids);
    ev->storage()->Schedule([ev, hbm_ids, num_of_hbm_ids,
                                     hbm_versions, hbm_freqs]() {
      embedding::BatchCache<K>* cache = ev->Cache();
      cache->update(hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs);
      delete[] hbm_ids;
      delete[] hbm_freqs;
    });
    delete cache_for_restore_hbm;
  }

  return Status::OK();
}

inline bool IsOldCheckpoint(const std::string& name_string,
    const std::string& curr_partid_str, BundleReader* reader,
    const std::string& part_offset_tensor_suffix) {
  // then check whether checkpoint is in old form
  bool is_oldform = false;

  string part_id = std::to_string(0);
  string pre_subname =
    name_string.substr(0, name_string.find(part_str));
  string post_subname = name_string.substr(
      name_string.find(part_str) + part_str.size() + curr_partid_str.size());
  string tensor_name = pre_subname + part_str + part_id + post_subname;

  TensorShape part_offset_shape;
  DataType part_offset_type;
  Status form_st = reader->LookupDtypeAndShape(
      tensor_name + part_offset_tensor_suffix,
      &part_offset_type, &part_offset_shape);
  if (!form_st.ok()) {
    is_oldform = true;
  }
  return is_oldform;
}

template<typename K, typename V>
Status EVRestoreOldFromCheckpoint(EmbeddingVar<K, V>* ev,
    const std::string& name_string, const std::string& curr_partid_str,
    const std::string& key_suffix, int partition_id,
    BundleReader* reader, int partition_num,
    const GPUDevice* device, bool reset_version=false) {
  // first get original partition number
  int orig_partnum = 0;
  for (;  ; orig_partnum++) {
    string part_id = std::to_string(orig_partnum);
    string pre_subname = name_string.substr(0, name_string.find(part_str));
    string post_subname = name_string.substr(name_string.find(part_str)
        + part_str.size() + curr_partid_str.size());
    string tensor_name = pre_subname + part_str + part_id + post_subname;

    string tensor_key = tensor_name + key_suffix;
    TensorShape key_shape;
    Status st = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!st.ok()) {
      break;
    }
  }

  VLOG(1) << "old form, EV name:" << name_string
          << ", partition_id:" << curr_partid_str
          << ", old partition_num:" << orig_partnum
          << ", new partition num:" << partition_num;
  Status s = DynamicRestoreValue(ev, reader, name_string,
      orig_partnum, device, partition_id, partition_num, reset_version);
  if (!s.ok()) {
    LOG(FATAL) <<  "EV restoring fail:" << s.ToString();
  }
  return s;
}

template<typename K, typename V>
Status EVRestoreDynamically(EmbeddingVar<K, V>* ev,
    const std::string& name_string, int partition_id,
    int partition_num, OpKernelContext* context,
    BundleReader* reader, const std::string& part_offset_tensor_suffix,
    const std::string& key_suffix, const std::string& value_suffix,
    const std::string& version_suffix, const std::string& freq_suffix,
    bool reset_version = false, const Eigen::GpuDevice* device = nullptr) {

  // first check whether there is partition
  if (name_string.find(part_str) == std::string::npos) {
    Status s = EVRestoreNoPartition(
        ev, reader, name_string + key_suffix,
        name_string + value_suffix, name_string + version_suffix,
        name_string + freq_suffix, device, reset_version);
    if (!s.ok()) {
      LOG(FATAL) <<  "EV restoring fail:" << s.ToString();
    }
    return s;
  }

  const string& curr_partid_str = std::to_string(partition_id);
  auto is_oldform = IsOldCheckpoint(name_string, curr_partid_str,
      reader, part_offset_tensor_suffix);

  if (is_oldform) {
    EVRestoreOldFromCheckpoint(ev, name_string, curr_partid_str, key_suffix,
        partition_id, reader, partition_num, device, reset_version);
  } else {
    // first find out which sub parts we should load
    bool filter_flag = true;
    bool restore_filter_flag = true;
    std::vector<int> loaded_parts;
    for (int i = 0; i < kSavedPartitionNum; i++) {
      if (i % partition_num == partition_id) {
        loaded_parts.push_back(i);
      }
    }

    // then we use primary partition number to compose with
    // sub partition number
    VLOG(1) << "new form:" << name_string
            << ", partition_id:" << partition_id
            << ", partition_num:" << partition_num;

    embedding::BatchCache<K>* cache_for_restore_hbm = nullptr;
    if (ev->IsMultiLevel() && ev->IsUseHbm()) {
      auto cache_strategy = ev->storage()->CacheStrategy();
      cache_for_restore_hbm = embedding::CacheFactory::Create<K>(
          cache_strategy, "hbm_restore_cache for " + name_string);
    }

    int orig_partnum = 0;
    size_t buffer_size = 8 << 20;
    RestoreBuffer restore_buff;
    restore_buff.key_buffer = new char[buffer_size];
    restore_buff.value_buffer = new char[buffer_size];
    restore_buff.version_buffer = new char[buffer_size];
    restore_buff.freq_buffer = new char[buffer_size];
    int64 newDim = ev->ValueLen();
    size_t value_unit_bytes_new = sizeof(V) * newDim;
    int64 idx = 0;
    bool restore_customDim;
    TF_CHECK_OK(ReadBoolFromEnvVar(
			          "TF_EV_RESTORE_CUSTOM_DIM", false, &restore_customDim));
    for (;  ; orig_partnum++) {
      string part_id = std::to_string(orig_partnum);
      string pre_subname = name_string.substr(0, name_string.find(part_str));
      string post_subname = name_string.substr(name_string.find(part_str)
          + part_str.size() + curr_partid_str.size());
      string tensor_name = pre_subname + part_str + part_id + post_subname;

      // first check whether is  old ckpt form
      string tensor_key = tensor_name + key_suffix;
      string tensor_value = tensor_name + value_suffix;
      string tensor_version = tensor_name + version_suffix;
      string tensor_freq = tensor_name + freq_suffix;
      TensorShape key_shape, value_shape, version_shape, freq_shape;
      TensorShape key_filter_shape, version_filter_shape, freq_filter_shape;
      Status st = reader->LookupTensorShape(tensor_key, &key_shape);
      if (!st.ok()) {
        VLOG(1) << "ev part " << tensor_key
                << " not exist, reach the end of restoring";
        break;
      }
      st = reader->LookupTensorShape(tensor_value, &value_shape);
      if (!st.ok()) {
        break;
      }
      st = reader->LookupTensorShape(tensor_version, &version_shape);
      if (!st.ok()) {
        break;
      }
      st = reader->LookupTensorShape(tensor_freq, &freq_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          freq_shape = version_shape;
        } else {
          return st;
        }
      }
      st = reader->LookupTensorShape(tensor_key + "_filtered",
          &key_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          key_filter_shape = key_shape;
        } else {
          return st;
        }
      }
      st = reader->LookupTensorShape(tensor_version + "_filtered",
          &version_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          version_filter_shape = version_shape;
        } else {
          return st;
        }
      }
      st = reader->LookupTensorShape(tensor_freq + "_filtered",
          &freq_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          freq_filter_shape = freq_shape;
        }else {
          return st;
        }
      }

      reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
      if (!st.ok()) {
        break;
      }
      st = reader->LookupHeader(tensor_value,
          sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
      if (!st.ok()) {
        break;
      }
      st = reader->LookupHeader(tensor_version,
          sizeof(int64) * version_shape.dim_size(0));
      if (!st.ok()) {
        break;
      }
      st = reader->LookupHeader(tensor_freq,
          sizeof(int64) * freq_shape.dim_size(0));
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          filter_flag = false;
        }else {
          return st;
        }
      }
      st = reader->LookupHeader(tensor_key + "_filtered",
          sizeof(K) * key_filter_shape.dim_size(0));
      if (!st.ok()){
        if (st.code() == error::NOT_FOUND){
          restore_filter_flag=false;
        }else {
          return st;
        }
      }
      st = reader->LookupHeader(tensor_version + "_filtered",
          sizeof(K) * version_filter_shape.dim_size(0));
      if (!st.ok() && st.code() != error::NOT_FOUND){
        return st;
      }
      st = reader->LookupHeader(tensor_freq + "_filtered",
          sizeof(K) * freq_filter_shape.dim_size(0));
      if (!st.ok() && st.code() != error::NOT_FOUND){
        return st;
      }

      TensorShape part_offset_shape, part_filter_offset_shape;
      DataType part_offset_type, part_filter_offset_type;
      string offset_tensor_name = tensor_name + part_offset_tensor_suffix;
      string offset_filter_tensor_name =
          tensor_name + "-partition_filter_offset";
      st = reader->LookupDtypeAndShape(offset_tensor_name,
          &part_offset_type, &part_offset_shape);
      if (!st.ok()) {
          LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      st = reader->LookupDtypeAndShape(offset_filter_tensor_name,
          &part_filter_offset_type, &part_filter_offset_shape);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      Tensor part_offset_tensor(cpu_allocator(),
          part_offset_type, part_offset_shape);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      Tensor part_filter_offset_tensor(cpu_allocator(),
          part_offset_type, part_offset_shape);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      st = reader->Lookup(offset_tensor_name, &part_offset_tensor);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      auto part_offset_flat = part_offset_tensor.flat<int64>();
      st = reader->Lookup(offset_filter_tensor_name, &part_filter_offset_tensor);
      if (!st.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
      }
      auto part_filter_offset_flat = part_filter_offset_tensor.flat<int64>();

      for (size_t i = 0; i < loaded_parts.size(); i++) {
        int subpart_id = loaded_parts[i];
        int subpart_offset = part_offset_flat(subpart_id);

        size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
        int64 tot_key_num = part_offset_flat(subpart_id + 1) - subpart_offset;
        int64 key_part_offset = subpart_offset * sizeof(K);
        int64 value_part_offset = subpart_offset *  value_unit_bytes;
        int64 version_part_offset = subpart_offset * sizeof(int64);
        int64 freq_part_offset = subpart_offset * sizeof(int64);

        VLOG(1) << "dynamically load ev : " << name_string
                << ", subpartid:" << loaded_parts[i]
                << ", subpart_offset:" << subpart_offset
                << ", partition_id:" << partition_id
                << ", partition_num:" << partition_num
                << ", keynum:" << tot_key_num;

        int64 tot_key_bytes_read(0);
        int64 tot_value_bytes_read(0);
        int64 tot_version_bytes_read(0);
        int64 tot_freq_bytes_read(0);
        size_t key_bytes_read = 0;
        size_t value_bytes_read = 0;
        size_t version_bytes_read = 0;
        size_t freq_bytes_read = 0;
        while(tot_key_num > 0) {
          size_t read_key_num = std::min(std::min(buffer_size / sizeof(K),
                buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
          read_key_num = std::min(read_key_num, buffer_size / value_unit_bytes_new);
          read_key_num = std::min((int64)read_key_num, tot_key_num);
          reader->LookupSegmentOffset(tensor_key,
              key_part_offset + tot_key_bytes_read, read_key_num * sizeof(K),
              restore_buff.key_buffer, key_bytes_read);

          reader->LookupSegmentOffset(tensor_value,
              value_part_offset + tot_value_bytes_read,
              read_key_num * value_unit_bytes, restore_buff.value_buffer,
              value_bytes_read);
          if (!reset_version) {
            reader->LookupSegmentOffset(tensor_version,
                version_part_offset + tot_version_bytes_read,
                read_key_num * sizeof(int64), restore_buff.version_buffer,
                version_bytes_read);
            if (version_bytes_read == 0) {
              memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
            }
          } else {
            int64 *version_tmp = (int64*)restore_buff.version_buffer;
            memset(version_tmp, 0, read_key_num * sizeof(int64));
          }
          if (filter_flag) {
            reader->LookupSegmentOffset(tensor_freq,
                freq_part_offset + tot_freq_bytes_read,
                read_key_num * sizeof(int64), restore_buff.freq_buffer,
                freq_bytes_read);
            if (freq_bytes_read == 0) {
              int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
              for (int64 i = 0; i < read_key_num; i++) {
                freq_tmp[i] = (ev->MinFreq() == 0) ? 1 : ev->MinFreq();
              }
            }
          } else {
            int64 *freq_tmp = (int64 *)restore_buff.freq_buffer;
            for (int64 i = 0; i < read_key_num; i++) {
              freq_tmp[i] = (ev->MinFreq() == 0) ? 1 : ev->MinFreq();
            }
          }
          if (key_bytes_read > 0) {
            read_key_num = key_bytes_read / sizeof(K);
            VLOG(2) << "restore, read_key_num:" << read_key_num;
            if (restore_customDim && value_shape.dim_size(1) != newDim) {
              VLOG(2) << "restore, read_value_reshape dim: from "
                      << value_shape.dim_size(1) << " to " << newDim;
              if (read_key_num * value_unit_bytes != value_bytes_read) {
                return tensorflow::errors::FailedPrecondition(
                    "Expected read_key_num * value_unit_bytes == "
                    "value_bytes_read, but got read_key_num * value_unit_bytes "
                    "!= value_bytes_read!");
              }

	      std::unique_ptr<char[]> tmp_ptr(new char[buffer_size]);
              size_t read_once =
                  std::min(value_unit_bytes, value_unit_bytes_new);
              for (int i = 0; i < read_key_num; ++i) {
                memcpy(tmp_ptr.get() + i * value_unit_bytes_new,
                       restore_buff.value_buffer + i * value_unit_bytes,
                       read_once);
                if (value_shape.dim_size(1) >= newDim) continue;
                auto p = ev->GetDefaultValue(idx);
                ++idx;
                memcpy(tmp_ptr.get() + i * value_unit_bytes_new +
                           value_unit_bytes,
                       p + value_unit_bytes,
                       value_unit_bytes_new - value_unit_bytes);
              }
              auto tmp = tmp_ptr.release();
              tmp_ptr.reset(restore_buff.value_buffer);
              restore_buff.value_buffer = tmp;
            }

            st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
                partition_id, partition_num, false, device);
            if (cache_for_restore_hbm) {
              cache_for_restore_hbm->update(
                  (K*)restore_buff.key_buffer, read_key_num,
                  (int64*)restore_buff.version_buffer,
                  (int64*)restore_buff.freq_buffer);
            }
            if (!st.ok()) {
              LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
            }
          }
          tot_key_num -= read_key_num;
          tot_key_bytes_read += key_bytes_read;
          tot_value_bytes_read += value_bytes_read;
          tot_version_bytes_read += version_bytes_read;
          tot_freq_bytes_read += freq_bytes_read;
        }

        if (restore_filter_flag) {
          int subpart_filter_offset = part_filter_offset_flat(subpart_id);
          int64 key_filter_part_offset = subpart_filter_offset * sizeof(K);
          int64 version_filter_part_offset = subpart_filter_offset * sizeof(int64);
          int64 freq_filter_part_offset = subpart_filter_offset * sizeof(int64);
          int64 tot_key_filter_num =
            part_filter_offset_flat(subpart_id + 1) - subpart_filter_offset;
          size_t key_filter_bytes_read = 0;
          size_t version_filter_bytes_read = 0;
          size_t freq_filter_bytes_read = 0;
          while (tot_key_filter_num > 0) {
            size_t read_key_num =
              std::min(buffer_size / sizeof(K), buffer_size / sizeof(int64));
            read_key_num = std::min((int64)read_key_num, tot_key_filter_num);
            reader->LookupSegmentOffset(tensor_key + "_filtered",
                key_filter_part_offset + key_filter_bytes_read,
                read_key_num * sizeof(K), restore_buff.key_buffer,
                key_filter_bytes_read);
            if (!reset_version) {
              reader->LookupSegmentOffset(tensor_version + "_filtered",
                  version_filter_part_offset + version_filter_bytes_read,
                  read_key_num * sizeof(int64), restore_buff.version_buffer,
                  version_filter_bytes_read);
            } else {
              int64 *version_tmp = (int64*)restore_buff.version_buffer;
              memset(version_tmp, 0, read_key_num * sizeof(int64));
            }
            reader->LookupSegmentOffset(tensor_freq + "_filtered",
                freq_filter_part_offset + freq_filter_bytes_read,
                read_key_num * sizeof(int64), restore_buff.freq_buffer,
                freq_filter_bytes_read);
            if (key_filter_bytes_read > 0) {
              read_key_num = key_filter_bytes_read / sizeof(K);
              VLOG(2) << "restore, read_key_num:" << read_key_num;
              st = ev->Import(restore_buff, read_key_num, kSavedPartitionNum,
                  partition_id, partition_num, true, device);
              if (cache_for_restore_hbm) {
                cache_for_restore_hbm->update(
                    (K*)restore_buff.key_buffer, read_key_num,
                    (int64*)restore_buff.version_buffer,
                    (int64*)restore_buff.freq_buffer);
              }
              if (!st.ok())
               return st;
              tot_key_filter_num -= read_key_num;
            }
          }
        }
      }
    }

    if (cache_for_restore_hbm) {
      int64 cache_capacity = ev->CacheSize();
      int64 num_of_hbm_ids =
          std::min(cache_capacity, (int64)cache_for_restore_hbm->size());
      K* hbm_ids = new K[num_of_hbm_ids];
      int64* hbm_freqs = new int64[num_of_hbm_ids];
      int64* hbm_versions = nullptr;
      cache_for_restore_hbm->get_cached_ids(
          hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs);
      ev->ImportToHbm(hbm_ids, num_of_hbm_ids);
      ev->storage()->Schedule([ev, hbm_ids, num_of_hbm_ids,
                                       hbm_versions, hbm_freqs]() {
        embedding::BatchCache<K>* cache = ev->Cache();
        cache->update(hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs);
        delete[] hbm_ids;
        delete[] hbm_freqs;
      });
      delete cache_for_restore_hbm;
    }
  }
  return Status::OK();
}


template<class K>
int64 ReadRecord(
    BundleReader* reader,
    const string& record_key,
    K** buffer) {
  TensorShape shape;
  Status st;
  reader->LookupTensorShape(record_key, &shape);
  st = reader->LookupHeader(record_key,
      sizeof(K) * shape.dim_size(0));
  if (!st.ok()) {
    LOG(FATAL)<<"Restore record "<<record_key<<" failed";
  }
  size_t bytes_read = 0;
  *buffer = new K[shape.dim_size(0)];
  reader->LookupSegment(
      record_key, sizeof(K) * shape.dim_size(0),
      (char*)*buffer, bytes_read);
  return shape.dim_size(0);
}

template<class K, class V>
void RestoreSsdRecord(
    EmbeddingVar<K, V>* ev,
    const std::string& ssd_record_file_name,
    const std::string& ssd_emb_file_name) {
  BundleReader ssd_record_reader(Env::Default(),
                                 ssd_record_file_name);
  //Read the data of embedding files
  int64* file_list = nullptr;
  int64 num_of_files =
      ReadRecord(&ssd_record_reader, "files", &file_list);

  int64* invalid_record_count_list = nullptr;
  ReadRecord(&ssd_record_reader,
             "invalid_record_count",
             &invalid_record_count_list);

  int64* record_count_list = nullptr;
  ReadRecord(&ssd_record_reader,
             "record_count",
             &record_count_list);

  //Read the data of keys
  K* key_list = nullptr;
  int64 num_of_keys =
      ReadRecord(&ssd_record_reader, "keys", &key_list);

  int64* key_file_id_list = nullptr;
  ReadRecord(&ssd_record_reader, "keys_file_id", &key_file_id_list);

  int64* key_offset_list = nullptr;
  ReadRecord(&ssd_record_reader, "keys_offset", &key_offset_list);

  //Import the meta of keys to SSDHashKV
  ev->RestoreSsdHashmap(key_list, key_file_id_list,
                        key_offset_list, num_of_keys,
                        file_list, invalid_record_count_list,
                        record_count_list, num_of_files,
                        ssd_emb_file_name);
  delete[] key_list;
  delete[] key_file_id_list;
  delete[] key_offset_list;
  delete[] file_list;
  delete[] invalid_record_count_list;
  delete[] record_count_list;
}

template<class K, class V>
void LoadSsdData(
    EmbeddingVar<K, V>* ev,
    const std::string& ssd_record_file_name,
    const std::string& ssd_emb_file_name) {
  BundleReader ssd_record_reader(Env::Default(),
                                 ssd_record_file_name);
  std::string record_key;

  K* key_list = nullptr;
  int64 num_of_keys =
      ReadRecord(&ssd_record_reader, "keys", &key_list);

  int64* key_file_id_list = nullptr;
  ReadRecord(&ssd_record_reader, "keys_file_id", &key_file_id_list);

  int64* key_offset_list = nullptr;
  ReadRecord(&ssd_record_reader, "keys_offset", &key_offset_list);

  //Load keys and embedding data on ssd
  ev->LoadSsdData(ssd_emb_file_name, key_list,
                  key_file_id_list, key_offset_list,
                  num_of_keys);
  delete[] key_list;
  delete[] key_file_id_list;
  delete[] key_offset_list;
}

Status MoveMatchingFiles(
    Env* env,
    const tstring& pattern,
    const tstring& merged_prefix,
    int64 input_prefix_size);

/*Move two files and one directory:
1. xxxxx-ssd_record.index
2. xxxxx-ssd_record.data 
3. xxxxxx-emb_files/ 
1 and 2 record the meta data of SSDHash,
and 3 records the embeddings on SSD*/
Status MoveSsdFiles(Env* env,
    const gtl::ArraySlice<tstring>& input_prefixes,
    const tstring& merged_prefix);
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_
