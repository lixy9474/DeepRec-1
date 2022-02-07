#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_ROCKSDB_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_ROCKSDB_KV_H_

/*#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"*/

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"

#include "leveldb/db.h"
#include "leveldb/comparator.h"

#include <sstream>
/*
using rocksdb::DB;
using rocksdb::Options;
using rocksdb::PinnableSlice;
using rocksdb::ReadOptions;
using rocksdb::WriteBatch;
using rocksdb::WriteOptions;
using rocksdb::Iterator;
*/

using leveldb::DB;
using leveldb::Options;
//using leveldb::PinnableSlice;
using leveldb::ReadOptions;
using leveldb::WriteBatch;
using leveldb::WriteOptions;
using leveldb::Iterator;

namespace tensorflow {
  template <class V>
  class ValuePtr;

  template <class K, class V>
  class RocksDBKV : public KVInterface<K, V> {
   public:
    RocksDBKV(std::string path) {
      path_ = path;
      options_.create_if_missing = true;
      //options_.IncreaseParallelism();
      //options_.OptimizeLevelStyleCompaction();
      leveldb::Status s = leveldb::DB::Open(options_, path_, &db_);
      //LOG(INFO)<<s.ToString();
      assert(s.ok());
    }

    void SetNewValuePtrFunc(std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn) {
      new_value_ptr_fn_ = new_value_ptr_fn;
    }

    void SetTotalDims(int total_dims) {
      total_dims_ = total_dims;
    }

    ~RocksDBKV() {
      delete db_;
    }

    Status Lookup(K key, ValuePtr<V>** value_ptr) {
      std::string key_str, val_str;
      std::stringstream ss;
      ss<<key;
      std::istringstream is(ss.str());
      is>>key_str;
      //leveldb::Slice db_key((char*)(&key), sizeof(void*));
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      leveldb::ReadOptions options;
      leveldb::Status s = db_->Get(options, key_str, &val_str);
      //LOG(INFO)<<"key: "<<key_str<<", "<<s.ToString();
      if (s.IsNotFound()) {
        return errors::NotFound(
            "Unable to find Key: ", key, " in RocksDB.");
      } else {
        memcpy((int64 *)(val->GetPtr()), &val_str[0], val_str.length());
        //LOG(INFO)<<val_str.size();
        /*V* st = (V*)val->GetPtr() + sizeof(FixLengthHeader) / sizeof(V);
        for (int i = 0; i < 3; i++)
          LOG(INFO)<<st[i];*/
        *value_ptr = val;
        return Status::OK();
      }
    }

    Status Insert(K key, const ValuePtr<V>* value_ptr) {
      char *val_str;
      std::string key_str;
      std::string value_res((char*)value_ptr->GetPtr(), sizeof(FixLengthHeader) +  total_dims_ * sizeof(V));
      std::ostringstream ss;
      ss<<key;
      std::istringstream is(ss.str());
      is>>key_str;
      //leveldb::Slice db_key((char*)(&key), sizeof(void*));
      leveldb::Status s = db_->Put(WriteOptions(), key_str, value_res);
      //
      //LOG(INFO)<<"Insert "<<key_str<<" Status: "<<s.ToString();
      if (!s.ok()){
        return errors::AlreadyExists(
            "already exists Key: ", key, " in RocksDB.");
      } else {
        return Status::OK();
      }
    } 

    Status Commit(K key, const ValuePtr<V>* value_ptr) {
      char *val_str;
      std::string key_str;
      std::ostringstream ss;
      ss<<key;
      std::istringstream is(ss.str());
      is>>key_str;
      std::string value_res((char*)value_ptr->GetPtr(), sizeof(FixLengthHeader) +  total_dims_ * sizeof(V));
      //leveldb::Slice db_key((char*)(&key), sizeof(void*));
     /* V* st = (V*)value_ptr->GetPtr() + sizeof(FixLengthHeader) / sizeof(V);
      for (int i = 0; i < 3; i++)
        LOG(INFO)<<st[i];*/
      //int64* st = (int64*)value_ptr->GetPtr();
      //LOG(INFO)<<((*st) >> 48);
      leveldb::Status s = db_->Put(WriteOptions(), key_str, value_res);
      //LOG(INFO)<<key_str<<", "<<s.ToString();
      /*std::string val_temp;
      leveldb::ReadOptions options;
      s = db_->Get(options, key_str, &val_temp);
      memcpy((int64 *)value_ptr->GetPtr(), &val_temp[0], val_temp.length());
      LOG(INFO)<<val_temp;
      st = (V*)value_ptr->GetPtr() + sizeof(FixLengthHeader) / sizeof(V);
      for (int i = 0; i < 3; i++)
          LOG(INFO)<<st[i];*/
      delete value_ptr;
      if (!s.ok()){
        return errors::AlreadyExists(
            "already exists Key: ", key, " in RocksDB.");
      } else {
        return Status::OK();
      }
    }

    Status Remove(K key) {
      std::string key_str;
      std::ostringstream key_temp;
      key_temp << key;
      std::istringstream is(key_temp.str());
      is >> key_str;
      leveldb::Status s = db_->Delete(WriteOptions(), key_str);
      if (s.ok()) {
        return Status::OK();
      } else {
        return errors::NotFound(
            "Unable to find Key: ", key, " in RocksDB.");
      }
    }
    

    Status GetSnapshot(std::vector<K>* key_list, std::vector<ValuePtr<V>* >* value_ptr_list) {
      ReadOptions options;
      options.snapshot = db_->GetSnapshot();
      Iterator* it = db_->NewIterator(options);
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key_str, value_str;
        ValuePtr<V>* value_ptr = new_value_ptr_fn_(total_dims_);
        key_str = it->key().ToString();
        value_str = it->value().ToString();
        std::istringstream is(key_str);
        int64 key_output;
        is >> key_output;
        key_list->emplace_back(key_output);
        void* ptr = value_ptr->GetPtr();
        memcpy(ptr, &value_ptr[0] ,total_dims_ * sizeof(V) + 2 * sizeof(int64));
        value_ptr_list->emplace_back(value_ptr);    
      }
      assert(it->status().ok());
      delete it;
      db_->ReleaseSnapshot(options.snapshot);
      return Status::OK();
    }

    int64 Size() const {
      return 0;
    }

    void DestoryValuePtr(std::vector<ValuePtr<V>*> value_ptr_list) {
      for (auto it : value_ptr_list) {
        delete it;
      }
    }

  std::string DebugString() const {
    return "";
  }
   private:
    DB* db_;
    Options options_;
    std::string path_;
    std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
    int total_dims_;

  };
} //namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_