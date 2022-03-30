#include <thread>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

#include <sys/resource.h>
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace embedding {
namespace {
const int THREADNUM = 16;
const int64 max = 2147483647;

struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory() : size(0), resident(0), share(0),
                 trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    //ret.push_back(reader->key().ToString());
    ret.push_back(std::string(reader->key()));
  }
  return ret;
}

TEST(TensorBundleTest, TestEVShrinkL2) {
  int64 value_size = 3;
  int64 insert_num = 5;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 1.0));
  //float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "name", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* emb_var
    = new EmbeddingVar<int64, float>("name",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", -1, 0, 99999, 14.0));
  emb_var ->Init(value, 1);
  
  for (int64 i=0; i < insert_num; ++i) {
    ValuePtr<float>* value_ptr = nullptr;
    Status s = emb_var->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = emb_var->flat(value_ptr);
    vflat += vflat.constant((float)i);
  }

  int size = emb_var->Size();
  emb_var->Shrink();
  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size:" << emb_var->Size();

  ASSERT_EQ(emb_var->Size(), 2);
}

TEST(TensorBundleTest, TestEVShrinkLockless) {

  int64 value_size = 64;
  int64 insert_num = 30;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  int steps_to_live = 5;
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "name", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* emb_var
    = new EmbeddingVar<int64, float>("name",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", steps_to_live));
  emb_var ->Init(value, 1);


  LOG(INFO) << "size:" << emb_var->Size();


  for (int64 i=0; i < insert_num; ++i) {
    ValuePtr<float>* value_ptr = nullptr;
    Status s = emb_var->LookupOrCreateKey(i, &value_ptr, i);
    typename TTypes<float>::Flat vflat = emb_var->flat(value_ptr);
  }

  int size = emb_var->Size();
  emb_var->Shrink(insert_num);

  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size: " << emb_var->Size();

  ASSERT_EQ(size, insert_num);
  ASSERT_EQ(emb_var->Size(), steps_to_live);

}


TEST(EmbeddingVariableTest, TestEmptyEV) {
  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  {
    auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
    TF_CHECK_OK(storage_manager->Init());
    EmbeddingVar<int64, float>* variable
              = new EmbeddingVar<int64, float>("EmbeddingVar",
                  storage_manager);
    variable->Init(value, 1);

    LOG(INFO) << "size:" << variable->Size();
    Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

    BundleWriter writer(Env::Default(), Prefix("foo"));
    DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
    TF_ASSERT_OK(writer.Finish());

    {
      BundleReader reader(Env::Default(), Prefix("foo"));
      TF_ASSERT_OK(reader.status());
      EXPECT_EQ(
          AllTensorKeys(&reader),
          std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
      {
        string key = "var/part_0-keys";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read keys:" << val.DebugString();
      }
      {
        string key = "var/part_0-values";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0, value_size});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read values:" << val.DebugString();
      }
      {
        string key = "var/part_0-versions";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read versions:" << val.DebugString();
      }
    }
  }
}

TEST(EmbeddingVariableTest, TestEVExportSmallLockless) {

  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", 5));
  variable->Init(value, 1);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  for (int64 i = 0; i < 5; i++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
    vflat(i) = 5.0;
  }

  LOG(INFO) << "size:" << variable->Size();


  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5, value_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}
/*
TEST(EmbeddingVariableTest, TestEVExportLargeLockless) {

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", 5));
  variable->Init(value, 1);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  int64 ev_size = 10048576;
  for (int64 i = 0; i < ev_size; i++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }

  LOG(INFO) << "size:" << variable->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size, value_size});
      LOG(INFO) << "read values:" << val.DebugString();
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}
*/
void multi_insertion(EmbeddingVar<int64, float>* variable, int64 value_size){
  for (long j = 0; j < 5; j++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(j, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }
}

TEST(EmbeddingVariableTest, TestMultiInsertion) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager);

  variable->Init(value, 1);

  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(multi_insertion,variable, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  std::vector<int64> tot_key_list;
  std::vector<float* > tot_valueptr_list;
  std::vector<int64> tot_version_list;
  std::vector<int64> tot_freq_list;
  embedding::Iterator* it = nullptr;
  int64 total_size = variable->GetSnapshot(&tot_key_list, &tot_valueptr_list, &tot_version_list, &tot_freq_list, &it);

  ASSERT_EQ(variable->Size(), 5);
  ASSERT_EQ(variable->Size(), total_size);
}

void InsertAndLookup(EmbeddingVar<int64, int64>* variable, int64 *keys, long ReadLoops, int value_size){
  for (long j = 0; j < ReadLoops; j++) {
    int64 *val = (int64 *)malloc((value_size+1)*sizeof(int64));
    variable->LookupOrCreate(keys[j], val, &(keys[j]));
    variable->LookupOrCreate(keys[j], val, (&keys[j]+1));
    ASSERT_EQ(keys[j] , val[0]);
    free(val);
  }
}

void MultiBloomFilter(EmbeddingVar<int64, float>* var, int value_size, int64 i) {
  for (long j = 0; j < 1; j++) {
    float *val = (float *)malloc((value_size+1)*sizeof(float));
    var->LookupOrCreate(i+1, val, nullptr);
  }
}

TEST(EmbeddingVariableTest, TestBloomFilter) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal", 10, 0.01));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));
  float *default_value = (float *)malloc((value_size+1)*sizeof(float));
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(1, val, default_value);
  var->LookupOrCreate(2, val, default_value);
  
  std::vector<int64> keylist;
  std::vector<float *> valuelist;
  std::vector<int64> version_list;
  std::vector<int64> freq_list;

  embedding::Iterator* it = nullptr;
  var->GetSnapshot(&keylist, &valuelist, &version_list, &freq_list, &it);
  ASSERT_EQ(var->Size(), keylist.size());  

}

TEST(EmbeddingVariableTest, TestBloomCounterInt64) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal", 10, 0.01, DT_UINT64));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int64* counter = (int64*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt32) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal", 10, 0.01, DT_UINT32));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int32* counter = (int32*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt16) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal_contiguous", 10, 0.01, DT_UINT16));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  //int16* counter = (int16 *)var->GetBloomCounter(); 
  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int16* counter = (int16*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt8) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 3, 99999, -1.0, "normal_contiguous", 10, 0.01, DT_UINT8));

  var->Init(value, 1);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilter<int64, float, EmbeddingVar<int64, float>>*>(filter);
  int8* counter = (int8*)bloom_filter->GetBloomCounter();//(int64 *)var->GetBloomCounter(); 
  //int8* counter = (int8 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestInsertAndLookup) {
  int64 value_size = 128;
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));
 // float* fill_v = (int64*)malloc(value_size * sizeof(int64));
  auto storage_manager = new embedding::StorageManager<int64, int64>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, int64>* variable
    = new EmbeddingVar<int64, int64>("EmbeddingVar",
       storage_manager/*, EmbeddingConfig(0, 0, 1, 0, "")*/);

  variable->Init(value, 1);

  int64 InsertLoops = 1000;
  bool* flag = (bool *)malloc(sizeof(bool)*max);
  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoops);
  long *counter = (long *)malloc(sizeof(long)*InsertLoops);

  for (long i = 0; i < max; i++) {
    flag[i] = 0;
  }

  for (long i = 0; i < InsertLoops; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < InsertLoops) {
    long j = rand() % max;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else { // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(InsertAndLookup, variable, &keys[i*InsertLoops/THREADNUM], InsertLoops/THREADNUM, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

}

void MultiFilter(EmbeddingVar<int64, float>* variable, int value_size) {
  float *val = (float *)malloc((value_size+1)*sizeof(float));
  variable->LookupOrCreate(20, val, nullptr);
}

TEST(EmbeddingVariableTest, TestFeatureFilterParallel) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* var 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(0, 0, 1, 1, "", 5, 7));

  var->Init(value, 1);
  float *val = (float *)malloc((value_size+1)*sizeof(float));
  int thread_num = 5;
  std::vector<std::thread> insert_threads(thread_num);
  for (size_t i = 0 ; i < thread_num; i++) {
    insert_threads[i] = std::thread(MultiFilter, var, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  ValuePtr<float>* value_ptr = nullptr;
  var->LookupOrCreateKey(20, &value_ptr);
  ASSERT_EQ(value_ptr->GetFreq(), thread_num);
}


EmbeddingVar<int64, float>* InitEV_Lockless(int64 value_size) {
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager);

  variable->Init(value, 1);
  return variable;
}

void MultiLookup(EmbeddingVar<int64, float>* variable, int64 InsertLoop, int thread_num, int i) {
  for (int64 j = i * InsertLoop/thread_num; j < (i+1)*InsertLoop/thread_num; j++) {
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(j, &value_ptr);
  }
}

void BM_MULTIREAD_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  float* fill_v = (float*)malloc(value_size * sizeof(float));

  for (int64 i = 0; i < InsertLoop; i++){
    ValuePtr<float>* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }

  testing::StartTiming();
  while(iters--){
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(MultiLookup, variable, InsertLoop, thread_num, i);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }

}

void hybrid_process(EmbeddingVar<int64, float>* variable, int64* keys, int64 InsertLoop, int thread_num, int64 i, int64 value_size) {
  float *val = (float *)malloc(sizeof(float)*(value_size + 1));
  for (int64 j = i * InsertLoop/thread_num; j < (i+1) * InsertLoop/thread_num; j++) {
    variable->LookupOrCreate(keys[j], val, nullptr);
  }
}

void BM_HYBRID_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoop);

  for (int64 i = 0; i < InsertLoop; i++) {
    keys[i] =  rand() % 1000;
  }

  testing::StartTiming();
  while (iters--) {
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(hybrid_process, variable, keys, InsertLoop, thread_num, i, value_size);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }
}

BENCHMARK(BM_MULTIREAD_LOCKLESS)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

BENCHMARK(BM_HYBRID_LOCKLESS)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);


TEST(EmbeddingVariableTest, TestAllocate) {
  int value_len = 8;
  double t0 = getResident()*getpagesize()/1024.0/1024.0;
  double t1 = 0;
  LOG(INFO) << "memory t0: " << t0;
  for (int64 i = 0; i < 1000; ++i) {
    float* tensor_val = TypedAllocator::Allocate<float>(ev_allocator(), value_len, AllocationAttributes());
    t1 = getResident()*getpagesize()/1024.0/1024.0;
    memset(tensor_val, 0, sizeof(float) * value_len);
  }
  double t2 = getResident()*getpagesize()/1024.0/1024.0;
  LOG(INFO) << "memory t1-t0: " << t1-t0;
  LOG(INFO) << "memory t2-t1: " << t2-t1;
  LOG(INFO) << "memory t2-t0: " << t2-t0;
}

TEST(EmbeddingVariableTest, TestEVStorageType_DRAM) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(/*emb_index = */0, /*primary_emb_index = */0,
                          /*block_num = */1, /*slot_num = */1,
                          /*name = */"", /*steps_to_live = */0,
                          /*filter_freq = */0, /*max_freq = */999999,
                          /*l2_weight_threshold = */-1.0, /*layout = */"normal",
                          /*max_element_size = */0, /*false_positive_probability = */-1.0,
                          /*counter_type = */DT_UINT64));
  variable->Init(value, 1);

  int64 ev_size = 100;
  for (int64 i = 0; i < ev_size; i++) {
    variable->LookupOrCreate(i, fill_v, nullptr);
  }

  LOG(INFO) << "size:" << variable->Size();
}

void t1(KVInterface<int64, float>* hashmap) {
  for (int i = 0; i< 100; ++i) {
    hashmap->Insert(i, new NormalValuePtr<float>(ev_allocator(), 100));
  }
}

TEST(EmbeddingVariableTest, TestRemoveLockless) {

  KVInterface<int64, float>* hashmap = new LocklessHashMap<int64, float>();
  ASSERT_EQ(hashmap->Size(), 0);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  auto t = std::thread(t1, hashmap);
  t.join();
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  ASSERT_EQ(hashmap->Size(), 100);
  TF_CHECK_OK(hashmap->Remove(1));
  TF_CHECK_OK(hashmap->Remove(2));
  ASSERT_EQ(hashmap->Size(), 98);
  LOG(INFO) << "2 size:" << hashmap->Size();
}

TEST(EmbeddingVariableTest, TestBatchCommitofDBKV) {
  int64 value_size = 4;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage_manager = new embedding::StorageManager<int64, float>(
                 "EmbeddingVar", embedding::StorageConfig(embedding::LEVELDB, testing::TmpDir(), 1000, "normal_contiguous"));
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        storage_manager,
          EmbeddingConfig(/*emb_index = */0, /*primary_emb_index = */0,
                          /*block_num = */1, /*slot_num = */0,
                          /*name = */"", /*steps_to_live = */0,
                          /*filter_freq = */0, /*max_freq = */999999,
                          /*l2_weight_threshold = */-1.0, /*layout = */"normal_contiguous",
                          /*max_element_size = */0, /*false_positive_probability = */-1.0,
                          /*counter_type = */DT_UINT64));
  variable->Init(value, 1);
  std::vector<ValuePtr<float>*> value_ptr_list;
  std::vector<int64> key_list;

  for(int64 i = 0; i < 6; i++) {
    key_list.emplace_back(i);
    ValuePtr<float>* tmp = new NormalContiguousValuePtr<float>(ev_allocator(), 4);
    value_ptr_list.emplace_back(tmp);
  }

  variable->BatchCommit(key_list, value_ptr_list);
  for(int64 i = 0; i < 6; i++) {
    ValuePtr<float>* tmp = nullptr;
    Status s = variable->storage_manager()->GetOrCreate(i, &tmp, 4);
    ASSERT_EQ(s.ok(), true);
  }
}

void InsertAndCommit(KVInterface<int64, float>* hashmap) {
  for (int64 i = 0; i< 100; ++i) {
    const ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 100);
    hashmap->Insert(i, tmp);
    hashmap->Commit(i, tmp);
  }
}

TEST(EmbeddingVariableTest, TestSizeDBKV) {
  KVInterface<int64, float>* hashmap = new LevelDBKV<int64, float>(testing::TmpDir());
  hashmap->SetTotalDims(100);
  ASSERT_EQ(hashmap->Size(), 0);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  auto t = std::thread(InsertAndCommit, hashmap);
  t.join();
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  ASSERT_EQ(hashmap->Size(), 100);
  TF_CHECK_OK(hashmap->Remove(1));
  TF_CHECK_OK(hashmap->Remove(2));
  ASSERT_EQ(hashmap->Size(), 98);
  LOG(INFO) << "2 size:" << hashmap->Size();
}

class DataLoader
{
public:
    std::vector<int64> ids;
    size_t offset;


    DataLoader(std::string filepath, size_t feature_id = 0, size_t size = 100000) // 10w
    {
        std::ifstream fp(filepath);
        if (!fp)
        {
            LOG(INFO) << "open " << filepath << " fail." << std::endl
                    << std::flush;
            exit(1);
        }
        std::string line;
        std::getline(fp, line); //跳过列名，第一行不做处理

        while (std::getline(fp, line) && ids.size()<size)
        {
            std::string number;
            std::istringstream readstr(line);
            for (size_t j = 0; j <= feature_id; ++j)
            {
                std::getline(readstr, number, ',');
            }
            ids.push_back(std::atoi(number.c_str()));
        }
        offset = 0;
    }
    void init() { offset = 0; }
    size_t sample(int64 *batch_ids, size_t batch_size)
    {
        size_t i = 0;
        for (i = 0; offset < ids.size() && i < batch_size; ++offset, ++i)
        {
            batch_ids[i] = ids[offset];
        }
        return i;
    }
    size_t size() { return ids.size(); }
};

void BatchCommit(KVInterface<int64, float>* hashmap, std::vector<int64> keys, int batch_size) {
  std::vector<ValuePtr<float>*> value_ptrs;
  for (int64 i = 0; i < keys.size(); ++i) {
    ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 128);
    tmp->SetValue(float(keys[i]), 128);
    value_ptrs.push_back(tmp);
  }
  ASSERT_EQ(keys.size(), value_ptrs.size());
  uint64 start = Env::Default()->NowNanos();
  
  for (int64 i = 0; i < keys.size();) {
    std::vector<int64> batch_keys;
    std::vector<ValuePtr<float>*> batch_value_ptrs;
    for(int j = 0; j < batch_size && i < keys.size(); ++j,++i){
      batch_keys.push_back(keys[i]);
      batch_value_ptrs.push_back(value_ptrs[i]);
    }
    hashmap->BatchCommit(batch_keys, batch_value_ptrs);
  }
  uint64 end = Env::Default()->NowNanos();
  uint64 result_cost = end - start;
  LOG(INFO) << "BatchCommit time: " << result_cost << " ns";
}

void SingleCommit(KVInterface<int64, float>* hashmap, std::vector<int64> keys) {
  std::vector<ValuePtr<float>*> value_ptrs;
  for (int64 i = 0; i < keys.size(); ++i) {
    ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 128);
    tmp->SetValue(float(keys[i]), 128);
    value_ptrs.push_back(tmp);
  }
  ASSERT_EQ(keys.size(), value_ptrs.size());
  uint64 start = Env::Default()->NowNanos();
  
  for (int64 i = 0; i < keys.size(); i++) {
    hashmap->Commit(keys[i], value_ptrs[i]);
  }
  uint64 end = Env::Default()->NowNanos();
  uint64 result_cost = end - start;
  LOG(INFO) << "SingleCommit time: " << result_cost << " ns";
}

void BatchLookup(KVInterface<int64, float>* hashmap, std::vector<int64> keys) {
  std::vector<ValuePtr<float>*> value_ptrs;
  for (int64 i = 0; i< keys.size(); ++i) {
    ValuePtr<float>* tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 128);
    value_ptrs.push_back(tmp);
  }
  ASSERT_EQ(keys.size(), value_ptrs.size());
  uint64 start = Env::Default()->NowNanos();
  for (int64 i = 0; i< keys.size(); ++i) {
    TF_CHECK_OK(hashmap->Lookup(keys[i], &value_ptrs[i]));
    ValuePtr<float>* ori_tmp= new NormalContiguousValuePtr<float>(ev_allocator(), 128);
    ori_tmp->SetValue(float(keys[i]), 128);
    if(!value_ptrs[i]->EqualTo(ori_tmp, 128)){
      LOG(INFO) << "keys[i]" << keys[i];
      LOG(INFO) << "value_ptrs[i]->PrintValue(128);";
      value_ptrs[i]->PrintValue(128);
      LOG(INFO) << "ori_tmp->SetValue(float(keys[i]), 128);";
      ori_tmp->PrintValue(128);
    }
    delete ori_tmp;
  }
  uint64 end = Env::Default()->NowNanos();
  uint64 result_cost = end - start;
  LOG(INFO) << "BatchLookup time: " << result_cost << " ns";
}


void LevelDBKVTest(int total_size, int batch_size){
  KVInterface<int64, float>* hashmap = new LevelDBKV<int64, float>("/tmp/db_ut1");
  hashmap->SetTotalDims(128);
  ASSERT_EQ(hashmap->Size(), 0);
  DataLoader dl("/home/code/DRAM-SSD-Storage/dataset/taobao/shuffled_sample.csv", 0, total_size);
  auto t1 = std::thread(BatchCommit, hashmap, dl.ids, batch_size);
  t1.join();
  auto t2 = std::thread(BatchLookup, hashmap, dl.ids);
  t2.join();
  delete hashmap;
}

TEST(KVInterfaceTest, TestLargeLEVELDBKV) {
  std::vector<int> total_size_list = {100000, 1000000};          // 10w, 100w
  std::vector<int> batch_size_list = {200, 2000, 20000, 100000}; // 200, 2000, 2w, 10w
  for (int total_size : total_size_list) {
    for (int batch_size : batch_size_list) {
      LOG(INFO) << "LevelDB total_size: " << total_size << ", batch_size: " << batch_size << std::endl;
      for(int e = 0; e < 5; e++){
        LOG(INFO) << "epoch: " << e << std::endl;
        LevelDBKVTest(total_size, batch_size);
      }
    }
  }
}

void SSDKVTest(int total_size, int batch_size){
  KVInterface<int64, float>* hashmap = new SSDKV<int64, float>("/tmp/ssd_ut1");
  hashmap->SetTotalDims(128);
  ASSERT_EQ(hashmap->Size(), 0);
  DataLoader dl("/home/code/DRAM-SSD-Storage/dataset/taobao/shuffled_sample.csv", 0, total_size);
  auto t1 = std::thread(BatchCommit, hashmap, dl.ids, batch_size);
  t1.join();
  auto t2 = std::thread(BatchLookup, hashmap, dl.ids);
  t2.join();
  delete hashmap;
}


TEST(KVInterfaceTest, TestLargeSSDKV) {
  std::vector<int> total_size_list = {100000, 1000000};          // 10w, 100w
  std::vector<int> batch_size_list = {200, 2000, 20000, 100000}; // 200, 2000, 2w, 10w
  for (int total_size : total_size_list) {
    for (int batch_size : batch_size_list) {
      LOG(INFO) << "SSD total_size: " << total_size << ", batch_size: " << batch_size;
      for(int e = 0; e < 5; e++){
        LOG(INFO) << "epoch: " << e;
        SSDKVTest(total_size, batch_size);
      }
      
    }
  }
}


/*
void LevelDBKVSingleTest(int total_size){
  KVInterface<int64, float>* hashmap = new LevelDBKV<int64, float>("/tmp/db_ut1");
  hashmap->SetTotalDims(128);
  ASSERT_EQ(hashmap->Size(), 0);
  DataLoader dl("/home/code/DRAM-SSD-Storage/dataset/taobao/shuffled_sample.csv", 0, total_size);
  auto t1 = std::thread(SingleCommit, hashmap, dl.ids);
  t1.join();
  auto t2 = std::thread(BatchLookup, hashmap, dl.ids);
  t2.join();
  delete hashmap;
}

TEST(KVInterfaceTest, TestLargeSingleLEVELDBKV) {
  std::vector<int> total_size_list = {10000, 50000};
  std::vector<int> batch_size_list = {5, 10, 15, 20};
  for (int total_size : total_size_list) {
    for (int batch_size : batch_size_list) {
      LOG(INFO) << "LevelDB total_size: " << total_size << ", batch_size: " << batch_size << std::endl;
      for(int e = 0; e < 5; e++){
        LOG(INFO) << "epoch: " << e << std::endl;
        LevelDBKVSingleTest(total_size*batch_size);
      }
    }
  }
}


void SSDKVSingleTest(int total_size){
  KVInterface<int64, float>* hashmap = new SSDKV<int64, float>("/tmp/ssd_ut1");
  hashmap->SetTotalDims(128);
  ASSERT_EQ(hashmap->Size(), 0);
  DataLoader dl("/home/code/DRAM-SSD-Storage/dataset/taobao/shuffled_sample.csv", 0, total_size);
  auto t1 = std::thread(SingleCommit, hashmap, dl.ids);
  t1.join();
  auto t2 = std::thread(BatchLookup, hashmap, dl.ids);
  t2.join();
  delete hashmap;
}


TEST(KVInterfaceTest, TestLargeSingleSSDKV) {
  std::vector<int> total_size_list = {10000, 50000};
  std::vector<int> batch_size_list = {5, 10, 15, 20};
  for (int total_size : total_size_list) {
    for (int batch_size : batch_size_list) {
      LOG(INFO) << "SSD total_size: " << total_size << ", batch_size: " << batch_size;
      for(int e = 0; e < 5; e++){
        LOG(INFO) << "epoch: " << e;
        SSDKVSingleTest(total_size*batch_size);
      }
      
    }
  }
}
*/

void BatchEviction(std::vector<std::pair<KVInterface<int64, float>*, Allocator*>>& kvs_, embedding::BatchCache<K>* cache_, int batch_size, size_t cache_capacity_, mutex& mu_, condition_variable& shutdown_cv_, bool& shutdown_) {
  // Env* env = Env::Default();
  const int kSize = 1000;
  // if (cache_capacity_ == -1) {
  //   while (true) {
  //     mutex_lock l(mu_);
  //     if (done_) {
  //       break;
  //     }
  //   }
  // }
  K evic_ids[kSize];
  while (true) {
    mutex_lock l(mu_);
    if (shutdown_) {
      break;
    }
    const int kTimeoutMilliseconds = 10 * 1;
    WaitForMilliseconds(&l, &shutdown_cv_, kTimeoutMilliseconds);

    int cache_count = cache_->size();
    if (cache_count > cache_capacity_) {
      // eviction
      int k_size = cache_count - cache_capacity_;
      k_size = std::min(k_size, kSize);
      size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      for (int64 i = 0; i < true_size; ++i) {
        if (kvs_[0].first->Lookup(evic_ids[i], &value_ptr).ok()) {
          TF_CHECK_OK(kvs_[0].first->Remove(evic_ids[i]));
          TF_CHECK_OK(kvs_[1].first->Commit(evic_ids[i], value_ptr));
          // delete value_ptr is nessary;
        } else {
          // bypass
        }
      }
    }
  }
}


void UpdateValuePtrAdd(std::vector<std::pair<KVInterface<int64, float>*, Allocator*>>& kvs_, embedding::BatchCache<K>* cache_, DataLoader& dl, int batch_size, mutex& mu_, condition_variable& shutdown_cv_, bool& shutdown_) {
  mutex_lock l(mu_);
  std::vector<ValuePtr<float>*> value_ptrs(nullptr, batch_size);
  int64 *batch_ids = new int64[batch_size];
  size_t true_size;
  true_size = dl.sample(batch_ids, batch_size);
  cache_->add_to_rank(batch_ids, true_size);
  int level = 0;
  while (true_size > 0){
      for (int i = 0; i < true_size; ++i){
        bool found = false;
        for (; level < 2; ++level) {
        Status s = kvs_[level].first->Lookup(batch_ids[i], &value_ptrs[i]);
        if (s.ok()) {
          found = true;
          break;
        }
      }
      if (!found) {
        value_ptrs[i] = new NormalContiguousValuePtr<float>(ev_allocator(), 128);;
        value_ptrs[i]->SetValue(float(batch_ids[i]), 128);
      }
      if (level || !found) {
        Status s = kvs_[0].first->Insert(batch_ids[i], value_ptrs[i]);
        if (s.ok()) {
          // Insert Success
          return s;
        } else {
          // Insert Failed, key already exist
          value_ptrs[i]->Destroy(kvs_[0].second);
          delete value_ptrs[i];
          s = kvs_[0].first->Lookup(batch_ids[i], value_ptrs[i]);
          return s;
        }
      }
    }
    for(int j = 0; j < true_size; ++j){
        value_ptrs[j]->UpdateTest();
    }
    true_size = dl.sample(batch_ids, batch_size);
  }
  shutdown_ = true;
  
}

void SSDKVConcurrentTest(int total_size, int batch_size){
  mutex mu_;
  condition_variable shutdown_cv_;
  bool shutdown_ GUARDED_BY(mu_) = false;
  std::vector<std::pair<KVInterface<int64, float>*, Allocator*>> kvs_;
  kvs_.push_back(std::make_pair(new LocklessHashMap<int64, float>(), ev_allocator()));
  kvs_.push_back(std::make_pair(new SSDKV<int64, float>("/tmp/ssd_ut1"), ev_allocator()));
  kvs_[1]->SetTotalDims(128);
  ASSERT_EQ(kvs_[1]->Size(), 0);
  embedding::BatchCache cache_ = new embedding::LRUCache<float>();
  DataLoader dl("/home/code/DRAM-SSD-Storage/dataset/taobao/shuffled_sample.csv", 0, total_size);
  auto t1 = std::thread(UpdateValuePtrAdd, kvs_, cache_, dl, batch_size, mu_, shutdown_, shutdown_cv_);
  // auto t2 = std::thread(UpdateValuePtrAdd, kvs_, dl, batch_size, mu_, shutdown_, shutdown_cv_);
  auto t3 = std::thread(BatchEviction, kvs_, cache_, batch_size, 30000, mu_, shutdown_, shutdown_cv_);
  t1.join();
  // t2.join();
  t3.join();
  // delete kvs_[0].first;
  // delete kvs_[1].first;
}

TEST(KVInterfaceTest, TestLargeConcurrentSSDKV) {
  std::vector<int> total_size_list = {100000, 1000000};          // 10w, 100w
  std::vector<int> batch_size_list = {200, 2000, 20000, 100000}; // 200, 2000, 2w, 10w
  for (int total_size : total_size_list) {
    for (int batch_size : batch_size_list) {
      LOG(INFO) << "SSD total_size: " << total_size << ", batch_size: " << batch_size;
      for(int e = 0; e < 5; e++){
        LOG(INFO) << "epoch: " << e;
        SSDKVConcurrentTest(total_size, batch_size);
      }
      
    }
  }
}


} // namespace
} // namespace embedding
} // namespace tensorflow
