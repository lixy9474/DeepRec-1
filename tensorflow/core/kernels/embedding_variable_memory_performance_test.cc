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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif //GOOGLE_CUDA

#include <time.h>
#include <sys/resource.h>
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
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
namespace embedding {
TEST(EmbeddingVariableTest, TestMemory) {
  setenv("EV_DATA_ALIGNED", "false", 1);
  double m0 = getResident();
  Allocator* allocator = ev_allocator();
  double m1 = getResident();
  auto embedding_config = EmbeddingConfig(
			0, 0, 1, 0, "emb_var", 0,
			0, 999999, -1.0, "light",
			0, -1.0, DT_UINT64, 4096,
			0.0, false, false, false);
	auto feat_desc = new embedding::FeatureDescriptor<float>(
      1, 1, allocator, embedding::StorageType::DRAM,
      embedding_config.is_save_freq(),
      embedding_config.is_save_version(),
      std::pair<bool, int64>(false, 0));
	auto storage =
      embedding::StorageFactory::Create<int64, float>(
          embedding::StorageConfig(
              embedding::StorageType::DRAM, "",
              {1024, 1024, 1024, 1024}, "light",
              embedding_config),
          allocator,
          feat_desc,
          "emb_var");
	auto ev = new EmbeddingVar<int64, float>(
      "emb_var",
      storage,
      embedding_config,
      allocator,
      feat_desc);
  int64 value_size = 1;
  Tensor default_value(DT_FLOAT, TensorShape({4096, value_size}));
	ev->Init(default_value, 4096);
  LOG(INFO)<<"Init memory used: "<<m1 - m0;
	const int64 num_of_ids = 65536;
	srand((unsigned)time(NULL));
	int64 key_list[num_of_ids];
	for (int i = 0; i < num_of_ids; i++)
	  key_list[i] = rand() % 10000000;
	
  m0 = getResident() * getpagesize();
	for (int i = 0 ; i < num_of_ids; i++) {
		void* value_ptr = nullptr;
		bool is_filter = false;
    ev->LookupOrCreateKey(key_list[i], &value_ptr,
                          &is_filter, false);
	}
  m1 = getResident() * getpagesize();
	
	LOG(INFO)<<"Memory used: "<<m1 - m0;
}

TEST(EmbeddingVariableTest, TestCounterFilterMemory) {
  //setenv("EV_DATA_ALIGNED", "false", 1);
  double m0 = getResident();
  Allocator* allocator = ev_allocator();
  double m1 = getResident();
  auto embedding_config = EmbeddingConfig(
			0, 0, 1, 0, "emb_var", 0,
			0, 999999, -1.0, "light",
			0, -1.0, DT_UINT64, 4096,
			0.0, false, false, false);
	auto feat_desc = new embedding::FeatureDescriptor<float>(
      1, 1, allocator, embedding::StorageType::DRAM,
      false,
      embedding_config.is_save_version(),
      std::pair<bool, int64>(false, 0));
	auto storage =
      embedding::StorageFactory::Create<int64, float>(
          embedding::StorageConfig(
              embedding::StorageType::DRAM, "",
              {1024, 1024, 1024, 1024}, "light",
              embedding_config),
          allocator,
          feat_desc,
          "emb_var");
	auto ev = new EmbeddingVar<int64, float>(
      "emb_var",
      storage,
      embedding_config,
      allocator,
      feat_desc);
  int64 value_size = 16;
  Tensor default_value(DT_FLOAT, TensorShape({4096, value_size}));
	ev->Init(default_value, 4096);
  LOG(INFO)<<"Init memory used: "<<m1 - m0;
	const int64 num_of_ids = 1000000;
	srand((unsigned)time(NULL));
	int64* key_list = new int64[num_of_ids];
	for (int i = 0; i < num_of_ids; i++)
	  key_list[i] = i;
	
  void* value_ptr = nullptr;
	bool is_filter = false;
  m0 = getResident() * getpagesize();
	for (int i = 0 ; i < num_of_ids; i++) {
    ev->LookupOrCreateKey(key_list[i], &value_ptr,
                          &is_filter, false);
    //if (i < num_of_ids / 2)
     // ev->LookupOrCreateKey(key_list[i], &value_ptr,
      //                    &is_filter, false);
	}
  m1 = getResident() * getpagesize();
  LOG(INFO)<<"Step 1: Memory used: "<<m1 - m0;


  /*//m0 = getResident() * getpagesize();
  for (int i = 0; i < num_of_ids; i++) {
    ev->LookupOrCreateKey(key_list[i], &value_ptr,
                          &is_filter, false);    
  }
  m1 = getResident() * getpagesize();
  LOG(INFO)<<"Step 2: Memory used: "<<m1 - m0;*/
	
  delete[] key_list;
}
} //namespace embedding
} //namespace tensorflow