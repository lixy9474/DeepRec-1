/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename TKey, typename TValue>
class EVExportL2WeightOp : public OpKernel {
 public:
  explicit EVExportL2WeightOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {ev->Size()}, &output));
    //No matter what device is used, 
    //only need worker_threads in EmbeddingVarContext<CPUDevice>.
    EmbeddingVarContext<CPUDevice> ev_ctx(ctx);
    if (!ev->IsSingleHbm()) {
      ev->GetL2WeightSnapshot(ev_ctx, (TValue*)output->data());
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportL2Weight")              \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),  \
                          EVExportL2WeightOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportL2Weight")                  \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          EVExportL2WeightOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class EVExportFrequencyOp : public OpKernel {
 public:
  explicit EVExportFrequencyOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {ev->Size()}, &output));
    //No matter what device is used, 
    //only need worker_threads in EmbeddingVarContext<CPUDevice>.
    EmbeddingVarContext<CPUDevice> ev_ctx(ctx);
    if (!ev->IsSingleHbm()) {
      ev->GetFrequencySnapshot(ev_ctx, (int64*)output->data());
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportFrequency")              \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),  \
                          EVExportFrequencyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportFrequency")                  \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          EVExportFrequencyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class EVExportVersionOp : public OpKernel {
 public:
  explicit EVExportVersionOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {ev->Size()}, &output));
    //No matter what device is used, 
    //only need worker_threads in EmbeddingVarContext<CPUDevice>.
    EmbeddingVarContext<CPUDevice> ev_ctx(ctx);
    if (!ev->IsSingleHbm()) {
      ev->GetVersionSnapshot(ev_ctx, (int64*)output->data());
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportVersion")              \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),  \
                          EVExportVersionOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                          \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("EVExportVersion")               \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          EVExportVersionOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                          \
  REGISTER_KERNELS(int64, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class EVExportKeyOp : public OpKernel {
 public:
  explicit EVExportKeyOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {ev->Size()}, &output));
    //No matter what device is used, 
    //only need worker_threads in EmbeddingVarContext<CPUDevice>.
    if (!ev->IsSingleHbm()) {
      ev->GetKeySnapshot((TKey*)output->data());
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportKey")                  \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),  \
                          EVExportKeyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExportKey")                  \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          EVExportKeyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA
}  // namespace tensorflow

