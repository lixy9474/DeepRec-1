#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedEmbeddingLocalSparseLookUp")
    .Attr("T: {float32}")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Input("emb_variable: T")
    .Output("emb_vectors: T")
    .Output("sp_values_offset: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle temp;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 1, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 2, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 1, &temp));
      ShapeHandle emb_var_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(3), 2, &emb_var_shape));

      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_var_shape, 1);
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_vec_size_dim});
      ctx->set_output(0, output_shape);

      return Status::OK();
    });

REGISTER_OP("FusedEmbeddingLocalSparseLookUpGrad")
    .Attr("T: {float32}")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("top_grad: T")
    .Input("emb_variable: T")
    .Input("sp_values: int64")
    .Input("sp_values_offset: int32")
    .Output("grad_emb_weight_sp_values: T")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle top_grad_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &top_grad_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(top_grad_shape, 1);
      ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim}));
      return Status::OK();
    })
    .Doc(R"doc(
The gradient ops for FusedEmbeddingLocalSparseLookUp. sp_values_offset from the forward op
need to be passed to this grad op as input.
    )doc");

REGISTER_OP("FusedEmbeddingSparsePreLookUp")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Input("partition_shapes: num_partitions * int64")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Output("partitioned_values: num_partitions * int64")
    .Output("partitioned_indices: num_partitions * int64")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));
      int partition_axis;
      TF_RETURN_IF_ERROR(ctx->GetAttr("partition_axis", &partition_axis));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(int(num_partitions)), 1, &unused));
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(int(num_partitions) + 1), 2, &unused));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));

      for (int i = 0; i < int(num_partitions); i++) {
        ShapeHandle partition_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 1, &partition_shape));
        TF_RETURN_IF_ERROR(
            ctx->WithValue(ctx->NumElements(partition_shape), 2, &unused_dim));

        ShapeHandle values_result_shape, indices_result_shape;
        if (int(partition_axis) == 0) {
          values_result_shape = ctx->MakeShape({ctx->UnknownDim()});
          indices_result_shape = ctx->MakeShape({ctx->UnknownDim(), 2});
        } else {
          return errors::InvalidArgument("partition_axis > 0 not implemented!");
        }
        ctx->set_output(i, values_result_shape);
        ctx->set_output(i + int(num_partitions), indices_result_shape);
      }
      return Status::OK();
    })
    .Doc(R"doc(
A fused embedding op, usually using for partitioned and distriuted embedding variables.
FusedEmbeddingSparsePreLookUp, FusedEmbeddingSparsePostLookUp should be used together.
This op will first read the partition pattern of embedding variables through partition_shapes,
then sort, re-calculate and assign the embedding indices to the corresponding partition. Several Gather ops
usually should be appended after this op to gather embedding shards from multiple partitioned embedding
variables. This op has no gradient function.
    )doc");

REGISTER_OP("FusedEmbeddingSparsePostLookUp")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("emb_shards: num_partitions * T")
    .Input("partitioned_indices: num_partitions * int64")
    .Input("sp_dense_shape: int64")
    .Input(
        "partitioned_values: num_partitions * int64")  // only for backward use.
                                                       // actually directly port
                                                       // to python grad op
                                                       // output
    .Output("emb_vectors: T")
    .Output("feature_nums: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      ShapeHandle first_emb_shard_shape;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(0), 2, &first_emb_shard_shape));

      ShapeHandle unused;
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 2, &unused));
        TF_RETURN_IF_ERROR(
            ctx->WithRank(ctx->input(i + num_partitions), 2, &unused));
      }

      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2 * num_partitions), 1, &unused));

      DimensionHandle emb_vec_size_dim = ctx->Dim(first_emb_shard_shape, 1);
      ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim}));
      ctx->set_output(1, ctx->MakeShape({ctx->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(
A fused embedding op, usually using for partitioned and distriuted embedding variables.
FusedEmbeddingSparsePreLookUp, FusedEmbeddingSparsePostLookUp should be used together.
There should be several Gather ops before this op. The Gather ops gather embedding shards from
embedding variable and this op glue them together, then apply combiner and max_morm according to
embedding indices.
    )doc");

REGISTER_OP("FusedEmbeddingSparsePostLookUpGrad")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("top_grad: T")
    .Input("emb_shards: num_partitions * T")
    .Input("partitioned_indices: num_partitions * int64")
    .Input("feature_nums: int32")
    .Output("grad_shards: num_partitions * T")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      ShapeHandle unused;
      ShapeHandle top_grad_shape;

      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &top_grad_shape));
      for (int i = 1; i < 2 * num_partitions + 1; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 2, &unused));
      }
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2 * num_partitions + 1), 1, &unused));

      DimensionHandle emb_vec_size_dim = ctx->Dim(top_grad_shape, 1);

      ShapeHandle output_shape =
          ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      for (int i = 0; i < num_partitions; i++) {
        ctx->set_output(i, output_shape);
      }
      return Status::OK();
    })
    .Doc(R"doc(
Calculate gradient of FusedEmbeddingSparsePostLookUp
    )doc");

}  // namespace tensorflow
