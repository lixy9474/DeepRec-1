# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Operations for embeddings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
# Imports gradient definitions.
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


def _clip(params, ids, max_norm):
  """Helper function for _embedding_lookup_and_transform.

  This function optionally clips embeddings to an l2-norm of max_norm.

  Args:
    params: A `Tensor` of embeddings retrieved by `gather`.
    ids: The `ids` argument that was passed to `gather`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` with the same type as `params`.
  """

  def _rank(x):
    """Helper function to retrieve the rank of a tensor.

    Args:
      x: Something convertible to `Tensor`.

    Returns:
      Either a pair `(rank, True)` where `rank` is an integer or a pair
      `(rank, False)` where `rank` is an integer `Tensor`. In either case,
      `rank` is the rank of `x`.
    """
    rank = ops.convert_to_tensor(x).get_shape().ndims
    if rank:
      return rank, True
    else:
      return array_ops.rank(x), False

  if max_norm is None:
    return params
  ids_rank, ids_static = _rank(ids)
  params_rank, params_static = _rank(params)
  return clip_ops.clip_by_norm(
      params,
      max_norm,
      axes=(list(range(ids_rank, params_rank)) if ids_static and params_static
            else math_ops.range(ids_rank, params_rank)))

def _gather_fae(ids, blocknums, embs, params):
  concat_embs=[]
  indices = math_ops.range(0, array_ops.squeeze(array_ops.shape(ids)), 1)
  for i in range(len(embs)):
    indice_cnt = array_ops.expand_dims(array_ops.boolean_mask(indices, math_ops.greater_equal(blocknums, i+1)), 1)
    #scatter_shape=tensor_shape.TensorShape([ids.get_shape()[0], params._ev_list[i].shape()[0]])
    concat_emb=array_ops.scatter_nd(indices=indice_cnt, updates=embs[i], shape=array_ops.shape(embs[0]))
    concat_embs.append(concat_emb)
  return array_ops.concat(concat_embs, 1)

def _embedding_lookup_and_transform(params,
                                    ids,
                                    partition_strategy="mod",
                                    name=None,
                                    max_norm=None,
                                    transform_fn=None,
                                    ev_init_value=None,
                                    blocknums=None,
                                    counts=None):
  """Helper function for embedding_lookup and _compute_sampled_logits.

  This function is a generalization of embedding_lookup that optionally
  applies a caller-specified transformation to each embedding. This is
  done through the `transform_fn` argument. If provided, the function is
  applied to each partitioned tensor of retrieved embeddings, colocated
  with the embeddings. This function will be called with a single `Tensor`
  argument of the same type as the `params` tensor and should return a
  `Tensor`. The shape of the argument will be the same as `params` except
  for the size of the first dimension. The first dimension of the result's
  shape must be the same size as the argument's.

  Args:
    params: See embedding_lookup.
    ids: See embedding_lookup.
    partition_strategy: See embedding_lookup.
    name: See embedding_lookup.
    max_norm: See embedding_lookup.
    transform_fn: An optional function to apply to each retrieved embedding. If
      max_norm is provided, transform_fn is applied to the norm-limited
      embeddings.

  Returns:
    See embedding_lookup for details.
  Raises:
    ValueError: If `params` is empty.
  """
  from tensorflow.python.ops.hash_table import hash_table
  from tensorflow.python.ops.hash_table import embedding
  if isinstance(params, hash_table.HashTable) or isinstance(params, hash_table.DistributedHashTable):
    return embedding.embedding_lookup(params, ids, name=name)[0]
  if isinstance(params, list) and len(params) == 1:
    if isinstance(params[0], hash_table.HashTable) or isinstance(params[0], hash_table.DistributedHashTable):
      return embedding.embedding_lookup(params[0], ids, name=name)[0]
  if params is None:
    raise ValueError("params must be specified")
  if isinstance(params, (list, tuple)) and not params:
    raise ValueError("Need at least one param")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if isinstance(params[0], kv_variable_ops.MultiHashVariable):
    if params[0].mhvconfig.strategy == "Q-R":
      ids_tensor = ops.convert_to_tensor(ids, dtypes.int64)
      ids_Q = math_ops.floordiv(ids_tensor, params[0].mhvconfig.size[0][0])
      ids_R = math_ops.floormod(ids_tensor, params[0].mhvconfig.size[1][0])
      result_Q = _embedding_lookup_and_transform(params[0]._val_list[0], ids_Q)
      result_R = _embedding_lookup_and_transform(params[0]._val_list[1], ids_R)
      if params[0].mhvconfig.operation == "add":
        return math_ops.add(result_Q, result_R)
      if params[0].mhvconfig.operation == "mul":
        return math_ops.multiply(result_Q, result_R)
      if params[0].mhvconfig.operation == "concat":
        return array_ops.concat([result_Q, result_R], 1)


  with ops.name_scope(name, "embedding_lookup", params + [ids]) as name:
    np = len(params)  # Number of partitions
    # Preserve the resource variable status to avoid accidental dense reads.
    if not any(
        isinstance(p, resource_variable_ops.ResourceVariable) for p in params):
      params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    ids = ops.convert_to_tensor(ids, name="ids")
    if np == 1 and (not transform_fn or ids.get_shape().ndims == 1):
      if isinstance(params[0], kv_variable_ops.DynamicEmbeddingVariable):
        if blocknums is None:
          raise ValueError("blocknums must be valid for dynamic embedding variable")
        ids_nozero = array_ops.boolean_mask(ids, math_ops.greater_equal(blocknums, 1))
        blocknums_nozero = array_ops.boolean_mask(blocknums, math_ops.greater_equal(blocknums, 1))
        with ops.colocate_with(params[0].mainev()):
          embs = params[0].sparse_read(ids_nozero, blocknums_nozero)
        embs_nozero = _gather_fae(ids_nozero, blocknums_nozero, embs, params[0])
        indices = math_ops.range(0, array_ops.squeeze(array_ops.shape(ids)), 1)
        indice_cnt = array_ops.expand_dims(array_ops.boolean_mask(indices, math_ops.greater_equal(blocknums, 1)), 1)
        return array_ops.scatter_nd(indices=indice_cnt, updates=embs_nozero, shape=[array_ops.shape(ids)[0], array_ops.shape(embs_nozero)[1]])
      else:
        with ops.colocate_with(params[0]):
          result = _clip(array_ops.gather(params[0], ids, name=name,
                                          ev_init_value=ev_init_value,
                                          counts=counts),
                         ids, max_norm)
          if transform_fn:
            result = transform_fn(result)
          return result
      # Make sure the final result does not have colocation contraints on the
      # params. Similar to the case np > 1 where parallel_dynamic_stitch is
      # outside the scioe of all with ops.colocate_with(params[p]).
      return array_ops.identity(result)
    else:
      # Flatten the ids. There are two cases where we need to do this.
      # - There is more than one params tensor.
      # - There is a transform_fn and ids is not statically known to be 1-D.
      #   We must flatten in this case because transform_fn expects a flat
      #   tensor of embeddings.
      flat_ids = array_ops.reshape(ids, [-1])
      original_indices = math_ops.range(array_ops.size(flat_ids))

      # Create p_assignments and set new_ids depending on the strategy.

      if blocknums is None and isinstance(params[0], kv_variable_ops.DynamicEmbeddingVariable):
        raise ValueError("blocknums must be valid for dynamic embedding variable")


      if isinstance(params[0], kv_variable_ops.EmbeddingVariable):
         new_ids = flat_ids
         p_assignments = flat_ids % 1000 % np 
      elif partition_strategy == "mod":
        p_assignments = flat_ids % np
        new_ids = flat_ids // np
      elif partition_strategy == "div":
        # Compute num_total_ids as the sum of dim-0 of params, then assign to
        # partitions based on a constant number of ids per partition. Optimize
        # if we already know the full shape statically.
        dim_0_size = tensor_shape.Dimension(
            tensor_shape.dimension_value(params[0].get_shape()[0]))
        for p in xrange(1, np):
          dim_0_size += tensor_shape.Dimension(
              tensor_shape.dimension_value(params[p].get_shape()[0]))
        if dim_0_size.value:
          num_total_ids = constant_op.constant(dim_0_size.value, flat_ids.dtype)
        else:
          dim_0_sizes = []
          for p in xrange(np):
            param_p_dim = tensor_shape.dimension_value(params[p].get_shape()[0])
            if param_p_dim is not None:
              dim_0_sizes.append(param_p_dim)
            else:
              with ops.colocate_with(params[p]):
                dim_0_sizes.append(array_ops.shape(params[p])[0])
          num_total_ids = math_ops.reduce_sum(
              math_ops.cast(array_ops.stack(dim_0_sizes), flat_ids.dtype))
        ids_per_partition = num_total_ids // np
        extras = num_total_ids % np

        p_assignments = math_ops.maximum(flat_ids // (ids_per_partition + 1),
                                         (flat_ids - extras) //
                                         ids_per_partition)

        # Emulate a conditional using a boolean indicator tensor
        new_ids = array_ops.where(p_assignments < extras,
                                  flat_ids % (ids_per_partition + 1),
                                  (flat_ids - extras) % ids_per_partition)
      else:
        raise ValueError("Unrecognized partition strategy: " +
                         partition_strategy)

      # Cast partition assignments to int32 for use in dynamic_partition.
      # There really should not be more than 2^32 partitions.
      p_assignments = math_ops.cast(p_assignments, dtypes.int32)
      # Partition list of ids based on assignments into np separate lists
      gather_ids = data_flow_ops.dynamic_partition(new_ids, p_assignments, np)
      gather_blocknums = None
      gather_ev_init_value = None
      if isinstance(params[0], kv_variable_ops.DynamicEmbeddingVariable): 
        gather_blocknums = data_flow_ops.dynamic_partition(blocknums, p_assignments, np)
      if ev_init_value is not None:
        gather_ev_init_value = data_flow_ops.dynamic_partition(ev_init_value, p_assignments, np)
      # Similarly, partition the original indices.
      pindices = data_flow_ops.dynamic_partition(original_indices,
                                                 p_assignments, np)
      # Do np separate lookups, finding embeddings for plist[p] in params[p]
      partitioned_result = []
      for p in range(np):
        pids = gather_ids[p]
        if isinstance(params[p], kv_variable_ops.DynamicEmbeddingVariable):
          pblocknums = gather_blocknums[p]
          embs = []
          pids_nozero = array_ops.boolean_mask(pids, math_ops.greater_equal(pblocknums , 1))
          pblocknums_nozero = array_ops.boolean_mask(pblocknums, math_ops.greater_equal(pblocknums, 1))
          for i in range(params[p].blocknum()):
            with ops.colocate_with(params[p]._ev_list[i]):
              evids = array_ops.boolean_mask(pids_nozero, math_ops.greater_equal(pblocknums_nozero, i + 1))
              gathered_emb = params[p]._ev_list[i].sparse_read(evids, name=None)
              embs.append(gathered_emb)
          result_nozero = _gather_fae(pids_nozero, pblocknums_nozero, embs, params[p])
          # suplement blocknum equal to zero
          indices = math_ops.range(0, array_ops.squeeze(array_ops.shape(pids)), 1)
          indice_cnt = array_ops.expand_dims(array_ops.boolean_mask(indices, math_ops.greater_equal(pblocknums, 1)), 1)
          result = array_ops.scatter_nd(indices=indice_cnt, updates=result_nozero, shape=[array_ops.shape(pids)[0], array_ops.shape(result_nozero)[1]])
          partitioned_result.append(result)
        else:
          with ops.colocate_with(params[p]):
            if ev_init_value is None:
              new_ev_init_value = None
            else:
              new_ev_init_value = gather_ev_init_value[p]
            result = array_ops.gather(params[p], pids, ev_init_value=new_ev_init_value, counts=counts)
            if transform_fn:
              # If transform_fn is provided, the clip_by_norm precedes
              # the transform and hence must be co-located. See below
              # for the counterpart if transform_fn is not proveded.
              result = transform_fn(_clip(result, pids, max_norm))
            partitioned_result.append(result)
      # Stitch these back together
      ret = data_flow_ops.parallel_dynamic_stitch(
          pindices, partitioned_result, name=name)

      # Determine the static element shape.
      if isinstance(params[0], kv_variable_ops.EmbeddingVariable) or \
        isinstance(params[0], kv_variable_ops.DynamicEmbeddingVariable):
        if transform_fn is None:
          element_shape_s = params[0].get_shape()[:]
          for p in params[1:]:
            element_shape_s = element_shape_s.merge_with(p.get_shape()[:])
        else:
          element_shape_s = ret.get_shape()[:]
      else:
        if transform_fn is None:
          element_shape_s = params[0].get_shape()[1:]
          for p in params[1:]:
            element_shape_s = element_shape_s.merge_with(p.get_shape()[1:])
        else:
          element_shape_s = ret.get_shape()[1:]


      # Compute the dynamic element shape.
      if element_shape_s.is_fully_defined():
        element_shape_d = element_shape_s
      elif transform_fn is None:
        # It's important that we compute params[0].shape on the right device
        # to avoid data motion.
        with ops.colocate_with(params[0]):
          params_shape = array_ops.shape(params[0])
        element_shape_d = params_shape[1:]
      else:
        element_shape_d = array_ops.shape(ret)[1:]

      # Reshape to reverse the flattening of ids.
      ret = array_ops.reshape(
          ret, array_ops.concat([array_ops.shape(ids), element_shape_d], 0))

      # Normally the reshape is sufficient, but setting shape explicitly
      # teaches shape inference that params[1:].get_shape() matters
      # (in the case that transform_fn is None).
      ret.set_shape(ids.get_shape().concatenate(element_shape_s))
      if not transform_fn:
        # If transform_fn was provided, the clip_by_norm was done above.
        ret = _clip(ret, ids, max_norm)
      return ret


@tf_export(v1=["nn.embedding_lookup"])
def embedding_lookup(
    params,
    ids,
    partition_strategy="mod",
    name=None,
    validate_indices=True,  # pylint: disable=unused-argument
    max_norm=None,
    ev_init_value=None,
    blocknums=None,
    counts=None):
  """Looks up `ids` in a list of embedding tensors.

  This function is used to perform parallel lookups on the list of
  tensors in `params`.  It is a generalization of
  `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.  `params` may be
  a `PartitionedVariable` as returned by using `tf.compat.v1.get_variable()`
  with a
  partitioner.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` according to the `partition_strategy`.
  In all strategies, if the id space does not evenly divide the number of
  partitions, each of the first `(max_id + 1) % len(params)` partitions will
  be assigned one more id.

  If `partition_strategy` is `"mod"`, we assign each id to partition
  `p = id % len(params)`. For instance,
  13 ids are split across 5 partitions as:
  `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

  If `partition_strategy` is `"div"`, we assign ids to partitions in a
  contiguous manner. In this case, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for the operation (optional).
    validate_indices: DEPRECATED. If this operation is assigned to CPU, values
      in `indices` are always validated to be within range.  If assigned to GPU,
      out-of-bound indices result in safe but unspecified behavior, which may
      include raising an error.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """
  return _embedding_lookup_and_transform(
      params=params,
      ids=ids,
      partition_strategy=partition_strategy,
      name=name,
      max_norm=max_norm,
      transform_fn=None,
      ev_init_value=ev_init_value,
      blocknums=blocknums,
      counts=counts)


@tf_export("nn.embedding_lookup", v1=[])
def embedding_lookup_v2(params, ids, max_norm=None, name=None):
  """Looks up `ids` in a list of embedding tensors.

  This function is used to perform parallel lookups on the list of
  tensors in `params`.  It is a generalization of
  `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.  `params` may be
  a `PartitionedVariable` as returned by using `tf.compat.v1.get_variable()`
  with a
  partitioner.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` according to the `partition_strategy`.
  In all strategies, if the id space does not evenly divide the number of
  partitions, each of the first `(max_id + 1) % len(params)` partitions will
  be assigned one more id.

  The `partition_strategy` is always `"div"` currently. This means that we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the 'div' `partition_strategy`.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """
  return embedding_lookup(params, ids, "div", name, max_norm=max_norm)


def _tile_combine_embedding(embeddings, segment_ids, column_ids, sp_shape):
  column_ids = math_ops.cast(column_ids, dtypes.int32)
  sp_shape = math_ops.cast(sp_shape, dtypes.int32)
  segment_ids = segment_ids * sp_shape[1] + column_ids
  total_size = sp_shape[0] * sp_shape[1]
  embeddings = math_ops.unsorted_segment_sum(embeddings, segment_ids, total_size)
  embeddings = array_ops.reshape(
      embeddings, [sp_shape[0], sp_shape[1] * array_ops.shape(embeddings)[-1]])
  return embeddings


@tf_export(v1=["nn.embedding_lookup_sparse"])
def embedding_lookup_sparse(params,
                            sp_ids,
                            sp_weights,
                            partition_strategy="mod",
                            name=None,
                            combiner=None,
                            max_norm=None,
                            blocknums=None):
  """Computes embeddings for the given ids and weights.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary.
    sp_weights: either a `SparseTensor` of float / double weights, or `None` to
      indicate all weights should be taken to be 1. If specified, `sp_weights`
      must have exactly the same shape and indices as `sp_ids`.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn",
      "tile" and "sum" are supported. "sum" computes the weighted sum of the
      embedding results for each row. "mean" is the weighted sum divided by the
      total weight. "sqrtn" is the weighted sum divided by the square root of the
      sum of the squares of the weights.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ..., dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner is None:
    logging.warn("The default value of combiner will change from \"mean\" "
                 "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum", "tile"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [sp_ids]) as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    if isinstance(params[0], kv_variable_ops.EmbeddingVariable) and params[0]._filter_freq == 0:
      ids, idx = array_ops.unique(ids)
      counts = None
    else:
      ids, idx, counts = array_ops.unique_with_counts(ids)

    uniqued_blocknums = None
    if blocknums is not None:
      if idx is None:
        raise ValueError("blocknums now require unqiue index to be generagted")
      else:
        uniqued_blocknums = math_ops.unsorted_segment_max(blocknums, idx, array_ops.squeeze(array_ops.shape(ids), 0))
    embeddings = embedding_lookup(
        params, ids, partition_strategy=partition_strategy, max_norm=max_norm,
        blocknums=uniqued_blocknums, counts = counts)
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      embeddings = math_ops.cast(embeddings, dtypes.float32)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      embeddings = array_ops.gather(embeddings, idx)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
      elif combiner == "tile":
        column_ids = sp_ids.indices[:, 1]
        embeddings = _tile_combine_embedding(embeddings,
                                             segment_ids,
                                             column_ids,
                                             sp_ids.dense_shape)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = math_ops.sparse_segment_sum(
            embeddings, idx, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(
            embeddings, idx, segment_ids, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(
            embeddings, idx, segment_ids, name=name)
      elif combiner == "tile":
        embeddings = array_ops.gather(embeddings, idx)
        column_ids = sp_ids.indices[:, 1]
        embeddings = _tile_combine_embedding(embeddings,
                                             segment_ids,
                                             column_ids,
                                             sp_ids.dense_shape)
      else:
        assert False, "Unrecognized combiner"

    return embeddings

@tf_export(v1=["nn.adaptive_embedding_lookup_sparse"])
def adaptive_embedding_lookup_sparse(hash_params,
                                     ev_params,
                                     sp_ids,
                                     hash_ev_ids,
                                     sp_weights,
                                     partition_strategy="mod",
                                     name=None,
                                     combiner=None,
                                     max_norm=None,
                                     bucket_size=None,
                                     adaptive_mask_tensor=None,
                                     blocknums=None):
  """Computes embeddings for the given ids and weights.
  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.
  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.
  Args:
    hash_params: A single tensor representing the complete embedding tensor,
      by normal Variable.
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    ev_params: A single tensor representing the complete embedding tensor,
      by EmbeddingVariable
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
      where N is typically batch size and M is arbitrary.
    sp_weights: either a SparseTensor of float / double weights, or None to
      indicate all weights should be taken to be 1. If specified, sp_weights
      must have exactly the same shape and indices as sp_ids.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.
      "sqrtn" is the weighted sum divided by the square root of the sum of the
      squares of the weights.
    max_norm: If provided, each embedding is normalized to have l2 norm equal
      to max_norm before combining.
  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by sp_ids, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.
    In other words, if
      shape(combined params) = [p0, p1, ..., pm]
    and
      shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]
    then
      shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].
    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
    with `combiner`="mean", then the output will be a 3x20 matrix where
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = params[0, :] * 1.0
      output[2, :] = params[1, :] * 3.0
  Raises:
    TypeError: If sp_ids is not a SparseTensor, or if sp_weights is neither
      None nor SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner is None:
    #logging.warn("The default value of combiner will change from \"mean\" "
    #             "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  # convert hash and ev to list
  if isinstance(hash_params, variables.PartitionedVariable):
    hash_params = list(hash_params)  # Iterate to get the underlying Variables.
  if not isinstance(hash_params, list):
    hash_params = [hash_params]
  if isinstance(ev_params, variables.PartitionedVariable):
    ev_params = list(ev_params)  # Iterate to get the underlying Variables.
  if not isinstance(ev_params, list):
    ev_params = [ev_params]
  if len(hash_params) < 1 or len(ev_params) < 1:
    raise ValueError("Missing hash_params: %s, ev_params:." % hash_params, ev_params)

  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.
  if not ignore_weights:
    raise ValueError("AdaptiveEmbedding lookup not support not ignore weights")
  if adaptive_mask_tensor is None:
     raise ValueError("AdaptiveEmbedding lookup not support not ignore weights")
  with ops.name_scope(name, "embedding_lookup_sparse",
                      ev_params + [sp_ids]) as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)
    ids = sp_ids.values
    flat_ids = array_ops.reshape(ids, [-1])
    original_indices = math_ops.range(array_ops.size(flat_ids))
    parts = data_flow_ops.dynamic_partition(original_indices, adaptive_mask_tensor,  2)
    spids_part = data_flow_ops.dynamic_partition(flat_ids, adaptive_mask_tensor,  2)

    hash_ids, hash_idx = array_ops.unique(spids_part[0])
    #ev_ids, ev_idx = array_ops.unique(spids_part[1])

    hash_embeddings = embedding_lookup(
        hash_params, hash_ids, partition_strategy=partition_strategy, max_norm=max_norm,
        blocknums=None)
    ev_init_value = embedding_lookup(
        hash_params, hash_ev_ids, partition_strategy=partition_strategy, max_norm=max_norm,
        blocknums=None)
    ev_embeddings = embedding_lookup(
        ev_params, spids_part[1], partition_strategy=partition_strategy, max_norm=max_norm,
        ev_init_value=ev_init_value,
        blocknums=None)
    if (hash_idx is not None):
      hash_segment_ids = math_ops.range(0, array_ops.squeeze(array_ops.shape(hash_idx)), 1)
      #ev_segment_ids = math_ops.range(0, array_ops.squeeze(array_ops.shape(spids_part[1])), 1)
      if combiner == "sum":
        hash_embeddings = math_ops.sparse_segment_sum(
            hash_embeddings, hash_idx, hash_segment_ids, name=name+"_hash")
        #ev_embeddings = math_ops.sparse_segment_sum(
        #    ev_embeddings, ev_idx, ev_segment_ids, name=name+"_ev")
      elif combiner == "mean":
        hash_embeddings = math_ops.sparse_segment_mean(
            hash_embeddings, hash_idx, hash_segment_ids, name=name+"_hash")
        #ev_embeddings = math_ops.sparse_segment_mean(
        #    ev_embeddings, ev_idx, ev_segment_ids, name=name+"_ev")
      elif combiner == "sqrtn":
        hash_embeddings = math_ops.sparse_segment_sqrt_n(
            hash_embeddings, hash_idx, hash_segment_ids, name=name+"_hash")
        #ev_embeddings = math_ops.sparse_segment_sqrt_n(
        #    ev_embeddings, ev_idx, ev_segment_ids, name=name+"_ev")
      else:
        assert False, "Unrecognized combiner"
    else:
      if combiner == "sum":
        embeddings = math_ops.segment_sum(
            embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_mean(
            embeddings, segment_ids, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sqrt_n(
            embeddings, segment_ids, name=name)
      else:
        assert False, "Unrecognized combiner"

    embeddings_result = data_flow_ops.dynamic_stitch(parts, [hash_embeddings, ev_embeddings])
    return embeddings_result

@tf_export("nn.embedding_lookup_sparse", v1=[])
def embedding_lookup_sparse_v2(params,
                               sp_ids,
                               sp_weights,
                               combiner=None,
                               max_norm=None,
                               name=None):
  """Computes embeddings for the given ids and weights.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for ``"div"`` `partition_strategy`.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary.
    sp_weights: either a `SparseTensor` of float / double weights, or `None` to
      indicate all weights should be taken to be 1. If specified, `sp_weights`
      must have exactly the same shape and indices as `sp_ids`.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn",
      "tile" and "sum" are supported. "sum" computes the weighted sum of the
      embedding results for each row. "mean" is the weighted sum divided by the
      total weight. "sqrtn" is the weighted sum divided by the square root of the
      sum of the squares of the weights.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    name: Optional name for the op.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ..., dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  return embedding_lookup_sparse(params, sp_ids, sp_weights, "div", name,
                                 combiner, max_norm)


@tf_export("nn.embedding_lookup_sparse_multi_dim")
def embedding_lookup_sparse_multi_dim(params,
                                      sp_ids,
                                      sp_weights,
                                      partition_strategy="mod",
                                      name=None,
                                      combiners=None,
                                      max_norm=None,
                                      weight_axis=-1):
  """Computes embeddings for the given ids and weights like
     embedding_lookup_sparse.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for ``"div"`` `partition_strategy`.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary.
    sp_weights: either a `SparseTensor` of float / double weights, or `None` to
      indicate all weights should be taken to be 1. If specified, `sp_weights`
      must have exactly the same shape and indices as `sp_ids`.
    combiners: A list of string specifying the reduction op. Currently "mean",
      "sqrtn", "tile" and "sum" are supported. "sum" computes the weighted sum of
      the embedding results for each row. "mean" is the weighted sum divided by the 
      total weight. "sqrtn" is the weighted sum divided by the square root of the 
      sum of the squares of the weights.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    name: Optional name for the op.
    weight_axis: Specify axis to use weight.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum", "tile"}.
  """

  if combiners is None:
    logging.warn("The default value of combiner will change from \"mean\" "
                 "to \"sqrtn\" after 2016/11/01.")
    combiners = ["mean"]
  if not isinstance(combiners, (list, tuple)):
    combiners = (combiners,)
  for comb in combiners:
    if comb not in ("mean", "sqrtn", "sum", "max", "min", "tile"):
      raise ValueError("combiner must be one of 'mean', 'sqrtn', 'sum', 'max' or 'min'")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  if not len(sp_ids.shape) == len(combiners) + 1:
    raise ValueError("SparseTensor to embedding lookup rank should be combiner nums -1,"
        "sparse tensor rank: {}, num combiners: {}".format(len(sp_ids.shape), len(combiners)))
  if weight_axis is None:
    weight_axis = -1
  if weight_axis < 0:
    weight_axis = len(sp_ids.shape) + weight_axis
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [sp_ids]) as name:
    ids = sp_ids.values
    ids, idx = array_ops.unique(ids)
    embeddings = embedding_lookup(
        params, ids, partition_strategy=partition_strategy, max_norm=max_norm)
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      embeddings = math_ops.to_float(embeddings)
    weights = None if sp_weights is None else sp_weights.values
    embeddings, _ = _combine_embedding(embeddings,
                                       sp_ids.indices,
                                       sp_ids.dense_shape,
                                       combiners,
                                       unique_idx=idx,
                                       weights=weights,
                                       weight_axis=weight_axis,
                                       name=name)
    return embeddings


def _internal_combine(embeddings, segment_ids, combiner,
                      weights=None, max_size=None, seg_offset=None,
                      use_weight=False,
                      name=None):
  if combiner == "sum":
    embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
  elif combiner == "mean":
    if use_weight:
      embeddings = math_ops.segment_sum(embeddings, segment_ids)
      weight_sum = math_ops.segment_sum(weights, segment_ids)
      embeddings = math_ops.div(embeddings, weight_sum, name=name)
    else:
      embeddings = math_ops.segment_mean(embeddings, segment_ids, name=name)
  elif combiner == "sqrtn":
    embeddings = math_ops.segment_sum(embeddings, segment_ids)
    if use_weight:
      weights = math_ops.pow(weights, 2)
    else:
      weights = array_ops.ones_like(segment_ids)
    weight_sum = math_ops.segment_sum(weights, segment_ids)
    weight_sum_sqrt = math_ops.sqrt(weight_sum)
    embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
  elif combiner == "max":
    embeddings = math_ops.segment_max(embeddings, segment_ids, name=name)
  elif combiner == "min":
    embeddings = math_ops.segment_min(embeddings, segment_ids, name=name)
  elif combiner == "tile":
    assert seg_offset is not None and max_size is not None, \
        "seg_offset or max_size not set when combine with tile"
    seg_offset = math_ops.cast(seg_offset, dtypes.int32)
    max_size = math_ops.cast(max_size, dtypes.int32)
    dynamic_ids = seg_offset + segment_ids * max_size
    full_size = (math_ops.reduce_max(segment_ids) + 1) * max_size
    embeddings = math_ops.unsorted_segment_sum(embeddings, dynamic_ids, full_size)
    embeddings = array_ops.reshape(
        embeddings, [-1, array_ops.shape(embeddings)[-1] * max_size])
  else:
    assert False, "Unrecognized combiner"
  if weights is not None:
    weights = math_ops.segment_mean(weights, segment_ids)
  return embeddings, weights


def _get_valid_embeddings(embeddings, weights, segment_ids, cur_indices, next_segment_ids):
  valid_index, valid_idx = array_ops.unique(next_segment_ids)
  embeddings = array_ops.gather(embeddings, valid_index)
  weights = array_ops.gather(weights, valid_index)
  segment_ids = math_ops.segment_max(segment_ids, valid_idx)
  cur_indices = math_ops.segment_max(cur_indices, valid_idx)
  return embeddings, weights, segment_ids, cur_indices


def _combine_embedding(embeddings,
                       indices,
                       dense_shape,
                       combiners,
                       segment_ids=None,
                       unique_idx=None,
                       weights=None,
                       weight_axis=1,
                       name=None):
  assert weight_axis > 0, "weight_axis should more than 1 in " \
      "_internal_embedding_combine, current weight_axis: {}".format(weight_axis)
  if segment_ids is None:
    segment_ids = indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  embeddings = array_ops.gather(embeddings, unique_idx)
  if weights is None:
    use_weight = False
    weights = array_ops.ones([array_ops.shape(embeddings)[0]], dtype=dtypes.float32)
  else:
    use_weight = True
  if weights.dtype != embeddings.dtype:
    weights = math_ops.cast(weights, embeddings.dtype)
  # Reshape weights to allow broadcast
  ones = array_ops.fill(
      array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
  bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                         0)
  
  orig_weights_shape = weights.get_shape()
  weights = array_ops.reshape(weights, bcast_weights_shape)
  
  # Set the weight shape, since after reshaping to bcast_weights_shape,
  # the shape becomes None.
  if embeddings.get_shape().ndims is not None:
    weights.set_shape(
        orig_weights_shape.concatenate(
            [1 for _ in range(embeddings.get_shape().ndims - 1)]))
  embeddings *= weights
  segment_ids_list = [segment_ids]
  for i in range(len(combiners) - 1):
    tmp_indices = math_ops.cast(indices[:, i + 1], dtypes.int32)
    segment_ids = segment_ids * math_ops.cast(dense_shape[i + 1], dtypes.int32) + tmp_indices
    segment_ids_list.append(segment_ids)
  for i in range(len(combiners)):
    axis = len(combiners) - i
    if not i == 0:
      cur_indices = indices[:, axis]
      embeddings, weights, segment_ids, cur_indice_offset = \
          _get_valid_embeddings(embeddings,
                                weights,
                                segment_ids_list[axis - 1],
                                cur_indices,
                                segment_ids_list[axis])
    else:
      cur_indice_offset = indices[:, axis]
      segment_ids = segment_ids_list[axis - 1]
    embeddings, weights = _internal_combine(embeddings,
                                            segment_ids,
                                            combiners[axis - 1],
                                            weights=weights,
                                            max_size=dense_shape[axis],
                                            seg_offset=cur_indice_offset,
                                            use_weight=use_weight and (weight_axis == axis),
                                            name=name + str(axis))
  return embeddings, weights


@tf_export("nn.safe_embedding_lookup_sparse", v1=[])
def safe_embedding_lookup_sparse_v2(embedding_weights,
                                    sparse_ids,
                                    sparse_weights=None,
                                    combiner="mean",
                                    default_id=None,
                                    max_norm=None,
                                    name=None):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Note: when doing embedding lookup on `embedding_weights`, "div" partition
  strategy will be used. Support for other partition strategy will be added
  later.

  Args:
    embedding_weights:  A list of `P` float `Tensor`s or values representing
      partitioned embedding `Tensor`s.  Alternatively, a `PartitionedVariable`
      created by partitioning along dimension 0.  The total unpartitioned shape
      should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the vocab size
      and `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
      float weights corresponding to `sparse_ids`, or `None` if all weights are
      be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn", "tile" and "sum" are supported, with
      "mean" the default.
    default_id: The id to use for an entry with no features.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
    name: A name for this operation (optional).

  Returns:
    Dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  return safe_embedding_lookup_sparse(
      embedding_weights,
      sparse_ids,
      sparse_weights=sparse_weights,
      combiner=combiner,
      default_id=default_id,
      name=name,
      partition_strategy="div",
      max_norm=max_norm)


@tf_export(v1=["nn.safe_embedding_lookup_sparse"])
def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids,
                                 sparse_weights=None,
                                 combiner="mean",
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div",
                                 max_norm=None,
                                 prune=True):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights:  A list of `P` float `Tensor`s or values representing
      partitioned embedding `Tensor`s.  Alternatively, a `PartitionedVariable`
      created by partitioning along dimension 0.  The total unpartitioned shape
      should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the vocab size
      and `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
      float weights corresponding to `sparse_ids`, or `None` if all weights are
      be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy. Currently
      `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.

  Returns:
    Dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if embedding_weights is None:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  tmp_embedding_weights = []
  for w in embedding_weights:
    from tensorflow.python.ops.hash_table import hash_table
    if not isinstance(w, (hash_table.DistributedHashTable, hash_table.HashTable)):
      if not (isinstance(w, resource_variable_ops.ResourceVariable) and dtype in (None, w.dtype)):
        w = ops.convert_to_tensor(w, dtype=dtype)
    tmp_embedding_weights.append(w)
  embedding_weights = tmp_embedding_weights

  with ops.name_scope(name, "embedding_lookup", embedding_weights +
                      [sparse_ids, sparse_weights]) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim is None else original_rank_dim)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)
    ])
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(sparse_ids.indices,
                                                  sparse_weights.values,
                                                  sparse_ids.dense_shape)

    if prune:
      # Prune invalid ids and weights.
      sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)
      if combiner != "sum":
        sparse_ids, sparse_weights = _prune_invalid_weights(
            sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

    result = embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]))

      result = array_ops.where(
          is_row_empty, array_ops.zeros_like(result), result, name=scope)

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat([
            array_ops.slice(
                math_ops.cast(original_shape, dtypes.int32), [0],
                [original_rank - 1]),
            array_ops.slice(array_ops.shape(result), [1], [-1])
        ], 0))
    final_result.set_shape(
        tensor_shape.unknown_shape(
            (tensor_shape.Dimension(original_rank_dim) - 1).value).concatenate(
                result.get_shape()[1:]))
    return final_result


@tf_export("nn.safe_embedding_lookup_multi_dim")
def safe_embedding_lookup_multi_dim(embedding_weights,
                                    sparse_ids,
                                    sparse_weights=None,
                                    combiners=['mean'],
                                    name=None,
                                    partition_strategy='div',
                                    max_norm=None,
                                    weight_axis=1,
                                    prune=True):
  if combiners is None:
    combiners = ["mean"]
  if not isinstance(combiners, (list, tuple)):
    combiners = (combiners,)
  for comb in combiners:
    if comb not in ("mean", "sqrtn", "sum", "max", "min", "tile"):
      raise ValueError("combiner must be one of 'mean', 'sqrtn', 'sum', 'max', 'min' or 'tile'")
  combiners = list(combiners)
  real_combiner_size = len(combiners)
  tile_combiner_nums = sum([1 if comb == 'tile' else 0 for comb in combiners])
  if sparse_ids.shape is not None and sparse_ids.shape.rank > len(combiners) + 1:
    tile_num = (sparse_ids.shape.rank - 1 - len(combiners))
    combiners = ['tile'] * tile_num + combiners
  if embedding_weights is None:
    raise ValueError('Missing embedding_weights %s.' % embedding_weights)
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError('Missing embedding_weights %s.' % embedding_weights)
  dtype = sparse_weights.dtype if sparse_weights is not None else None
  embedding_weights = [
      ops.convert_to_tensor(w, dtype=dtype) for w in embedding_weights
  ]
  with ops.name_scope(name, 'embedding_lookup',
                      embedding_weights + [sparse_ids,
                                           sparse_weights]) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    if prune:
      # Prune invalid ids and weights.
      sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)
      if 'sum' not in combiners:
        sparse_ids, sparse_weights = _prune_invalid_weights(
            sparse_ids, sparse_weights)
    result = embedding_lookup_sparse_multi_dim(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiners=combiners,
        partition_strategy=partition_strategy,
        name=None,
        max_norm=max_norm,
        weight_axis=weight_axis)
    batch_size = math_ops.cast(original_shape[0], dtype=dtypes.int32)
    pad_list = [[0, batch_size - array_ops.shape(result)[0]], [0, 0]]
    result = array_ops.pad(result, pad_list)
    output_shape = array_ops.concat([
      array_ops.slice(math_ops.cast(original_shape, dtypes.int32),
                      [0],
                      [array_ops.size(original_shape) - real_combiner_size]),
      [-1]
      ], 0)
    result = array_ops.reshape(result, output_shape)
    return result

@tf_export("nn.safe_adaptive_embedding_lookup_sparse")
def safe_adaptive_embedding_lookup_sparse(hash_embedding_weights,
                                          ev_embedding_weights,
                                          sparse_ids,
                                          hash_ev_ids,
                                          sparse_weights=None,
                                          combiner=None,
                                          default_id=None,
                                          name=None,
                                          partition_strategy="div",
                                          max_norm=None,
                                          bucket_size=None,
                                          adaptive_mask_tensor=None,
                                          blocknums=None):
  """Lookup embedding results, accounting for invalid IDs and empty features.
  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using `tf.get_variable()` with a
  partitioner.
  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.
  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.
  Args:
    hash_embedding_weights: A list of `P` float tensors or values representing
        partitioned embedding tensors by hash-bucket size variable.
        Alternatively, a `PartitionedVariable`,
        created by partitioning along dimension 0.  The total unpartitioned
        shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
        vocab size and `e_1, ..., e_m` are the embedding dimensions.
    embedding_weights:  A list of `P` float tensors or values representing
        partitioned embedding tensors by EmbeddingVariable.
        Alternatively, a `PartitionedVariable`,
        created by partitioning along dimension 0.  The total unpartitioned
        shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
        vocab size and `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights
        are be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy.
        Currently `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not None, all embeddings are l2-normalized to max_norm before
        combining.
  Returns:
    Dense tensor of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.
  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if combiner is None:
    #logging.warn("The default value of combiner will change from \"mean\" "
    #             "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"


  dtype = sparse_weights.dtype if sparse_weights is not None else None
  if isinstance(hash_embedding_weights, variables.PartitionedVariable):
    hash_embedding_weights = list(hash_embedding_weights)
  if not isinstance(hash_embedding_weights, list):
    hash_embedding_weights = [hash_embedding_weights]
  hash_embedding_weights = [
        ops.convert_to_tensor(w, dtype=dtype) for w in hash_embedding_weights
    ]
  check_ops.assert_same_float_dtype(hash_embedding_weights +
                                              [sparse_weights])
  '''
  if not isinstance(embedding_weights[0],
      (kv_variable_ops.EmbeddingVariable, kv_variable_ops.DynamicEmbeddingVariable)):

  '''
  with ops.name_scope(name, "embedding_lookup",
                      hash_embedding_weights + [sparse_ids,
                                           sparse_weights]) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = sparse_ids.dense_shape.get_shape()[0]
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim.value is None
        else original_rank_dim.value)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)])
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(
          sparse_ids.indices,
          sparse_weights.values, sparse_ids.dense_shape)
    # Prune invalid ids and weights.
    sparse_ids, sparse_weights = _prune_invalid_ids(
      sparse_ids, sparse_weights)
    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(sparse_ids,
                                                                 default_id or
                                                                 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)
    result = adaptive_embedding_lookup_sparse(
        hash_embedding_weights,
        ev_embedding_weights,
        sparse_ids,
        hash_ev_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm,
        bucket_size=bucket_size,
        adaptive_mask_tensor=adaptive_mask_tensor,
        blocknums=blocknums)
    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]))
      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name=scope)
    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat([
            array_ops.slice(
                math_ops.cast(original_shape, dtypes.int32), [0],
                [original_rank - 1]),
            array_ops.slice(array_ops.shape(result), [1], [-1])
        ], 0))
    final_result.set_shape(tensor_shape.unknown_shape(
        (original_rank_dim - 1).value).concatenate(result.get_shape()[1:]))
    return final_result

def _prune_invalid_ids(sparse_ids, sparse_weights):
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
  if sparse_weights is not None:
    is_id_valid = math_ops.logical_and(
        is_id_valid,
        array_ops.ones_like(sparse_weights.values, dtype=dtypes.bool))
  sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
  if sparse_weights is not None:
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
  return sparse_ids, sparse_weights


def _prune_invalid_weights(sparse_ids, sparse_weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if sparse_weights is not None:
    is_weights_valid = math_ops.greater(sparse_weights.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
  return sparse_ids, sparse_weights
