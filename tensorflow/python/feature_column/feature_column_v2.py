# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""This API defines FeatureColumn abstraction.

FeatureColumns provide a high level abstraction for ingesting and representing
features. FeatureColumns are also the primary way of encoding features for
canned `tf.estimator.Estimator`s.

When using FeatureColumns with `Estimators`, the type of feature column you
should choose depends on (1) the feature type and (2) the model type.

1. Feature type:

  * Continuous features can be represented by `numeric_column`.
  * Categorical features can be represented by any `categorical_column_with_*`
  column:
    - `categorical_column_with_vocabulary_list`
    - `categorical_column_with_vocabulary_file`
    - `categorical_column_with_hash_bucket`
    - `categorical_column_with_identity`
    - `weighted_categorical_column`

2. Model type:

  * Deep neural network models (`DNNClassifier`, `DNNRegressor`).

    Continuous features can be directly fed into deep neural network models.

      age_column = numeric_column("age")

    To feed sparse features into DNN models, wrap the column with
    `embedding_column` or `indicator_column`. `indicator_column` is recommended
    for features with only a few possible values. For features with many
    possible values, to reduce the size of your model, `embedding_column` is
    recommended.

      embedded_dept_column = embedding_column(
          categorical_column_with_vocabulary_list(
              "department", ["math", "philosophy", ...]), dimension=10)

  * Wide (aka linear) models (`LinearClassifier`, `LinearRegressor`).

    Sparse features can be fed directly into linear models. They behave like an
    indicator column but with an efficient implementation.

      dept_column = categorical_column_with_vocabulary_list("department",
          ["math", "philosophy", "english"])

    It is recommended that continuous features be bucketized before being
    fed into linear models.

      bucketized_age_column = bucketized_column(
          source_column=age_column,
          boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    Sparse features can be crossed (also known as conjuncted or combined) in
    order to form non-linearities, and then fed into linear models.

      cross_dept_age_column = crossed_column(
          columns=["department", bucketized_age_column],
          hash_bucket_size=1000)

Example of building canned `Estimator`s using FeatureColumns:

  ```python
  # Define features and transformations
  deep_feature_columns = [age_column, embedded_dept_column]
  wide_feature_columns = [dept_column, bucketized_age_column,
      cross_dept_age_column]

  # Build deep model
  estimator = DNNClassifier(
      feature_columns=deep_feature_columns,
      hidden_units=[500, 250, 50])
  estimator.train(...)

  # Or build a wide model
  estimator = LinearClassifier(
      feature_columns=wide_feature_columns)
  estimator.train(...)

  # Or build a wide and deep model!
  estimator = DNNLinearCombinedClassifier(
      linear_feature_columns=wide_feature_columns,
      dnn_feature_columns=deep_feature_columns,
      dnn_hidden_units=[500, 250, 50])
  estimator.train(...)
  ```


FeatureColumns can also be transformed into a generic input layer for
custom models using `input_layer`.

Example of building model using FeatureColumns, this can be used in a
`model_fn` which is given to the {tf.estimator.Estimator}:

  ```python
  # Building model via layers

  deep_feature_columns = [age_column, embedded_dept_column]
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=deep_feature_columns)
  first_layer = input_layer(
      features=columns_to_tensor,
      feature_columns=deep_feature_columns)
  second_layer = fully_connected(first_layer, ...)
  ```

NOTE: Functions prefixed with "_" indicate experimental or private parts of
the API subject to change, and should not be relied upon!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import math

import numpy as np
import six
import json

from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.feature_column import coalesced_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
# TODO(b/118385027): Dependency on keras can be problematic if Keras moves out
# of the main repo.
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_feature_column_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.compat import collections_abc


_FEATURE_COLUMN_DEPRECATION_DATE = None
_FEATURE_COLUMN_DEPRECATION = ('The old _FeatureColumn APIs are being '
                               'deprecated. Please use the new FeatureColumn '
                               'APIs instead.')


class StateManager(object):
  """Manages the state associated with FeatureColumns.

  Some `FeatureColumn`s create variables or resources to assist their
  computation. The `StateManager` is responsible for creating and storing these
  objects since `FeatureColumn`s are supposed to be stateless configuration
  only.
  """

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    """Creates a new variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      trainable: Whether this variable is trainable or not.
      use_resource: If true, we use resource variables. Otherwise we use
        RefVariable.
      initializer: initializer instance (callable).

    Returns:
      The created variable.
    """
    del feature_column, name, shape, dtype, trainable, use_resource, initializer
    raise NotImplementedError('StateManager.create_variable')

  def create_hashtable(self,
                       feature_column,
                       name,
                       shape,
                       dtype,
                       initializer,
                       partitioner,
                       trainable):
    """Creates a new variable.
    """
    del feature_column, name, shape, dtype, initializer, partitioner, trainable
    raise NotImplementedError('StateManager.create_hashtable')

  def add_variable(self, feature_column, var):
    """Adds an existing variable to the state.

    Args:
      feature_column: A `FeatureColumn` object to associate this variable with.
      var: The variable.
    """
    del feature_column, var
    raise NotImplementedError('StateManager.add_variable')

  def get_variable(self, feature_column, name):
    """Returns an existing variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
    """
    del feature_column, name
    raise NotImplementedError('StateManager.get_var')

  def add_resource(self, feature_column, name, resource):
    """Creates a new resource.

    Resources can be things such as tables etc.

    Args:
      feature_column: A `FeatureColumn` object this resource corresponds to.
      name: Name of the resource.
      resource: The resource.

    Returns:
      The created resource.
    """
    del feature_column, name, resource
    raise NotImplementedError('StateManager.add_resource')

  def get_resource(self, feature_column, name):
    """Returns an already created resource.

    Resources can be things such as tables etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: Name of the resource.
    """
    del feature_column, name
    raise NotImplementedError('StateManager.get_resource')


class _StateManagerImpl(StateManager):
  """Manages the state of DenseFeatures and LinearLayer."""

  def __init__(self, layer, trainable):
    """Creates an _StateManagerImpl object.

    Args:
      layer: The input layer this state manager is associated with.
      trainable: Whether by default, variables created are trainable or not.
    """
    self._trainable = trainable
    self._layer = layer
    if self._layer is not None:
      self._layer._maybe_create_attribute('_resources', [])  # pylint: disable=protected-access
    self._cols_to_vars_map = collections.defaultdict(lambda: {})
    # TODO(vbardiovsky): Make sure the resources are tracked by moving them to
    # the layer (inheriting from AutoTrackable), e.g.:
    # self._layer._resources_map = data_structures.Mapping()
    self._cols_to_resources_map = collections.defaultdict(lambda: {})

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    if name in self._cols_to_vars_map[feature_column]:
      raise ValueError('Variable already exists.')

    var = self._layer.add_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=self._trainable and trainable,
        use_resource=use_resource,
        # TODO(rohanj): Get rid of this hack once we have a mechanism for
        # specifying a default partitioner for an entire layer. In that case,
        # the default getter for Layers should work.
        getter=variable_scope.get_variable)
    self._cols_to_vars_map[feature_column][name] = var
    return var

  def create_hashtable(self,
                       feature_column,
                       name,
                       shape,
                       dtype,
                       initializer,
                       partitioner,
                       trainable):
    if name in self._cols_to_vars_map[feature_column]:
      raise ValueError('Variable already exists.')

    table = variable_scope.get_hash_table(
      name, shape, dtype=dtype,
      initializer=initializer,
      partitioner=partitioner,
      trainable=self._trainable and trainable)
    self._cols_to_vars_map[feature_column][name] = table
    return table

  def get_variable(self, feature_column, name):
    if name in self._cols_to_vars_map[feature_column]:
      return self._cols_to_vars_map[feature_column][name]
    raise ValueError('Variable does not exist.')

  def add_resource(self, feature_column, name, resource):
    self._cols_to_resources_map[feature_column][name] = resource
    if self._layer is not None:
      self._layer._resources.append(resource)  # pylint: disable=protected-access

  def get_resource(self, feature_column, name):
    if name in self._cols_to_resources_map[feature_column]:
      return self._cols_to_resources_map[feature_column][name]
    raise ValueError('Resource does not exist.')


class _StateManagerImplV2(_StateManagerImpl):
  """Manages the state of DenseFeatures."""

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    if name in self._cols_to_vars_map[feature_column]:
      raise ValueError('Variable already exists.')

    var = self._layer.add_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=self._trainable and trainable,
        use_resource=use_resource)
    self._cols_to_vars_map[feature_column][name] = var
    return var


class _BaseFeaturesLayer(Layer):
  """Base class for DenseFeatures and SequenceFeatures.

  Defines common methods and helpers.

  Args:
    feature_columns: An iterable containing the FeatureColumns to use as
      inputs to your model.
    expected_column_type: Expected class for provided feature columns.
    trainable:  Boolean, whether the layer's variables will be updated via
      gradient descent during training.
    name: Name to give to the DenseFeatures.
    **kwargs: Keyword arguments to construct a layer.

  Raises:
    ValueError: if an item in `feature_columns` doesn't match
      `expected_column_type`.
  """
  def __init__(self, feature_columns, expected_column_type, trainable, name,
               **kwargs):
    super(_BaseFeaturesLayer, self).__init__(
        name=name, trainable=trainable, **kwargs)
    self._feature_columns = _normalize_feature_columns(feature_columns)
    self._state_manager = _StateManagerImpl(self, self.trainable)
    for column in self._feature_columns:
      if not isinstance(column, expected_column_type):
        raise ValueError(
            'Items of feature_columns must be a {}. '
            'You can wrap a categorical column with an '
            'embedding_column or indicator_column. Given: {}'.format(
                expected_column_type, column))

  def build(self, _):
    for column in self._feature_columns:
      with variable_scope._pure_variable_scope(self.name):  # pylint: disable=protected-access
        with variable_scope._pure_variable_scope(column.var_scope_name):  # pylint: disable=protected-access
          column.create_state(self._state_manager)
    super(_BaseFeaturesLayer, self).build(None)

  def _output_shape(self, input_shape, num_elements):
    """Computes expected output shape of the layer or a column's dense tensor.

    Args:
      input_shape: Tensor or array with batch shape.
      num_elements: Size of the last dimension of the output.

    Returns:
      Tuple with output shape.
    """
    raise NotImplementedError('Calling an abstract method.')

  def compute_output_shape(self, input_shape):
    total_elements = 0
    for column in self._feature_columns:
      total_elements += column.variable_shape.num_elements()
    return self._target_shape(input_shape, total_elements)

  def _process_dense_tensor(self, column, tensor):
    """Reshapes the dense tensor output of a column based on expected shape.

    Args:
      column: A DenseColumn or SequenceDenseColumn object.
      tensor: A dense tensor obtained from the same column.

    Returns:
      Reshaped dense tensor."""
    target_shape = column.output_shape(tensor)
    return array_ops.reshape(tensor, shape=target_shape)

  def _verify_and_concat_tensors(self, output_tensors):
    """Verifies and concatenates the dense output of several columns."""
    _verify_static_batch_size_equality(output_tensors, self._feature_columns)
    return array_ops.concat(output_tensors, -1)

  def get_config(self):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    column_configs = serialization.serialize_feature_columns(
        self._feature_columns)
    config = {'feature_columns': column_configs}

    base_config = super(  # pylint: disable=bad-super-call
        _BaseFeaturesLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    config_cp = config.copy()
    config_cp['feature_columns'] = serialization.deserialize_feature_columns(
        config['feature_columns'], custom_objects=custom_objects)

    return cls(**config_cp)


class _LinearModelLayer(Layer):
  """Layer that contains logic for `LinearModel`."""

  def __init__(self,
               feature_columns,
               units=1,
               sparse_combiner='sum',
               trainable=True,
               name=None,
               **kwargs):
    super(_LinearModelLayer, self).__init__(
        name=name, trainable=trainable, **kwargs)

    self._feature_columns = _normalize_feature_columns(feature_columns)
    for column in self._feature_columns:
      if not isinstance(column, (DenseColumn, CategoricalColumn)):
        raise ValueError(
            'Items of feature_columns must be either a '
            'DenseColumn or CategoricalColumn. Given: {}'.format(column))
      if isinstance(column, (SequenceMultiHashEmbeddingColumn, SequenceEmbeddingColumn)):
        raise ValueError(
            'Items of feature_columns must not be a '
            'SequenceEmbeddingColumn or SequenceMultiHashEmbeddingColumn'
            '. Given: {}'.format(column))

    self._units = units
    self._sparse_combiner = sparse_combiner

    self._state_manager = _StateManagerImpl(self, self.trainable)
    self.bias = None

  def build(self, _):
    # We need variable scopes for now because we want the variable partitioning
    # information to percolate down. We also use _pure_variable_scope's here
    # since we want to open up a name_scope in the `call` method while creating
    # the ops.
    with variable_scope._pure_variable_scope(self.name):  # pylint: disable=protected-access
      for column in self._feature_columns:
        with variable_scope._pure_variable_scope(column.name):  # pylint: disable=protected-access
          # Create the state for each feature column
          column.create_state(self._state_manager)

          # Create a weight variable for each column.
          if isinstance(column, CategoricalColumn):
            first_dim = column.num_buckets
          else:
            first_dim = column.variable_shape.num_elements()
          self._state_manager.create_variable(
              column,
              name='weights',
              dtype=dtypes.float32,
              shape=(first_dim, self._units),
              initializer=init_ops.zeros_initializer(),
              trainable=self.trainable)

      # Create a bias variable.
      self.bias = self.add_variable(
          name='bias_weights',
          dtype=dtypes.float32,
          shape=[self._units],
          initializer=init_ops.zeros_initializer(),
          trainable=self.trainable,
          use_resource=True,
          # TODO(rohanj): Get rid of this hack once we have a mechanism for
          # specifying a default partitioner for an entire layer. In that case,
          # the default getter for Layers should work.
          getter=variable_scope.get_variable)

    super(_LinearModelLayer, self).build(None)

  def call(self, features):
    if not isinstance(features, dict):
      raise ValueError('We expected a dictionary here. Instead we got: {}'
                       .format(features))
    with ops.name_scope(self.name):
      transformation_cache = FeatureTransformationCache(features)
      weighted_sums = []
      for column in self._feature_columns:
        with ops.name_scope(column.name):
          # All the weights used in the linear model are owned by the state
          # manager associated with this Linear Model.
          weight_var = self._state_manager.get_variable(column, 'weights')

          weighted_sum = _create_weighted_sum(
              column=column,
              transformation_cache=transformation_cache,
              state_manager=self._state_manager,
              sparse_combiner=self._sparse_combiner,
              weight_var=weight_var)
          weighted_sums.append(weighted_sum)

      _verify_static_batch_size_equality(weighted_sums, self._feature_columns)
      predictions_no_bias = math_ops.add_n(
          weighted_sums, name='weighted_sum_no_bias')
      predictions = nn_ops.bias_add(
          predictions_no_bias, self.bias, name='weighted_sum')
      return predictions

  def get_config(self):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    column_configs = serialization.serialize_feature_columns(
        self._feature_columns)
    config = {
        'feature_columns': column_configs,
        'units': self._units,
        'sparse_combiner': self._sparse_combiner
    }

    base_config = super(  # pylint: disable=bad-super-call
        _LinearModelLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    config_cp = config.copy()
    columns = serialization.deserialize_feature_columns(
        config_cp['feature_columns'], custom_objects=custom_objects)

    del config_cp['feature_columns']
    return cls(feature_columns=columns, **config_cp)


# TODO(tanzheny): Cleanup it with respect to Premade model b/132690565.
class LinearModel(training.Model):
  """Produces a linear prediction `Tensor` based on given `feature_columns`.

  This layer generates a weighted sum based on output dimension `units`.
  Weighted sum refers to logits in classification problems. It refers to the
  prediction itself for linear regression problems.

  Note on supported columns: `LinearLayer` treats categorical columns as
  `indicator_column`s. To be specific, assume the input as `SparseTensor` looks
  like:

  ```python
    shape = [2, 2]
    {
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
    }
  ```
  `linear_model` assigns weights for the presence of "a", "b", "c' implicitly,
  just like `indicator_column`, while `input_layer` explicitly requires wrapping
  each of categorical columns with an `embedding_column` or an
  `indicator_column`.

  Example of usage:

  ```python
  price = numeric_column('price')
  price_buckets = bucketized_column(price, boundaries=[0., 10., 100., 1000.])
  keywords = categorical_column_with_hash_bucket("keywords", 10K)
  keywords_price = crossed_column('keywords', price_buckets, ...)
  columns = [price_buckets, keywords, keywords_price ...]
  linear_model = LinearLayer(columns)

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  prediction = linear_model(features)
  ```
  """

  def __init__(self,
               feature_columns,
               units=1,
               sparse_combiner='sum',
               trainable=True,
               name=None,
               **kwargs):
    """Constructs a LinearLayer.

    Args:
      feature_columns: An iterable containing the FeatureColumns to use as
        inputs to your model. All items should be instances of classes derived
        from `_FeatureColumn`s.
      units: An integer, dimensionality of the output space. Default value is 1.
      sparse_combiner: A string specifying how to reduce if a categorical column
        is multivalent. Except `numeric_column`, almost all columns passed to
        `linear_model` are considered as categorical columns.  It combines each
        categorical column independently. Currently "mean", "sqrtn" and "sum"
        are supported, with "sum" the default for linear model. "sqrtn" often
        achieves good accuracy, in particular with bag-of-words columns.
          * "sum": do not normalize features in the column
          * "mean": do l1 normalization on features in the column
          * "sqrtn": do l2 normalization on features in the column
        For example, for two features represented as the categorical columns:

          ```python
          # Feature 1

          shape = [2, 2]
          {
              [0, 0]: "a"
              [0, 1]: "b"
              [1, 0]: "c"
          }

          # Feature 2

          shape = [2, 3]
          {
              [0, 0]: "d"
              [1, 0]: "e"
              [1, 1]: "f"
              [1, 2]: "g"
          }
          ```

        with `sparse_combiner` as "mean", the linear model outputs conceptly are
        ```
        y_0 = 1.0 / 2.0 * ( w_a + w_ b) + w_c + b_0
        y_1 = w_d + 1.0 / 3.0 * ( w_e + w_ f + w_g) + b_1
        ```
        where `y_i` is the output, `b_i` is the bias, and `w_x` is the weight
        assigned to the presence of `x` in the input features.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: Name to give to the Linear Model. All variables and ops created will
        be scoped by this name.
      **kwargs: Keyword arguments to construct a layer.

    Raises:
      ValueError: if an item in `feature_columns` is neither a `DenseColumn`
        nor `CategoricalColumn`.
    """

    super(LinearModel, self).__init__(name=name, **kwargs)
    self.layer = _LinearModelLayer(
        feature_columns,
        units,
        sparse_combiner,
        trainable,
        name=self.name,
        **kwargs)

  def call(self, features):
    """Returns a `Tensor` the represents the predictions of a linear model.

    Args:
      features: A mapping from key to tensors. `_FeatureColumn`s look up via
        these keys. For example `numeric_column('price')` will look at 'price'
        key in this dict. Values are `Tensor` or `SparseTensor` depending on
        corresponding `_FeatureColumn`.

    Returns:
      A `Tensor` which represents predictions/logits of a linear model. Its
      shape is (batch_size, units) and its dtype is `float32`.

    Raises:
      ValueError: If features are not a dictionary.
    """
    return self.layer(features)

  @property
  def bias(self):
    return self.layer.bias


def _transform_features_v2(features, feature_columns, state_manager):
  """Returns transformed features based on features columns passed in.

  Please note that most probably you would not need to use this function. Please
  check `input_layer` and `linear_model` to see whether they will
  satisfy your use case or not.

  Example:

  ```python
  # Define features and transformations
  crosses_a_x_b = crossed_column(
      columns=["sparse_feature_a", "sparse_feature_b"], hash_bucket_size=10000)
  price_buckets = bucketized_column(
      source_column=numeric_column("price"), boundaries=[...])

  columns = [crosses_a_x_b, price_buckets]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  transformed = transform_features(features=features, feature_columns=columns)

  assertCountEqual(columns, transformed.keys())
  ```

  Args:
    features: A mapping from key to tensors. `FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values can be a `SparseTensor` or a `Tensor` depends on
      corresponding `FeatureColumn`.
    feature_columns: An iterable containing all the `FeatureColumn`s.
    state_manager: A StateManager object that holds the FeatureColumn state.

  Returns:
    A `dict` mapping `FeatureColumn` to `Tensor` and `SparseTensor` values.
  """
  feature_columns = _normalize_feature_columns(feature_columns)
  outputs = {}
  with ops.name_scope(
      None, default_name='transform_features', values=features.values()):
    transformation_cache = FeatureTransformationCache(features)
    for column in feature_columns:
      with ops.name_scope(None, default_name=column.name):
        outputs[column] = transformation_cache.get(column, state_manager)
  return outputs


@tf_export('feature_column.make_parse_example_spec', v1=[])
def make_parse_example_spec_v2(feature_columns):
  """Creates parsing spec dictionary from input feature_columns.

  The returned dictionary can be used as arg 'features' in
  `tf.io.parse_example`.

  Typical usage example:

  ```python
  # Define features and transformations
  feature_a = categorical_column_with_vocabulary_file(...)
  feature_b = numeric_column(...)
  feature_c_bucketized = bucketized_column(numeric_column("feature_c"), ...)
  feature_a_x_feature_c = crossed_column(
      columns=["feature_a", feature_c_bucketized], ...)

  feature_columns = set(
      [feature_b, feature_c_bucketized, feature_a_x_feature_c])
  features = tf.io.parse_example(
      serialized=serialized_examples,
      features=make_parse_example_spec(feature_columns))
  ```

  For the above example, make_parse_example_spec would return the dict:

  ```python
  {
      "feature_a": parsing_ops.VarLenFeature(tf.string),
      "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
      "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
  }
  ```

  Args:
    feature_columns: An iterable containing all feature columns. All items
      should be instances of classes derived from `FeatureColumn`.

  Returns:
    A dict mapping each feature key to a `FixedLenFeature` or `VarLenFeature`
    value.

  Raises:
    ValueError: If any of the given `feature_columns` is not a `FeatureColumn`
      instance.
  """
  result = {}
  for column in feature_columns:
    if not isinstance(column, FeatureColumn):
      raise ValueError('All feature_columns must be FeatureColumn instances. '
                       'Given: {}'.format(column))
    config = column.parse_example_spec
    for key, value in six.iteritems(config):
      if key in result and value != result[key]:
        raise ValueError(
            'feature_columns contain different parse_spec for key '
            '{}. Given {} and {}'.format(key, value, result[key]))
    result.update(config)
  return result


@tf_export('feature_column.embedding_column')
def embedding_column(categorical_column,
                     dimension,
                     combiner='mean',
                     initializer=None,
                     ckpt_to_load_from=None,
                     tensor_name_in_ckpt=None,
                     max_norm=None,
                     trainable=True,
                     coalesced_scope=None):
  """`DenseColumn` that converts from sparse, categorical input.

  Use this when your inputs are sparse, but you want to convert them to a dense
  representation (e.g., to feed to a DNN).

  Inputs must be a `CategoricalColumn` created by any of the
  `categorical_column_*` function. Here is an example of using
  `embedding_column` with `DNNClassifier`:

  ```python
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [embedding_column(video_id, 9),...]

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `embedding_column` with model_fn:

  ```python
  def model_fn(features, ...):
    video_id = categorical_column_with_identity(
        key='video_id', num_buckets=1000000, default_value=0)
    columns = [embedding_column(video_id, 9),...]
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_column: A `CategoricalColumn` created by a
      `categorical_column_with_*` function. This column produces the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    `DenseColumn` that converts from sparse input.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: If eager execution is enabled.
  """
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))
  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  column = EmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable,
      coalesced_scope=coalesced_scope)
  if coalesced_scope:
    coalesced_scope.add_column(column)
    coalesced_utils.add_embedding_signature(
        column, dimension, combiner, initializer, trainable,
        categorical_column._num_buckets)
  return column


@tf_export('feature_column.sequence_embedding_column')
def sequence_embedding_column(dense_column, sequence_length):
  """`DenseColumn` that converts from dense input.

  Args:
    dense_column: A `DenseColumn`. This column produces the dense tensor
      that are inputs to reshape.
    sequence_length: The sequence length of sample.

  Returns:
    `DenseColumn` that converts from dense input.

  Raises:
    ValueError: if `sequence_length` not > 0.
  """
  if sequence_length is None:
    raise ValueError('sequence_length must be set.')
  if sequence_length < 1:
    raise ValueError('sequence_length must be at least 1.')
  return SequenceEmbeddingColumn(dense_column, sequence_length)


@tf_export('feature_column.sequence_multi_hash_embedding_column')
def sequence_multi_hash_embedding_column(dense_column, sequence_length):
  """`DenseColumn` that converts from dense input.

  Args:
    dense_column: A `SharedMultiHashEmbeddingColumn`. This column
      produces the dense tensor that are inputs to reshape.
    sequence_length: The sequence length of sample.

  Returns:
    `DenseColumn` that converts from dense input.

  Raises:
    ValueError: if `sequence_length` not > 0.
  """
  if sequence_length is None:
    raise ValueError('sequence_length must be set.')
  if sequence_length < 1:
    raise ValueError('sequence_length must be at least 1.')
  if not isinstance(dense_column, SharedMultiHashEmbeddingColumn):
    raise ValueError('input column must be multi_hash_embedding_column')
  return SequenceMultiHashEmbeddingColumn(dense_column, sequence_length)


@tf_export(v1=['feature_column.shared_embedding_columns'])
def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None,
                             ckpt_to_load_from=None,
                             tensor_name_in_ckpt=None,
                             max_norm=None,
                             trainable=True):
  """List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Here is an example embedding of two features for a DNNClassifier model:

  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `shared_embedding_columns` with model_fn:

  ```python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)

  c0 = sorted_columns[0]
  num_buckets = c0._num_buckets  # pylint: disable=protected-access
  if not isinstance(c0, fc_old._CategoricalColumn):  # pylint: disable=protected-access
    raise ValueError(
        'All categorical_columns must be subclasses of _CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  if isinstance(c0,
                (fc_old._WeightedCategoricalColumn, WeightedCategoricalColumn)):  # pylint: disable=protected-access
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    if isinstance(
        c, (fc_old._WeightedCategoricalColumn, WeightedCategoricalColumn)):  # pylint: disable=protected-access
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column of the same type. '
          'Given column: {} of type: {} does not match given column: {} of '
          'type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c._num_buckets:  # pylint: disable=protected-access
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c._num_buckets))  # pylint: disable=protected-access

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  result = []
  for column in categorical_columns:
    result.append(
        fc_old._SharedEmbeddingColumn(  # pylint: disable=protected-access
            categorical_column=column,
            initializer=initializer,
            dimension=dimension,
            combiner=combiner,
            shared_embedding_collection_name=shared_embedding_collection_name,
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt,
            max_norm=max_norm,
            trainable=trainable))

  return result

@tf_export('feature_column.shared_embedding_column', v1=[])
def shared_embedding_column(categorical_column,
                            dimension,
                            shared_name,
                            combiner='mean',
                            initializer=None,
                            ckpt_to_load_from=None,
                            tensor_name_in_ckpt=None,
                            max_norm=None,
                            trainable=True,
                            coalesced_scope=None):
  """Dense column that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a 
  embedding column that can share the same embedding weights.

  Here is an example embedding of two features for a DNNClassifier model:

  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  wvi_column = shared_embedding_column(
      watched_video_id, 10, 'id')
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  ivi_column = shared_embedding_column(
      impression_video_id, 10, 'id')
  columns = [wvi_column, ivi_column]

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Args:
    categorical_column: a categorical column created by a
      `categorical_column_with_*` function. This column produce the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    shared_embedding_collection_name: Shared collective name of column.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard
      deviation `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A dense column that converts from sparse input.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if the given `categorical_column` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_column are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))

  if not isinstance(categorical_column, CategoricalColumn):
    raise ValueError(
        'Input categorical_column must be subclasses of CategoricalColumn. '
        'Given: {}, of type: {}'.format(categorical_column, type(categorical_column)))
  num_buckets = categorical_column.num_buckets
  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  column = SharedEmbeddingColumnV2(
      categorical_column=categorical_column,
      dimension=dimension,
      shared_name=shared_name,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable,
      coalesced_scope=coalesced_scope)
  if coalesced_scope:
    coalesced_scope.add_column(column)
    coalesced_utils.add_embedding_signature(
        column, dimension, combiner, initializer, trainable,
        categorical_column.num_buckets)
  return column


@tf_export('feature_column.multi_hash_embedding_column', v1=[])
def multi_hash_embedding_column(categorical_column,
                            dimension,
                            shared_name,
                            hash_combiner='mean',
                            combiner='mean',
                            initializer=None,
                            ckpt_to_load_from=None,
                            tensor_name_in_ckpt=None,
                            max_norm=None,
                            trainable=True,
                            coalesced_scope=None):
  """Dense column that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a 
  embedding column that can share the same embedding weights.

  Args:
    categorical_column: a categorical column created by a
      `categorical_column_with_*` function. This column produce the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    shared_embedding_collection_name: Shared collective name of column.
    hash_combiner: A string specifying how to reduce if there are multiple
      hash function in a single row.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard
      deviation `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A dense column that converts from sparse input.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('multi_hash_embedding_column are not supported when eager '
                       'execution is enabled.')
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')
  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))
  input_column = categorical_column
  while isinstance(input_column, CutoffCategoricalColumn):
    input_column = input_column.categorical_column
  if not isinstance(input_column, (MultiHashedCategoricalColumn, WeightedMultiHashedCategoricalColumn)):
    raise ValueError(
        'In multi_hash_embedding_column, '
        'categorical_column must be one of type '
        'categorical_column_with_multi_hash_bucket or '
        'weighted_categorical_column_with_multi_hash_bucket. '
        'Given (type {}): {}'.format(type(input_column),
                                     input_column))
  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  num_buckets = categorical_column.num_buckets
  column = SharedMultiHashEmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      shared_name=shared_name,
      hash_combiner=hash_combiner,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable,
      coalesced_scope=coalesced_scope)
  if coalesced_scope:
    coalesced_scope.add_column(column)
    coalesced_utils.add_embedding_signature(
        column, dimension, combiner, initializer, trainable,
        categorical_column._num_buckets, hash_combiner=hash_combiner)
  return column


@tf_export('feature_column.shared_embeddings', v1=[])
def shared_embedding_columns_v2(categorical_columns,
                                dimension,
                                combiner='mean',
                                initializer=None,
                                shared_embedding_collection_name=None,
                                ckpt_to_load_from=None,
                                tensor_name_in_ckpt=None,
                                max_norm=None,
                                trainable=True,
                                coalesced_scope=None):
  """List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Here is an example embedding of two features for a DNNClassifier model:

  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `shared_embedding_columns` with model_fn:

  ```python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard
      deviation `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional collective name of these columns.
      If not given, a reasonable name will be chosen based on the names of
      `categorical_columns`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)

  c0 = sorted_columns[0]
  num_buckets = c0.num_buckets
  if not isinstance(c0, CategoricalColumn):
    raise ValueError(
        'All categorical_columns must be subclasses of CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  if isinstance(c0, WeightedCategoricalColumn):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    if isinstance(c, WeightedCategoricalColumn):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column of the same type. '
          'Given column: {} of type: {} does not match given column: {} of '
          'type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c.num_buckets:
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c.num_buckets))

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  column_creator = SharedEmbeddingColumnCreator(
      dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
      num_buckets, trainable, shared_embedding_collection_name)

  result = []
  for column in categorical_columns:
    result_column = column_creator(categorical_column=column,
                                   combiner=combiner,
                                   max_norm=max_norm,
                                   coalesced_scope=coalesced_scope)
    if coalesced_scope:
      coalesced_scope.add_column(result_column)
      coalesced_utils.add_embedding_signature(
          result_column, dimension, combiner, initializer, trainable,
          column.num_buckets)
    result.append(result_column)
  return result


@tf_export('feature_column.numeric_column')
def numeric_column(key,
                   shape=(1,),
                   default_value=None,
                   dtype=dtypes.float32,
                   normalizer_fn=None):
  """Represents real valued or numerical features.

  Example:

  ```python
  price = numeric_column('price')
  columns = [price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  # or
  bucketized_price = bucketized_column(price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    shape: An iterable of integers specifies the shape of the `Tensor`. An
      integer can be given which means a single dimension `Tensor` with given
      width. The `Tensor` representing the column will have the shape of
      [batch_size] + `shape`.
    default_value: A single value compatible with `dtype` or an iterable of
      values compatible with `dtype` which the column takes on during
      `tf.Example` parsing if data is missing. A default value of `None` will
      cause `tf.io.parse_example` to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every item. If an iterable of values is provided,
      the shape of the `default_value` should be equal to the given `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.

  Returns:
    A `NumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int
    ValueError: if any dimension in shape is not a positive integer
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  shape = _check_shape(shape, key)
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype must be convertible to float. '
                     'dtype: {}, key: {}'.format(dtype, key))
  default_value = fc_utils.check_default_value(
      shape, default_value, dtype, key)

  if normalizer_fn is not None and not callable(normalizer_fn):
    raise TypeError(
        'normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))

  fc_utils.assert_key_is_string(key)
  return NumericColumn(
      key,
      shape=shape,
      default_value=default_value,
      dtype=dtype,
      normalizer_fn=normalizer_fn)


@tf_export('feature_column.sparse_numeric_column')
def sparse_numeric_column(key,
                          shape=None,
                          dtype=dtypes.int64):
  """Represents sparse format real valued or numerical features.

  Example:

  ```python
  price = sparse_numeric_column('price')
  columns = [price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    shape: An iterable of integers specifies the shape of the `Tensor`. An
      integer can be given which means a single dimension `Tensor` with given
      width. The `Tensor` representing the column will have the shape of
      [batch_size] + `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
  Returns:
    A `SparseNumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int
    ValueError: if any dimension in shape is not a positive integer
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  if shape is not None:
    shape = _check_shape(shape, key)
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype must be convertible to float. '
                     'dtype: {}, key: {}'.format(dtype, key))
  fc_utils.assert_key_is_string(key)
  return SparseNumericColumn(
      key,
      shape=shape,
      dtype=dtype)


@tf_export('feature_column.bucketized_column')
def bucketized_column(source_column, boundaries):
  """Represents discretized dense input.

  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.

  For example, if the inputs are

  ```python
  boundaries = [0, 10, 100]
  input tensor = [[-5, 10000]
                  [150,   10]
                  [5,    100]]
  ```

  then the output will be

  ```python
  output = [[0, 3]
            [3, 2]
            [1, 3]]
  ```

  Example:

  ```python
  price = numeric_column('price')
  bucketized_price = bucketized_column(price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  # or
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  `bucketized_column` can also be crossed with another categorical column using
  `crossed_column`:

  ```python
  price = numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = bucketized_column(price, boundaries=[...])
  # 'keywords' is a string feature.
  price_x_keywords = crossed_column([bucketized_price, 'keywords'], 50K)
  columns = [price_x_keywords, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.

  Returns:
    A `BucketizedColumn`.

  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  """
  if not isinstance(source_column, (NumericColumn, fc_old._NumericColumn)):  # pylint: disable=protected-access
    raise ValueError(
        'source_column must be a column generated with numeric_column(). '
        'Given: {}'.format(source_column))
  if len(source_column.shape) > 1:
    raise ValueError(
        'source_column must be one-dimensional column. '
        'Given: {}'.format(source_column))
  if not boundaries:
    raise ValueError('boundaries must not be empty.')
  if not (isinstance(boundaries, list) or isinstance(boundaries, tuple)):
    raise ValueError('boundaries must be a sorted list.')
  for i in range(len(boundaries) - 1):
    if boundaries[i] >= boundaries[i + 1]:
      raise ValueError('boundaries must be a sorted list.')
  return BucketizedColumn(source_column, tuple(boundaries))


@tf_export('feature_column.sparse_bucketized_column')
def sparse_bucketized_column(source_column, boundaries):
  """Represents discretized dense input.

  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.

  Example:

  ```python
  price = sparse_numeric_column('price')
  bucketized_price = sparse_bucketized_column(price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  ```
  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.

  Returns:
    A `SparseBucketizedColumn`.

  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  """
  if not isinstance(source_column, (NumericColumn, fc_old._NumericColumn, SparseNumericColumn)):  # pylint: disable=protected-access
    raise ValueError(
        'source_column must be a column generated with numeric_column() or sparse_numeric_column. '
        'Given: {}'.format(source_column))
  if source_column.shape is not None and len(source_column.shape) > 1:
    raise ValueError(
        'source_column must be one-dimensional column. '
        'Given: {}'.format(source_column))
  if not boundaries:
    raise ValueError('boundaries must not be empty.')
  if not (isinstance(boundaries, list) or isinstance(boundaries, tuple)):
    raise ValueError('boundaries must be a sorted list.')
  for i in range(len(boundaries) - 1):
    if boundaries[i] >= boundaries[i + 1]:
      raise ValueError('boundaries must be a sorted list.')
  return SparseBucketizedColumn(source_column, tuple(boundaries))


@tf_export('feature_column.cutoff_categorical_column')
def cutoff_categorical_column(categorical_column,
                              cutoff_length,
                              cutoff_side='right',
                              cutoff_axis=1,
                              reverse=False):
  """Cutoff cateforical column sparse id.
  Example:

  ```python
  price = sparse_numeric_column('price')
  bucketized_price = sparse_bucketized_column(price, boundaries=[...])
  cutoff_price = cutoff_categorical_column(bucketized_price, 5)
  columns = [cutoff_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  ```
  Args:
    categorical_column: A categorical column.
    cutoff_length: max size of categorical column input sparse tensor after cutoff
    cutoff_side: side to cut
    cutoff_axis: axis to cut
    reverse: if do reverse ids after cut

  Returns:
    A `CutoffCategoricalColumn`.

  Raises:
    ValueError: If `cutoff_length` is not a valid int.
    ValueError: If `cutoff_axis` is smaller than 1.
    ValueError: If `cutoff_side` is not one of `right/left`.
  """
  if cutoff_length is None:
    raise ValueError('cutoff_length must be set.')
  if cutoff_length < 1:
    raise ValueError('cutoff_length must be at least 1.')
  if cutoff_axis < 1:
    raise ValueError('cutoff_axis must be at least 1.')
  if cutoff_side not in ('right', 'left'):
    raise ValueError('cutoff_side must be one of `right` or `left`.')
  return CutoffCategoricalColumn(categorical_column,
                                 cutoff_length,
                                 cutoff_side,
                                 cutoff_axis,
                                 reverse)


@tf_export('feature_column.categorical_column_with_hash')
def categorical_column_with_hash(key,
                                 hash_type='farm',
                                 allow_neg=True,
                                 dtype=dtypes.string):
  """Represents sparse feature where ids are set by hashing.

  Use this when your sparse features are in string or integer format, and you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) for string type input.
  For int type input, the value is converted to its string representation first
  and then hashed by the same formula.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example:

  ```python
  keywords = categorical_column_with_hash("keywords", 10K)
  columns = [keywords, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    hash_type: Hash function
    allow_neg: Allow hash result a negative int
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `HashedOnlyCategoricalColumn`.

  Raises:
    ValueError: `hash_type` is not set.
  """
  if hash_type is None:
    raise ValueError('hash_type must be set. ' 'key: {}'.format(key))

  fc_utils.assert_key_is_string(key)
  fc_utils.assert_key_is_string(hash_type)
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))

  return HashOnlyCategoricalColumn(key, hash_type, allow_neg, dtype)


@tf_export('feature_column.categorical_column_with_multi_hash_bucket')
def categorical_column_with_multi_hash_bucket(key,
                                              hash_bucket_size,
                                              hash_types,
                                              dtype=dtypes.string):
  """Represents sparse feature where ids are set by hashing.

  Use this when your sparse features are in string or integer format, and you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) for string type input.
  For int type input, the value is converted to its string representation first
  and then hashed by the same formula.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example:

  ```python
  keywords = categorical_column_with_hash("keywords", 10K)
  columns = [keywords, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    hash_type: Hash function
    allow_neg: Allow hash result a negative int
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `HashedOnlyCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not valid.
    ValueError: `hash_types` is not set.
  """
  if hash_bucket_size is None:
    raise ValueError('hash_bucket_size must be set. ' 'key: {}'.format(key))

  if hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be at least 1. '
                     'hash_bucket_size: {}, key: {}'.format(
                         hash_bucket_size, key))

  if hash_types is None:
    raise ValueError('hash_types must be set. ' 'key: {}'.format(key))
  hash_types = tuple(hash_types)

  fc_utils.assert_key_is_string(key)
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))

  return MultiHashedCategoricalColumn(key, hash_bucket_size, hash_types, dtype)


@tf_export('feature_column.weighted_categorical_column_with_multi_hash_bucket')
def weighted_categorical_column_with_multi_hash_bucket(
    categorical_column, weight_feature_key, dtype=dtypes.float32):
  """Applies weight values to a `MultiHashCategoricalColumn`.

  Use this when each of your sparse inputs has both an ID and a value.

  Example:

  ```python
  categorical_column = categorical_column_with_multi_hash_bucket(
      column_name='terms', hash_bucket_size=1000, hash_types=['murmur', 'farm'])
  weighted_column = weighted_categorical_multi_hash_column(
      categorical_column=categorical_column, weight_feature_key='frequencies')
  columns = [weighted_column, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```
  Args:
    categorical_column: A `CategoricalColumn` created by
      `categorical_column_with_*` functions.
    weight_feature_key: String key for weight values.
    dtype: Type of weights, such as `tf.float32`. Only float and integer weights
      are supported.

  Returns:
    A `CategoricalColumn` composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if `dtype` is not convertible to float.
    ValueError: if `categorical_column` is not a MultiHashedCategoricalColumn.
  """
  if (dtype is None) or not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype {} is not convertible to float.'.format(dtype))
  if not isinstance(categorical_column, MultiHashedCategoricalColumn):
    raise ValueError(
        'In weighted_categorical_column_with_multi_hash_bucket, '
        'categorical_column must be type '
        'categorical_column_with_multi_hash_bucket. '
        'Given (type {}): {}'.format(type(categorical_column),
                                     categorical_column))
  return WeightedMultiHashedCategoricalColumn(
      categorical_column=categorical_column,
      weight_feature_key=weight_feature_key,
      dtype=dtype)


@tf_export('feature_column.categorical_column_with_embedding')
def categorical_column_with_embedding(key,
                                      dtype=dtypes.string,
                                      partition_num=None,
                                      ev_option=variables.EmbeddingVariableOption()
                                      ):
  return EmbeddingCategoricalColumn(key, dtype, partition_num, ev_option)

@tf_export('feature_column.categorical_column_with_adaptive_embedding')
def categorical_column_with_adaptive_embedding(key,
                                               hash_bucket_size,
                                               dtype=dtypes.string,
                                               partition_num=None,
                                               ev_option=variables.EmbeddingVariableOption()
                                               ):
  return AdaptiveEmbeddingCategoricalColumn(key,
                                            hash_bucket_size,
                                            dtype,
                                            partition_num,
                                            ev_option)

@tf_export('feature_column.categorical_column_with_hash_bucket')
def categorical_column_with_hash_bucket(key,
                                        hash_bucket_size,
                                        dtype=dtypes.string):
  """Represents sparse feature where ids are set by hashing.

  Use this when your sparse features are in string or integer format, and you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) % bucket_size for string type input.
  For int type input, the value is converted to its string representation first
  and then hashed by the same formula.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example:

  ```python
  keywords = categorical_column_with_hash_bucket("keywords", 10K)
  columns = [keywords, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  # or
  keywords_embedded = embedding_column(keywords, 16)
  columns = [keywords_embedded, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `HashedCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  """
  if hash_bucket_size is None:
    raise ValueError('hash_bucket_size must be set. ' 'key: {}'.format(key))

  if hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be at least 1. '
                     'hash_bucket_size: {}, key: {}'.format(
                         hash_bucket_size, key))

  fc_utils.assert_key_is_string(key)
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))

  return HashedCategoricalColumn(key, hash_bucket_size, dtype)


@tf_export(v1=['feature_column.categorical_column_with_vocabulary_file'])
def categorical_column_with_vocabulary_file(key,
                                            vocabulary_file,
                                            vocabulary_size=None,
                                            num_oov_buckets=0,
                                            default_value=None,
                                            dtype=dtypes.string):
  """A `CategoricalColumn` with a vocabulary file.

  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Example with `default_value`:
  File '/us/states.txt' contains 51 lines - the first line is 'XX', and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal 'XX'
  in input, and other values missing from the file, will be assigned ID 0. All
  others are assigned the corresponding line number 1-50.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=51,
      default_value=0)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(states, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `CategoricalColumn` with a vocabulary file.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  return categorical_column_with_vocabulary_file_v2(
      key, vocabulary_file, vocabulary_size,
      dtype, default_value,
      num_oov_buckets)


@tf_export('feature_column.categorical_column_with_vocabulary_file', v1=[])
def categorical_column_with_vocabulary_file_v2(key,
                                               vocabulary_file,
                                               vocabulary_size=None,
                                               dtype=dtypes.string,
                                               default_value=None,
                                               num_oov_buckets=0):
  """A `CategoricalColumn` with a vocabulary file.

  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  File `'/us/states.txt'` contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Example with `default_value`:
  File `'/us/states.txt'` contains 51 lines - the first line is `'XX'`, and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal
  `'XX'` in input, and other values missing from the file, will be assigned
  ID 0. All others are assigned the corresponding line number 1-50.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=51,
      default_value=0)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(states, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    dtype: The type of features. Only string and integer types are supported.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.

  Returns:
    A `CategoricalColumn` with a vocabulary file.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  if not vocabulary_file:
    raise ValueError('Missing vocabulary_file in {}.'.format(key))

  if vocabulary_size is None:
    if not gfile.Exists(vocabulary_file):
      raise ValueError('vocabulary_file in {} does not exist.'.format(key))

    with gfile.GFile(vocabulary_file) as f:
      vocabulary_size = sum(1 for _ in f)
    logging.info(
        'vocabulary_size = %d in %s is inferred from the number of elements '
        'in the vocabulary_file %s.', vocabulary_size, key, vocabulary_file)

  # `vocabulary_size` isn't required for lookup, but it is for `_num_buckets`.
  if vocabulary_size < 1:
    raise ValueError('Invalid vocabulary_size in {}.'.format(key))
  if num_oov_buckets:
    if default_value is not None:
      raise ValueError(
          'Can\'t specify both num_oov_buckets and default_value in {}.'.format(
              key))
    if num_oov_buckets < 0:
      raise ValueError('Invalid num_oov_buckets {} in {}.'.format(
          num_oov_buckets, key))
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))
  fc_utils.assert_key_is_string(key)
  return VocabularyFileCategoricalColumn(
      key=key,
      vocabulary_file=vocabulary_file,
      vocabulary_size=vocabulary_size,
      num_oov_buckets=0 if num_oov_buckets is None else num_oov_buckets,
      default_value=-1 if default_value is None else default_value,
      dtype=dtype)


@tf_export('feature_column.categorical_column_with_vocabulary_list')
def categorical_column_with_vocabulary_list(key,
                                            vocabulary_list,
                                            dtype=None,
                                            default_value=-1,
                                            num_oov_buckets=0):
  """A `CategoricalColumn` with in-memory vocabulary.

  Use this when your inputs are in string or integer format, and you have an
  in-memory vocabulary mapping each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  In the following example, each input in `vocabulary_list` is assigned an ID
  0-3 corresponding to its index (e.g., input 'B' produces output 2). All other
  inputs are hashed and assigned an ID 4-5.

  ```python
  colors = categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  columns = [colors, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  Example with `default_value`:
  In the following example, each input in `vocabulary_list` is assigned an ID
  0-4 corresponding to its index (e.g., input 'B' produces output 3). All other
  inputs are assigned `default_value` 0.


  ```python
  colors = categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('X', 'R', 'G', 'B', 'Y'), default_value=0)
  columns = [colors, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(colors, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    vocabulary_list: An ordered iterable defining the vocabulary. Each feature
      is mapped to the index of its value (if present) in `vocabulary_list`.
      Must be castable to `dtype`.
    dtype: The type of features. Only string and integer types are supported. If
      `None`, it will be inferred from `vocabulary_list`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
      hash of the input value. A positive `num_oov_buckets` can not be specified
      with `default_value`.

  Returns:
    A `CategoricalColumn` with in-memory vocabulary.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  """
  if (vocabulary_list is None) or (len(vocabulary_list) < 1):
    raise ValueError(
        'vocabulary_list {} must be non-empty, column_name: {}'.format(
            vocabulary_list, key))
  if len(set(vocabulary_list)) != len(vocabulary_list):
    raise ValueError(
        'Duplicate keys in vocabulary_list {}, column_name: {}'.format(
            vocabulary_list, key))
  vocabulary_dtype = dtypes.as_dtype(np.array(vocabulary_list).dtype)
  if num_oov_buckets:
    if default_value != -1:
      raise ValueError(
          'Can\'t specify both num_oov_buckets and default_value in {}.'.format(
              key))
    if num_oov_buckets < 0:
      raise ValueError('Invalid num_oov_buckets {} in {}.'.format(
          num_oov_buckets, key))
  fc_utils.assert_string_or_int(
      vocabulary_dtype, prefix='column_name: {} vocabulary'.format(key))
  if dtype is None:
    dtype = vocabulary_dtype
  elif dtype.is_integer != vocabulary_dtype.is_integer:
    raise ValueError(
        'dtype {} and vocabulary dtype {} do not match, column_name: {}'.format(
            dtype, vocabulary_dtype, key))
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))
  fc_utils.assert_key_is_string(key)

  return VocabularyListCategoricalColumn(
      key=key,
      vocabulary_list=tuple(vocabulary_list),
      dtype=dtype,
      default_value=default_value,
      num_oov_buckets=num_oov_buckets)


@tf_export('feature_column.categorical_column_with_identity')
def categorical_column_with_identity(key, num_buckets, default_value=None):
  """A `CategoricalColumn` that returns identity values.

  Use this when your inputs are integers in the range `[0, num_buckets)`, and
  you want to use the input value itself as the categorical ID. Values outside
  this range will result in `default_value` if specified, otherwise it will
  fail.

  Typically, this is used for contiguous ranges of integer indexes, but
  it doesn't have to be. This might be inefficient, however, if many of IDs
  are unused. Consider `categorical_column_with_hash_bucket` in that case.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  In the following examples, each input in the range `[0, 1000000)` is assigned
  the same value. All other inputs are assigned `default_value` 0. Note that a
  literal 0 in inputs will result in the same default ID.

  Linear model:

  ```python
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [video_id, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  Embedding for a DNN model:

  ```python
  columns = [embedding_column(video_id, 9),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    num_buckets: Range of inputs and outputs is `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range
      `[0, num_buckets)`, and will replace inputs in that range.

  Returns:
    A `CategoricalColumn` that returns identity values.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  """
  if num_buckets < 1:
    raise ValueError(
        'num_buckets {} < 1, column_name {}'.format(num_buckets, key))
  if (default_value is not None) and (
      (default_value < 0) or (default_value >= num_buckets)):
    raise ValueError(
        'default_value {} not in range [0, {}), column_name {}'.format(
            default_value, num_buckets, key))
  fc_utils.assert_key_is_string(key)
  return IdentityCategoricalColumn(
      key=key, number_buckets=num_buckets, default_value=default_value)


@tf_export('feature_column.indicator_column')
def indicator_column(categorical_column):
  """Represents multi-hot representation of given categorical column.

  - For DNN model, `indicator_column` can be used to wrap any
    `categorical_column_*` (e.g., to feed to DNN). Consider to Use
    `embedding_column` if the number of buckets/unique(values) are large.

  - For Wide (aka linear) model, `indicator_column` is the internal
    representation for categorical column when passing categorical column
    directly (as any element in feature_columns) to `linear_model`. See
    `linear_model` for details.

  ```python
  name = indicator_column(categorical_column_with_vocabulary_list(
      'name', ['bob', 'george', 'wanda'])
  columns = [name, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
  dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
  dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]
  ```

  Args:
    categorical_column: A `CategoricalColumn` which is created by
      `categorical_column_with_*` or `crossed_column` functions.

  Returns:
    An `IndicatorColumn`.
  """
  return IndicatorColumn(categorical_column)


@tf_export('feature_column.weighted_categorical_column')
def weighted_categorical_column(categorical_column,
                                weight_feature_key,
                                dtype=dtypes.float32):
  """Applies weight values to a `CategoricalColumn`.

  Use this when each of your sparse inputs has both an ID and a value. For
  example, if you're representing text documents as a collection of word
  frequencies, you can provide 2 parallel sparse input features ('terms' and
  'frequencies' below).

  Example:

  Input `tf.Example` objects:

  ```proto
  [
    features {
      feature {
        key: "terms"
        value {bytes_list {value: "very" value: "model"}}
      }
      feature {
        key: "frequencies"
        value {float_list {value: 0.3 value: 0.1}}
      }
    },
    features {
      feature {
        key: "terms"
        value {bytes_list {value: "when" value: "course" value: "human"}}
      }
      feature {
        key: "frequencies"
        value {float_list {value: 0.4 value: 0.1 value: 0.2}}
      }
    }
  ]
  ```

  ```python
  categorical_column = categorical_column_with_hash_bucket(
      column_name='terms', hash_bucket_size=1000)
  weighted_column = weighted_categorical_column(
      categorical_column=categorical_column, weight_feature_key='frequencies')
  columns = [weighted_column, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  This assumes the input dictionary contains a `SparseTensor` for key
  'terms', and a `SparseTensor` for key 'frequencies'. These 2 tensors must have
  the same indices and dense shape.

  Args:
    categorical_column: A `CategoricalColumn` created by
      `categorical_column_with_*` functions.
    weight_feature_key: String key for weight values.
    dtype: Type of weights, such as `tf.float32`. Only float and integer weights
      are supported.

  Returns:
    A `CategoricalColumn` composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if `dtype` is not convertible to float.
  """
  if (dtype is None) or not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype {} is not convertible to float.'.format(dtype))
  return WeightedCategoricalColumn(
      categorical_column=categorical_column,
      weight_feature_key=weight_feature_key,
      dtype=dtype)


@tf_export('feature_column.crossed_column')
def crossed_column(keys, hash_bucket_size, hash_key=None):
  """Returns a column for performing crosses of categorical features.

  Crossed features will be hashed according to `hash_bucket_size`. Conceptually,
  the transformation can be thought of as:
    Hash(cartesian product of features) % `hash_bucket_size`

  For example, if the input features are:

  * SparseTensor referred by first key:

    ```python
    shape = [2, 2]
    {
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
    }
    ```

  * SparseTensor referred by second key:

    ```python
    shape = [2, 1]
    {
        [0, 0]: "d"
        [1, 0]: "e"
    }
    ```

  then crossed feature will look like:

  ```python
   shape = [2, 2]
  {
      [0, 0]: Hash64("d", Hash64("a")) % hash_bucket_size
      [1, 0]: Hash64("e", Hash64("b")) % hash_bucket_size
      [1, 1]: Hash64("e", Hash64("c")) % hash_bucket_size
  }
  ```

  Here is an example to create a linear model with crosses of string features:

  ```python
  keywords_x_doc_terms = crossed_column(['keywords', 'doc_terms'], 50K)
  columns = [keywords_x_doc_terms, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  You could also use vocabulary lookup before crossing:

  ```python
  keywords = categorical_column_with_vocabulary_file(
      'keywords', '/path/to/vocabulary/file', vocabulary_size=1K)
  keywords_x_doc_terms = crossed_column([keywords, 'doc_terms'], 50K)
  columns = [keywords_x_doc_terms, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  If an input feature is of numeric type, you can use
  `categorical_column_with_identity`, or `bucketized_column`, as in the example:

  ```python
  # vertical_id is an integer categorical feature.
  vertical_id = categorical_column_with_identity('vertical_id', 10K)
  price = numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = bucketized_column(price, boundaries=[...])
  vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
  columns = [vertical_id_x_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  To use crossed column in DNN model, you need to add it in an embedding column
  as in this example:

  ```python
  vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
  vertical_id_x_price_embedded = embedding_column(vertical_id_x_price, 10)
  dense_tensor = input_layer(features, [vertical_id_x_price_embedded, ...])
  ```

  Args:
    keys: An iterable identifying the features to be crossed. Each element can
      be either:
      * string: Will use the corresponding feature which must be of string type.
      * `CategoricalColumn`: Will use the transformed tensor produced by this
        column. Does not support hashed categorical column.
    hash_bucket_size: An int > 1. The number of buckets.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseCrossOp (optional).

  Returns:
    A `CrossedColumn`.

  Raises:
    ValueError: If `len(keys) < 2`.
    ValueError: If any of the keys is neither a string nor `CategoricalColumn`.
    ValueError: If any of the keys is `HashedCategoricalColumn`.
    ValueError: If `hash_bucket_size < 1`.
  """
  if not hash_bucket_size or hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be > 1. '
                     'hash_bucket_size: {}'.format(hash_bucket_size))
  if not keys or len(keys) < 2:
    raise ValueError(
        'keys must be a list with length > 1. Given: {}'.format(keys))
  for key in keys:
    if (not isinstance(key, six.string_types) and
        not isinstance(key, (CategoricalColumn, fc_old._CategoricalColumn))):  # pylint: disable=protected-access
      raise ValueError(
          'Unsupported key type. All keys must be either string, or '
          'categorical column except HashedCategoricalColumn. '
          'Given: {}'.format(key))
    if isinstance(key,
                  (HashedCategoricalColumn, fc_old._HashedCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'categorical_column_with_hash_bucket is not supported for crossing. '
          'Hashing before crossing will increase probability of collision. '
          'Instead, use the feature name as a string. Given: {}'.format(key))
  return CrossedColumn(
      keys=tuple(keys), hash_bucket_size=hash_bucket_size, hash_key=hash_key)


@six.add_metaclass(abc.ABCMeta)
class FeatureColumn(object):
  """Represents a feature column abstraction.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  To distinguish between the concept of a feature family and a specific binary
  feature within a family, we refer to a feature family like "country" as a
  feature column. For example, we can have a feature in a `tf.Example` format:
    {key: "country",  value: [ "US" ]}
  In this example the value of feature is "US" and "country" refers to the
  column of the feature.

  This class is an abstract class. Users should not create instances of this.
  """

  @abc.abstractproperty
  def name(self):
    """Returns string. Used for naming."""
    pass

  @property
  def var_scope_name(self):
    return self.name

  @property
  def embedding_name(self):
    return self.name

  def __lt__(self, other):
    """Allows feature columns to be sorted in Python 3 as they are in Python 2.

    Feature columns need to occasionally be sortable, for example when used as
    keys in a features dictionary passed to a layer.

    In CPython, `__lt__` must be defined for all objects in the
    sequence being sorted.

    If any objects in teh sequence being sorted do not have an `__lt__` method
    compatible with feature column objects (such as strings), then CPython will
    fall back to using the `__gt__` method below.
    https://docs.python.org/3/library/stdtypes.html#list.sort

    Args:
      other: The other object to compare to.

    Returns:
      True if the string representation of this object is lexicographically less
      than the string representation of `other`. For FeatureColumn objects,
      this looks like "<__main__.FeatureColumn object at 0xa>".
    """
    return str(self) < str(other)

  def __gt__(self, other):
    """Allows feature columns to be sorted in Python 3 as they are in Python 2.

    Feature columns need to occasionally be sortable, for example when used as
    keys in a features dictionary passed to a layer.

    `__gt__` is called when the "other" object being compared during the sort
    does not have `__lt__` defined.
    Example: http://gpaste/4803354716798976

    Args:
      other: The other object to compare to.

    Returns:
      True if the string representation of this object is lexicographically
      greater than the string representation of `other`. For FeatureColumn
      objects, this looks like "<__main__.FeatureColumn object at 0xa>".
    """
    return str(self) > str(other)

  @abc.abstractmethod
  def transform_feature(self, transformation_cache, state_manager):
    """Returns intermediate representation (usually a `Tensor`).

    Uses `transformation_cache` to create an intermediate representation
    (usually a `Tensor`) that other feature columns can use.

    Example usage of `transformation_cache`:
    Let's say a Feature column depends on raw feature ('raw') and another
    `FeatureColumn` (input_fc). To access corresponding `Tensor`s,
    transformation_cache will be used as follows:

    ```python
    raw_tensor = transformation_cache.get('raw', state_manager)
    fc_tensor = transformation_cache.get(input_fc, state_manager)
    ```

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Transformed feature `Tensor`.
    """
    pass

  @abc.abstractproperty
  def parse_example_spec(self):
    """Returns a `tf.Example` parsing spec as dict.

    It is used for get_parsing_spec for `tf.io.parse_example`. Returned spec is
    a dict from keys ('string') to `VarLenFeature`, `FixedLenFeature`, and other
    supported objects. Please check documentation of `tf.io.parse_example` for
    all supported spec objects.

    Let's say a Feature column depends on raw feature ('raw') and another
    `FeatureColumn` (input_fc). One possible implementation of
    parse_example_spec is as follows:

    ```python
    spec = {'raw': tf.io.FixedLenFeature(...)}
    spec.update(input_fc.parse_example_spec)
    return spec
    ```
    """
    pass

  def create_state(self, state_manager):
    """Uses the `state_manager` to create state for the FeatureColumn.

    Args:
      state_manager: A `StateManager` to create / access resources such as
        lookup tables and variables.
    """
    pass

  @abc.abstractproperty
  def _is_v2_column(self):
    """Returns whether this FeatureColumn is fully conformant to the new API.

    This is needed for composition type cases where an EmbeddingColumn etc.
    might take in old categorical columns as input and then we want to use the
    old API.
    """
    pass

  @abc.abstractproperty
  def parents(self):
    """Returns a list of immediate raw feature and FeatureColumn dependencies.

    For example:
    # For the following feature columns
    a = numeric_column('f1')
    c = crossed_column(a, 'f2')
    # The expected parents are:
    a.parents = ['f1']
    c.parents = [a, 'f2']
    """
    pass

  @abc.abstractmethod
  def _get_config(self):
    """Returns the config of the feature column.

    A FeatureColumn config is a Python dictionary (serializable) containing the
    configuration of a FeatureColumn. The same FeatureColumn can be
    reinstantiated later from this configuration.

    The config of a feature column does not include information about feature
    columns depending on it nor the FeatureColumn class name.

    Example with (de)serialization practices followed in this file:
    ```python
    class SerializationExampleFeatureColumn(
        FeatureColumn, collections.namedtuple(
            'SerializationExampleFeatureColumn',
            ('dimension', 'parent', 'dtype', 'normalizer_fn'))):

      def _get_config(self):
        # Create a dict from the namedtuple.
        # Python attribute literals can be directly copied from / to the config.
        # For example 'dimension', assuming it is an integer literal.
        config = dict(zip(self._fields, self))

        # (De)serialization of parent FeatureColumns should use the provided
        # (de)serialize_feature_column() methods that take care of de-duping.
        config['parent'] = serialize_feature_column(self.parent)

        # Many objects provide custom (de)serialization e.g: for tf.DType
        # tf.DType.name, tf.as_dtype() can be used.
        config['dtype'] = self.dtype.name

        # Non-trivial dependencies should be Keras-(de)serializable.
        config['normalizer_fn'] = generic_utils.serialize_keras_object(
            self.normalizer_fn)

        return config

      @classmethod
      def _from_config(cls, config, custom_objects=None, columns_by_name=None):
        # This should do the inverse transform from `_get_config` and construct
        # the namedtuple.
        kwargs = config.copy()
        kwargs['parent'] = deserialize_feature_column(
            config['parent'], custom_objects, columns_by_name)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        kwargs['normalizer_fn'] = generic_utils.deserialize_keras_object(
          config['normalizer_fn'], custom_objects=custom_objects)
        return cls(**kwargs)

    ```
    Returns:
      A serializable Dict that can be used to deserialize the object with
      from_config.
    """
    pass

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """Creates a FeatureColumn from its config.

    This method should be the reverse of `_get_config`, capable of instantiating
    the same FeatureColumn from the config dictionary. See `_get_config` for an
    example of common (de)serialization practices followed in this file.

    TODO(b/118939620): This is a private method until consensus is reached on
    supporting object deserialization deduping within Keras.

    Args:
      config: A Dict config acquired with `_get_config`.
      custom_objects: Optional dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization.
      columns_by_name: A Dict[String, FeatureColumn] of existing columns in
        order to avoid duplication. Should be passed to any calls to
        deserialize_feature_column().

    Returns:
      A FeatureColumn for the input config.
    """
    pass


class DenseColumn(FeatureColumn):
  """Represents a column which can be represented as `Tensor`.

  Some examples of this type are: numeric_column, embedding_column,
  indicator_column.
  """

  @abc.abstractproperty
  def variable_shape(self):
    """`TensorShape` of `get_dense_tensor`, without batch dimension."""
    pass

  @abc.abstractmethod
  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns a `Tensor`.

    The output of this function will be used by model-builder-functions. For
    example the pseudo code of `input_layer` will be like:

    ```python
    def input_layer(features, feature_columns, ...):
      outputs = [fc.get_dense_tensor(...) for fc in feature_columns]
      return tf.concat(outputs)
    ```

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      `Tensor` of shape [batch_size] + `variable_shape`.
    """
    pass

  def output_shape(self, inputs):
    """Tuple of column output shape"""
    batch_size = array_ops.shape(inputs)[0]
    num_elements = self.variable_shape.num_elements()
    return (batch_size, num_elements)


def is_feature_column_v2(feature_columns):
  """Returns True if all feature columns are V2."""
  for feature_column in feature_columns:
    if not isinstance(feature_column, FeatureColumn):
      return False
    if not feature_column._is_v2_column:  # pylint: disable=protected-access
      return False
  return True


def _create_weighted_sum(column, transformation_cache, state_manager,
                         sparse_combiner, weight_var):
  """Creates a weighted sum for a dense/categorical column for linear_model."""
  if isinstance(column, CategoricalColumn):
    return _create_categorical_column_weighted_sum(
        column=column,
        transformation_cache=transformation_cache,
        state_manager=state_manager,
        sparse_combiner=sparse_combiner,
        weight_var=weight_var)
  else:
    return _create_dense_column_weighted_sum(
        column=column,
        transformation_cache=transformation_cache,
        state_manager=state_manager,
        weight_var=weight_var)


def _create_dense_column_weighted_sum(column, transformation_cache,
                                      state_manager, weight_var):
  """Create a weighted sum of a dense column for linear_model."""
  tensor = column.get_dense_tensor(transformation_cache, state_manager)
  num_elements = column.variable_shape.num_elements()
  batch_size = array_ops.shape(tensor)[0]
  tensor = array_ops.reshape(tensor, shape=(batch_size, num_elements))
  return math_ops.matmul(tensor, weight_var, name='weighted_sum')


class CategoricalColumn(FeatureColumn):
  """Represents a categorical feature.

  A categorical feature typically handled with a `tf.SparseTensor` of IDs.
  """

  IdWeightPair = collections.namedtuple(  # pylint: disable=invalid-name
      'IdWeightPair', ('id_tensor', 'weight_tensor'))

  @abc.abstractproperty
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    pass

  @abc.abstractmethod
  def get_sparse_tensors(self, transformation_cache, state_manager):
    """Returns an IdWeightPair.

    `IdWeightPair` is a pair of `SparseTensor`s which represents ids and
    weights.

    `IdWeightPair.id_tensor` is typically a `batch_size` x `num_buckets`
    `SparseTensor` of `int64`. `IdWeightPair.weight_tensor` is either a
    `SparseTensor` of `float` or `None` to indicate all weights should be
    taken to be 1. If specified, `weight_tensor` must have exactly the same
    shape and indices as `sp_ids`. Expected `SparseTensor` is same as parsing
    output of a `VarLenFeature` which is a ragged matrix.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    pass


def _create_categorical_column_weighted_sum(
    column, transformation_cache, state_manager, sparse_combiner, weight_var):
  # pylint: disable=g-doc-return-or-yield,g-doc-args
  """Create a weighted sum of a categorical column for linear_model.

  Note to maintainer: As implementation details, the weighted sum is
  implemented via embedding_lookup_sparse toward efficiency. Mathematically,
  they are the same.

  To be specific, conceptually, categorical column can be treated as multi-hot
  vector. Say:

  ```python
    x = [0 0 1]  # categorical column input
    w = [a b c]  # weights
  ```
  The weighted sum is `c` in this case, which is same as `w[2]`.

  Another example is

  ```python
    x = [0 1 1]  # categorical column input
    w = [a b c]  # weights
  ```
  The weighted sum is `b + c` in this case, which is same as `w[2] + w[3]`.

  For both cases, we can implement weighted sum via embedding_lookup with
  sparse_combiner = "sum".
  """

  sparse_tensors = column.get_sparse_tensors(transformation_cache,
                                             state_manager)
  id_tensor = sparse_ops.sparse_reshape(sparse_tensors.id_tensor, [
      array_ops.shape(sparse_tensors.id_tensor)[0], -1
  ])
  weight_tensor = sparse_tensors.weight_tensor
  if weight_tensor is not None:
    weight_tensor = sparse_ops.sparse_reshape(
        weight_tensor, [array_ops.shape(weight_tensor)[0], -1])

  return embedding_ops.safe_embedding_lookup_sparse(
      weight_var,
      id_tensor,
      sparse_weights=weight_tensor,
      combiner=sparse_combiner,
      name='weighted_sum')


class SequenceDenseColumn(FeatureColumn):
  """Represents dense sequence data."""

  TensorSequenceLengthPair = collections.namedtuple(  # pylint: disable=invalid-name
      'TensorSequenceLengthPair', ('dense_tensor', 'sequence_length'))

  @abc.abstractmethod
  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """Returns a `TensorSequenceLengthPair`.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    pass


class FeatureTransformationCache(object):
  """Handles caching of transformations while building the model.

  `FeatureColumn` specifies how to digest an input column to the network. Some
  feature columns require data transformations. This class caches those
  transformations.

  Some features may be used in more than one place. For example, one can use a
  bucketized feature by itself and a cross with it. In that case we
  should create only one bucketization op instead of creating ops for each
  feature column separately. To handle re-use of transformed columns,
  `FeatureTransformationCache` caches all previously transformed columns.

  Example:
  We're trying to use the following `FeatureColumn`s:

  ```python
  bucketized_age = fc.bucketized_column(fc.numeric_column("age"), ...)
  keywords = fc.categorical_column_with_hash_buckets("keywords", ...)
  age_X_keywords = fc.crossed_column([bucketized_age, "keywords"])
  ... = linear_model(features,
                          [bucketized_age, keywords, age_X_keywords]
  ```

  If we transform each column independently, then we'll get duplication of
  bucketization (one for cross, one for bucketization itself).
  The `FeatureTransformationCache` eliminates this duplication.
  """

  def __init__(self, features):
    """Creates a `FeatureTransformationCache`.

    Args:
      features: A mapping from feature column to objects that are `Tensor` or
        `SparseTensor`, or can be converted to same via
        `sparse_tensor.convert_to_tensor_or_sparse_tensor`. A `string` key
        signifies a base feature (not-transformed). A `FeatureColumn` key
        means that this `Tensor` is the output of an existing `FeatureColumn`
        which can be reused.
    """
    self._features = features.copy()
    self._feature_tensors = {}

  def set(self, key, value):
    if key in self._feature_tensors:
      self._feature_tensors[key] = value
    elif key in self._features:
      self._features[key] = value
    else:
      raise ValueError("LazyBUilder set error: Key name not appear "
                       "in Lazybuilder, key name: ", key)

  def get_features(self):
    feature_result = self._features.copy()
    feature_result.update(self._feature_tensors)
    return feature_result

  def get(self, key, state_manager):
    """Returns a `Tensor` for the given key.

    A `str` key is used to access a base feature (not-transformed). When a
    `FeatureColumn` is passed, the transformed feature is returned if it
    already exists, otherwise the given `FeatureColumn` is asked to provide its
    transformed output, which is then cached.

    Args:
      key: a `str` or a `FeatureColumn`.
      state_manager: A StateManager object that holds the FeatureColumn state.

    Returns:
      The transformed `Tensor` corresponding to the `key`.

    Raises:
      ValueError: if key is not found or a transformed `Tensor` cannot be
        computed.
    """
    if key in self._feature_tensors:
      # FeatureColumn is already transformed or converted.
      return self._feature_tensors[key]

    if key in self._features:
      feature_tensor = self._get_raw_feature_as_tensor(key)
      self._feature_tensors[key] = feature_tensor
      return feature_tensor

    if isinstance(key, six.string_types):
      raise ValueError('Feature {} is not in features dictionary.'.format(key))

    if not isinstance(key, FeatureColumn):
      raise TypeError('"key" must be either a "str" or "FeatureColumn". '
                      'Provided: {}'.format(key))

    column = key
    logging.debug('Transforming feature_column %s.', column)
    transformed = column.transform_feature(self, state_manager)
    if transformed is None:
      raise ValueError('Column {} is not supported.'.format(column.name))
    self._feature_tensors[column] = transformed
    return transformed

  def _get_raw_feature_as_tensor(self, key):
    """Gets the raw_feature (keyed by `key`) as `tensor`.

    The raw feature is converted to (sparse) tensor and maybe expand dim.

    For both `Tensor` and `SparseTensor`, the rank will be expanded (to 2) if
    the rank is 1. This supports dynamic rank also. For rank 0 raw feature, will
    error out as it is not supported.

    Args:
      key: A `str` key to access the raw feature.

    Returns:
      A `Tensor` or `SparseTensor`.

    Raises:
      ValueError: if the raw feature has rank 0.
    """
    raw_feature = self._features[key]
    feature_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        raw_feature)

    def expand_dims(input_tensor):
      # Input_tensor must have rank 1.
      if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
        return sparse_ops.sparse_reshape(
            input_tensor, [array_ops.shape(input_tensor)[0], 1])
      else:
        return array_ops.expand_dims(input_tensor, -1)

    rank = feature_tensor.get_shape().ndims
    if rank is not None:
      if rank == 0:
        raise ValueError(
            'Feature (key: {}) cannot have rank 0. Given: {}'.format(
                key, feature_tensor))
      return feature_tensor if rank != 1 else expand_dims(feature_tensor)

    # Handle dynamic rank.
    with ops.control_dependencies([
        check_ops.assert_positive(
            array_ops.rank(feature_tensor),
            message='Feature (key: {}) cannot have rank 0. Given: {}'.format(
                key, feature_tensor))]):
      return control_flow_ops.cond(
          math_ops.equal(1, array_ops.rank(feature_tensor)),
          lambda: expand_dims(feature_tensor),
          lambda: feature_tensor)


# TODO(ptucker): Move to third_party/tensorflow/python/ops/sparse_ops.py
def _to_sparse_input_and_drop_ignore_values(input_tensor, ignore_value=None):
  """Converts a `Tensor` to a `SparseTensor`, dropping ignore_value cells.

  If `input_tensor` is already a `SparseTensor`, just return it.

  Args:
    input_tensor: A string or integer `Tensor`.
    ignore_value: Entries in `dense_tensor` equal to this value will be
      absent from the resulting `SparseTensor`. If `None`, default value of
      `dense_tensor`'s dtype will be used ('' for `str`, -1 for `int`).

  Returns:
    A `SparseTensor` with the same shape as `input_tensor`.

  Raises:
    ValueError: when `input_tensor`'s rank is `None`.
  """
  input_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
      input_tensor)
  if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
    return input_tensor
  with ops.name_scope(None, 'to_sparse_input', (input_tensor, ignore_value,)):
    if ignore_value is None:
      if input_tensor.dtype == dtypes.string:
        # Exception due to TF strings are converted to numpy objects by default.
        ignore_value = ''
      elif input_tensor.dtype.is_integer:
        ignore_value = -1  # -1 has a special meaning of missing feature
      else:
        # NOTE: `as_numpy_dtype` is a property, so with the parentheses this is
        # constructing a new numpy object of the given type, which yields the
        # default value for that type.
        ignore_value = input_tensor.dtype.as_numpy_dtype()
    ignore_value = math_ops.cast(
        ignore_value, input_tensor.dtype, name='ignore_value')
    indices = array_ops.where_v2(
        math_ops.not_equal(input_tensor, ignore_value), name='indices')
    return sparse_tensor_lib.SparseTensor(
        indices=indices,
        values=array_ops.gather_nd(input_tensor, indices, name='values'),
        dense_shape=array_ops.shape(
            input_tensor, out_type=dtypes.int64, name='dense_shape'))


def _normalize_feature_columns(feature_columns):
  """Normalizes the `feature_columns` input.

  This method converts the `feature_columns` to list type as best as it can. In
  addition, verifies the type and other parts of feature_columns, required by
  downstream library.

  Args:
    feature_columns: The raw feature columns, usually passed by users.

  Returns:
    The normalized feature column list.

  Raises:
    ValueError: for any invalid inputs, such as empty, duplicated names, etc.
  """
  if isinstance(feature_columns, FeatureColumn):
    feature_columns = [feature_columns]

  if isinstance(feature_columns, collections_abc.Iterator):
    feature_columns = list(feature_columns)

  if isinstance(feature_columns, dict):
    raise ValueError('Expected feature_columns to be iterable, found dict.')

  for column in feature_columns:
    if not isinstance(column, FeatureColumn):
      raise ValueError('Items of feature_columns must be a FeatureColumn. '
                       'Given (type {}): {}.'.format(type(column), column))
  if not feature_columns:
    raise ValueError('feature_columns must not be empty.')
  name_to_column = {}
  for column in feature_columns:
    if column.name in name_to_column:
      raise ValueError('Duplicate feature column name found for columns: {} '
                       'and {}. This usually means that these columns refer to '
                       'same base feature. Either one must be discarded or a '
                       'duplicated but renamed item must be inserted in '
                       'features dict.'.format(column,
                                               name_to_column[column.name]))
    name_to_column[column.name] = column

  return sorted(feature_columns, key=lambda x: x.name)


class NumericColumn(
    DenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'NumericColumn',
        ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
  """see `numeric_column`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {
        self.key:
            parsing_ops.FixedLenFeature(self.shape, self.dtype,
                                        self.default_value)
    }

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError(
          'The corresponding Tensor of numerical column must be a Tensor. '
          'SparseTensor is not supported. key: {}'.format(self.key))
    if self.normalizer_fn is not None:
      input_tensor = self.normalizer_fn(input_tensor)
    return math_ops.cast(input_tensor, dtypes.float32)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = inputs.get(self.key)
    return self._transform_input_tensor(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class.

    In this case, we apply the `normalizer_fn` to the input tensor.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Normalized input tensor.
    Raises:
      ValueError: If a SparseTensor is passed in.
    """
    input_tensor = transformation_cache.get(self.key, state_manager)
    return self._transform_input_tensor(input_tensor)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(self.shape)

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.variable_shape

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns dense `Tensor` representing numeric feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.
    """
    # Feature has been already transformed. Return the intermediate
    # representation created by _transform_feature.
    return transformation_cache.get(self, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    return inputs.get(self)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['normalizer_fn'] = generic_utils.serialize_keras_object(
        self.normalizer_fn)
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['normalizer_fn'] = generic_utils.deserialize_keras_object(
        config['normalizer_fn'], custom_objects=custom_objects)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])

    return cls(**kwargs)


class SparseNumericColumn(
    DenseColumn, CategoricalColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SparseNumericColumn',
        ('key', 'shape', 'dtype'))):
  """see `sparse_numeric_column`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {
        self.key:
            parsing_ops.VarLenFeature(self.dtype)
    }

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError(
          'The corresponding Tensor of sparse numerical column must be a SparseTensor. '
          'Tensor is not supported. key: {}'.format(self.key))
    fc_utils.assert_float_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    return input_tensor

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = inputs.get(self.key)
    return self._transform_input_tensor(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Input tensor.
    Raises:
      ValueError: If a Tensor is passed in.
    """
    input_tensor = transformation_cache.get(self.key, state_manager)
    return self._transform_input_tensor(input_tensor)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    if self.shape is not None:
      return tensor_shape.TensorShape(self.shape)
    else:
      # not define shape
      return None

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.variable_shape

  def output_shape(self, inputs):
    """See `DenseColumn` base class."""
    if self.shape is not None:
      batch_size = array_ops.shape(inputs)[0]
      num_elements = self.variable_shape.num_elements()
      return (batch_size, num_elements)
    else:
      # keep origin sparse tensor's dense shape
      return array_ops.shape(inputs)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _output_shape(self, inputs):
    return self.output_shape(inputs)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns dense `Tensor` representing sparse numeric feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.
    """
    # Feature has been already transformed. Return the intermediate
    # representation created by _transform_feature.
    input_tensor = transformation_cache.get(self, state_manager)
    return sparse_ops.sparse_tensor_to_dense(input_tensor, default_value=0)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return sparse_ops.sparse_tensor_to_dense(input_tensor, default_value=0)

  @property
  def num_buckets(self):
    """See `CategoricalColumn` base class."""
    raise ValueError("sparse_numeric_column does not has attr `_num_buckets`"
        " if you want to look up embedding with embedding column, "
        "please use hashtable feature column.")

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """Returns `SparseTensor` representing sparse numeric feature."""
    input_tensor = transformation_cache.get(self, state_manager)
    return CategoricalColumn.IdWeightPair(input_tensor, None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    """Returns `SparseTensor` representing sparse numeric feature."""
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])

    return cls(**kwargs)


class BucketizedColumn(
    DenseColumn,
    CategoricalColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('BucketizedColumn',
                           ('source_column', 'boundaries'))):
  """See `bucketized_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.source_column, FeatureColumn) and
            self.source_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_bucketized'.format(self.source_column.name)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.source_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.source_column._parse_example_spec  # pylint: disable=protected-access

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    """Returns bucketized categorical `source_column` tensor."""
    source_tensor = inputs.get(self.source_column)
    return math_ops._bucketize(  # pylint: disable=protected-access
        source_tensor,
        boundaries=self.boundaries)

  def transform_feature(self, transformation_cache, state_manager):
    """Returns bucketized categorical `source_column` tensor."""
    source_tensor = transformation_cache.get(self.source_column, state_manager)
    return math_ops._bucketize(  # pylint: disable=protected-access
        source_tensor,
        boundaries=self.boundaries)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(
        tuple(self.source_column.shape) + (len(self.boundaries) + 1,))

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.variable_shape

  def _get_dense_tensor_for_input_tensor(self, input_tensor):
    return array_ops.one_hot(
        indices=math_ops.cast(input_tensor, dtypes.int64),
        depth=len(self.boundaries) + 1,
        on_value=1.,
        off_value=0.)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns one hot encoded dense `Tensor`."""
    input_tensor = transformation_cache.get(self, state_manager)
    return self._get_dense_tensor_for_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return self._get_dense_tensor_for_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """See `CategoricalColumn` base class."""
    # By construction, source_column is always one-dimensional.
    return (len(self.boundaries) + 1) * self.source_column.shape[0]

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def _get_sparse_tensors_for_input_tensor(self, input_tensor):
    batch_size = array_ops.shape(input_tensor)[0]
    # By construction, source_column is always one-dimensional.
    source_dimension = self.source_column.shape[0]

    i1 = array_ops.reshape(
        array_ops.tile(
            array_ops.expand_dims(math_ops.range(0, batch_size), 1),
            [1, source_dimension]),
        (-1,))
    i2 = array_ops.tile(math_ops.range(0, source_dimension), [batch_size])
    # Flatten the bucket indices and unique them across dimensions
    # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
    bucket_indices = (
        array_ops.reshape(input_tensor, (-1,)) +
        (len(self.boundaries) + 1) * i2)

    indices = math_ops.cast(
        array_ops.transpose(array_ops.stack((i1, i2))), dtypes.int64)
    dense_shape = math_ops.cast(
        array_ops.stack([batch_size, source_dimension]), dtypes.int64)
    sparse_tensor = sparse_tensor_lib.SparseTensor(
        indices=indices,
        values=bucket_indices,
        dense_shape=dense_shape)
    return CategoricalColumn.IdWeightPair(sparse_tensor, None)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    input_tensor = transformation_cache.get(self, state_manager)
    return self._get_sparse_tensors_for_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return self._get_sparse_tensors_for_input_tensor(input_tensor)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.source_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['source_column'] = serialize_feature_column(self.source_column)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['source_column'] = deserialize_feature_column(
        config['source_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


class SparseBucketizedColumn(
    DenseColumn,
    CategoricalColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('SparseBucketizedColumn',
                           ('source_column', 'boundaries'))):
  """See `sparse_bucketized_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.source_column, FeatureColumn) and
            self.source_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_sparse_bucketized'.format(self.source_column.name)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.source_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.source_column._parse_example_spec  # pylint: disable=protected-access

  def _trasform_input_tensor(self, input_tensor):
    input_tensor = _to_sparse_input_and_drop_ignore_values(input_tensor)
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseBucketizeColumn input must be a SparseTensor.')
    fc_utils.assert_float_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.name))
    return sparse_tensor_lib.SparseTensor(
        input_tensor.indices,
        math_ops._bucketize(  # pylint: disable=protected-access
            input_tensor.values,
            boundaries=self.boundaries),
        input_tensor.dense_shape)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    """Returns bucketized categorical `source_column` tensor."""
    source_tensor = inputs.get(self.source_column)
    return self._trasform_input_tensor(source_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    """Returns bucketized categorical `source_column` tensor."""
    source_tensor = transformation_cache.get(self.source_column, state_manager)
    return self._trasform_input_tensor(source_tensor)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    if self.source_column.shape is not None:
      return tensor_shape.TensorShape(
          tuple(self.source_column.shape) + (len(self.boundaries) + 1,))
    else:
      return None

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.variable_shape

  def output_shape(self, inputs):
    """See `DenseColumn` base class."""
    if self.source_column.shape is not None:
      batch_size = array_ops.shape(inputs)[0]
      num_elements = self.variable_shape.num_elements()
      return (batch_size, num_elements)
    else:
      # keep origin sparse tensor's dense shape
      batch_size = array_ops.shape(inputs)[0]
      num_elements = math_ops.reduce_prod(array_ops.shape(inputs)[1:])
      return (batch_size, num_elements)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _output_shape(self, inputs):
    return self.output_shape(inputs)


  def _get_dense_tensor_for_input_tensor(self, input_tensor):
    dense_tensor = sparse_ops.sparse_tensor_to_dense(input_tensor, default_value=0)
    return array_ops.one_hot(
        indices=math_ops.cast(dense_tensor, dtypes.int64),
        depth=len(self.boundaries) + 1,
        on_value=1.,
        off_value=0.)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns one hot encoded dense `Tensor`."""
    input_tensor = transformation_cache.get(self, state_manager)
    return self._get_dense_tensor_for_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return self._get_dense_tensor_for_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """See `CategoricalColumn` base class."""
    # By construction, source_column is always one-dimensional.
    return len(self.boundaries) + 1

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    input_tensor = transformation_cache.get(self, state_manager)
    return CategoricalColumn.IdWeightPair(input_tensor, None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return CategoricalColumn.IdWeightPair(input_tensor, None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.source_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['source_column'] = serialize_feature_column(self.source_column)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['source_column'] = deserialize_feature_column(
        config['source_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


class CutoffCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('CutoffCategoricalColumn',
                           ('categorical_column', 'cutoff_length',
                            'cutoff_side', 'cutoff_axis', 'reverse'))):
  """See `cutoff_categorical_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_cutoff'.format(self.categorical_column.name)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  def _trasform_input_tensor(self, input_tensor):
    if isinstance(input_tensor, (tuple, list)):
      id_tensor = input_tensor[0]
      weight_tensor = input_tensor[1]
    else:
      id_tensor = input_tensor
      weight_tensor = None
    id_tensor = sparse_ops.sparse_valid_cutoff(
        id_tensor,
        self.cutoff_axis,
        self.cutoff_length,
        self.cutoff_side,
        self.reverse)
    if weight_tensor is not None:
      weight_tensor = sparse_ops.sparse_valid_cutoff(
          weight_tensor,
          self.cutoff_axis,
          self.cutoff_length,
          self.cutoff_side,
          self.reverse)
    return (id_tensor, weight_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    """Returns tensor after cutoff."""
    source_tensors = inputs.get(self.categorical_column)
    return self._trasform_input_tensor(source_tensors)

  def transform_feature(self, transformation_cache, state_manager):
    """Returns tensor after cutoff."""
    source_tensors = transformation_cache.get(self.categorical_column, state_manager)
    return self._trasform_input_tensor(source_tensors)

  @property
  def num_buckets(self):
    """See `CategoricalColumn` base class."""
    return self.categorical_column.num_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    input_tensors = transformation_cache.get(self, state_manager)
    return CategoricalColumn.IdWeightPair(input_tensors[0], input_tensors[1])

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    del weight_collections
    del trainable
    input_tensors = inputs.get(self)
    return CategoricalColumn.IdWeightPair(input_tensors[0], input_tensors[1])

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(self.categorical_column)
    config['cutoff_length'] = self.cutoff_length
    config['cutoff_side'] = self.cutoff_side
    config['cutoff_axis'] = self.cutoff_axis
    config['reverse'] = self.reverse
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    kwargs['cutoff_length'] = config['categorical_column']
    kwargs['cutoff_side'] = config['cutoff_side']
    kwargs['cutoff_axis'] = config['cutoff_axis']
    kwargs['reverse'] = config['reverse']
    return cls(**kwargs)


class EmbeddingColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'EmbeddingColumn',
        ('categorical_column', 'dimension', 'combiner', 'initializer',
         'ckpt_to_load_from', 'tensor_name_in_ckpt', 'max_norm', 'trainable',
         'coalesced_scope'))):
  """See `embedding_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_embedding'.format(self.categorical_column.name)

  @property
  def var_scope_name(self):
    if self.coalesced_scope:
      return self.coalesced_scope.get_coalesced_name_by_column(self)
    else:
      return self.name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  def transform_feature(self, transformation_cache, state_manager):
    """Transforms underlying `categorical_column`."""
    return transformation_cache.get(self.categorical_column, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    return inputs.get(self.categorical_column)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape([self.dimension])

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.variable_shape

  def create_state(self, state_manager):
    """Creates the embedding lookup variable."""
    if self.coalesced_scope:
      self.coalesced_scope.create_state_by_column(
          self)
    else:
      num_buckets = getattr(self.categorical_column, 'num_buckets',
                            self.categorical_column._num_buckets)  # pylint: disable=protected-access
      embedding_shape = (num_buckets, self.dimension)
      state_manager.create_variable(
          self,
          name='embedding_weights',
          shape=embedding_shape,
          dtype=dtypes.float32,
          trainable=self.trainable,
          use_resource=True,
          initializer=self.initializer)

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    if self.ckpt_to_load_from is not None:
      to_restore = embedding_weights
      if isinstance(to_restore, variables.PartitionedVariable):
        to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
      checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {
          self.tensor_name_in_ckpt: to_restore
      })

    # Return embedding lookup result.
    return embedding_ops.safe_embedding_lookup_sparse(
        embedding_weights=embedding_weights,
        sparse_ids=sparse_ids,
        sparse_weights=sparse_weights,
        combiner=self.combiner,
        name='%s_weights' % self.name,
        max_norm=self.max_norm)

  def _get_dense_tensor_internal_adaptive_helper(self, sparse_tensors,
                                                 hash_embeddings, ev_embeddings):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    if self.ckpt_to_load_from is not None:
      for to_restore in [hash_embeddings, ev_embeddings]:
        if isinstance(to_restore, variables.PartitionedVariable):
          to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
        checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {
            self.tensor_name_in_ckpt: to_restore
        })

    # Return embedding lookup result.
    return embedding_ops.safe_adaptive_embedding_lookup_sparse(
        hash_embedding_weights=hash_embeddings,
        ev_embedding_weights=ev_embeddings,
        sparse_ids=sparse_ids,
        hash_ev_ids=self.categorical_column.hash_ev_ids,
        sparse_weights=sparse_weights,
        combiner=self.combiner,
        name='%s_weights' % self.name,
        max_norm=self.max_norm,
        adaptive_mask_tensor=self.categorical_column.adaptive_mask_tensor)


  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    """Private method that follows the signature of get_dense_tensor."""
    embedding_weights = state_manager.get_variable(
        self, name='embedding_weights')
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def _old_get_dense_tensor_internal(self, sparse_tensors, weight_collections,
                                     trainable):
    """Private method that follows the signature of _get_dense_tensor."""
    embedding_shape = (self.categorical_column._num_buckets, self.dimension)  # pylint: disable=protected-access
    if (weight_collections and
        ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections):
      weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
    if isinstance(self.categorical_column, AdaptiveEmbeddingCategoricalColumn) \
      or isinstance(self.categorical_column, EmbeddingCategoricalColumn):
      if self.categorical_column.partition_num is None:
        partitioner = None
      else:
        partitioner = partitioned_variables.fixed_size_partitioner(self.categorical_column.partition_num)
    if isinstance(self.categorical_column, AdaptiveEmbeddingCategoricalColumn):
      ev_embeddings = variable_scope.get_embedding_variable_internal(
        name="ev_weights",
        embedding_dim=self.dimension,
        initializer=self.initializer,
        trainable=(trainable and self.trainable),
        collections=weight_collections,
        partitioner=partitioner,
        ev_option=self.categorical_column.ev_option)
      hash_embeddings = variable_scope.get_variable(
        name="hash_weights",
        shape=embedding_shape,
        dtype=dtypes.float32,
        initializer=self.initializer,
        trainable=(trainable and self.trainable),
        collections=weight_collections)
      return self._get_dense_tensor_internal_adaptive_helper(sparse_tensors,
                                                             hash_embeddings, ev_embeddings)
    elif isinstance(self.categorical_column, EmbeddingCategoricalColumn):
      embedding_weights = variable_scope.get_embedding_variable_internal(
        name='embedding_weights',
        embedding_dim=self.dimension,
        initializer=self.initializer,
        trainable=self.trainable and trainable,
        collections=weight_collections,
        partitioner=partitioner,
        ev_option=self.categorical_column.ev_option
      )
      return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                    embedding_weights)
    else:
      embedding_weights = variable_scope.get_variable(
        name='embedding_weights',
        shape=embedding_shape,
        dtype=dtypes.float32,
        initializer=self.initializer,
        trainable=self.trainable and trainable,
        collections=weight_collections)
      return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                    embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns tensor after doing the embedding lookup.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Embedding lookup tensor.

    Raises:
      ValueError: `categorical_column` is SequenceCategoricalColumn.
    """
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Get sparse IDs and weights.
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column_v2(
          self, transformation_cache, state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    if isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type _SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column(  # pylint: disable=protected-access
          self, inputs, weight_collections, trainable)
    sparse_tensors = self.categorical_column._get_sparse_tensors(  # pylint: disable=protected-access
        inputs, weight_collections, trainable)
    return self._old_get_dense_tensor_internal(sparse_tensors,
                                               weight_collections, trainable)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))

      raise NotImplementedError(
          'get_sequence_dense_tensor function not implemented on coalesced mod')
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    dense_tensor = self._get_dense_tensor_internal(sparse_tensors,
                                                   state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    if not isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    dense_tensor = self._old_get_dense_tensor_internal(
        sparse_tensors,
        weight_collections=weight_collections,
        trainable=trainable)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    config['initializer'] = initializers.serialize(self.initializer)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    kwargs['initializer'] = initializers.deserialize(
        config['initializer'], custom_objects=custom_objects)
    return cls(**kwargs)


class SharedEmbeddingColumnV2(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SharedEmbeddingColumnV2',
        ('categorical_column', 'dimension', 'shared_name','combiner',
         'initializer', 'ckpt_to_load_from', 'tensor_name_in_ckpt',
         'max_norm', 'trainable', 'coalesced_scope'))):
  """See `shared_embedding_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_shared_embedding_v2'.format(self.categorical_column.name)

  @property
  def var_scope_name(self):
    if self.coalesced_scope:
      return self.coalesced_scope.get_coalesced_name_by_column(self)
    else:
      return self.shared_name

  @property
  def embedding_name(self):
    return self.shared_name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  def transform_feature(self, transformation_cache, state_manager):
    """Transforms underlying `categorical_column`."""
    return transformation_cache.get(self.categorical_column, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    return inputs.get(self.categorical_column)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape([self.dimension])

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.variable_shape

  def get_embedding(self):
    shared_embedding_collection = ops.get_collection(self.embedding_name)
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError(
            'Collection {} can only contain one variable. '
            'Suggested fix A: Choose a unique name for this collection. '
            'Suggested fix B: Do not add any variables to this collection. '
            'The feature_column library already adds a variable under the '
            'hood.'.format(shared_embedding_collection))
      embedding_weights = shared_embedding_collection[0]
      return embedding_weights
    else:
      raise ValueError("Embedding not created yet.")
 
  def create_state(self, state_manager):
    """Creates the embedding lookup variable."""
    if self.coalesced_scope:
      self.coalesced_scope.create_state_by_column(
          self)
    else:
      shared_embedding_collection = ops.get_collection(self.embedding_name)
      if shared_embedding_collection:
        if len(shared_embedding_collection) > 1:
          raise ValueError(
              'Collection {} can only contain one variable. '
              'Suggested fix A: Choose a unique name for this collection. '
              'Suggested fix B: Do not add any variables to this collection. '
              'The feature_column library already adds a variable under the '
              'hood.'.format(shared_embedding_collection))
      else:
        num_buckets = getattr(self.categorical_column, 'num_buckets',
                              self.categorical_column._num_buckets)  # pylint: disable=protected-access
        embedding_shape = (num_buckets, self.dimension)
        state_manager.create_variable(
            self,
            name='embedding_weights',
            shape=embedding_shape,
            dtype=dtypes.float32,
            trainable=self.trainable,
            use_resource=True,
            initializer=self.initializer)
        embedding_weights = state_manager.get_variable(
            self, name='embedding_weights')
        ops.add_to_collection(self.embedding_name,
                              embedding_weights)

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    if self.ckpt_to_load_from is not None:
      to_restore = embedding_weights
      if isinstance(to_restore, variables.PartitionedVariable):
        to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
      checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {
          self.tensor_name_in_ckpt: to_restore
      })

    # Return embedding lookup result.
    return embedding_ops.safe_embedding_lookup_sparse(
        embedding_weights=embedding_weights,
        sparse_ids=sparse_ids,
        sparse_weights=sparse_weights,
        combiner=self.combiner,
        name='%s_weights' % self.name,
        max_norm=self.max_norm)

  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    """Private method that follows the signature of get_dense_tensor."""
    embedding_weights = self.get_embedding()
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def _old_get_dense_tensor_internal(self, sparse_tensors, weight_collections,
                                     trainable):
    """Private method that follows the signature of _get_dense_tensor."""
    embedding_shape = (self.categorical_column._num_buckets, self.dimension)  # pylint: disable=protected-access
    if (weight_collections and
        ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections):
      weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
    embedding_weights = variable_scope.get_variable(
        name='embedding_weights',
        shape=embedding_shape,
        dtype=dtypes.float32,
        initializer=self.initializer,
        trainable=self.trainable and trainable,
        collections=weight_collections)
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns tensor after doing the embedding lookup.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Embedding lookup tensor.

    Raises:
      ValueError: `categorical_column` is SequenceCategoricalColumn.
    """
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Get sparse IDs and weights.
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column_v2(
          self, transformation_cache, state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    if isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type _SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column(  # pylint: disable=protected-access
          self, inputs, weight_collections, trainable)
    sparse_tensors = self.categorical_column._get_sparse_tensors(  # pylint: disable=protected-access
        inputs, weight_collections, trainable)
    return self._old_get_dense_tensor_internal(sparse_tensors,
                                               weight_collections, trainable)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))

      raise NotImplementedError(
          'get_sequence_dense_tensor function not implemented on coalesced mod')
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    dense_tensor = self._get_dense_tensor_internal(sparse_tensors,
                                                   state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    if not isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    dense_tensor = self._old_get_dense_tensor_internal(
        sparse_tensors,
        weight_collections=weight_collections,
        trainable=trainable)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    config['initializer'] = initializers.serialize(self.initializer)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    kwargs['initializer'] = initializers.deserialize(
        config['initializer'], custom_objects=custom_objects)
    return cls(**kwargs)


class SequenceEmbeddingColumn(
    DenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SequenceEmbeddingColumn',
        ('dense_column', 'sequence_length'))):
  """See `sequence_embedding_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.dense_column, FeatureColumn) and
            self.dense_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_sequence'.format(self.dense_column.name)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    out_spec = {}
    dense_spec = self.dense_column.parse_example_spec
    for key, parsing in dense_spec.items():
      if isinstance(parsing, parsing_ops.FixedLenFeature):
        out_spec[key] = parsing_ops.FixedLenSequenceFeature(parsing.shape,
                                                            parsing.dtype,
                                                            True,
                                                            parsing.default_value)
      else:
        out_spec[key] = parsing
    return out_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    out_spec = {}
    dense_spec = self.dense_column._parse_example_spec
    for key, parsing in dense_spec.items():
      if isinstance(parsing, parsing_ops.FixedLenFeature):
        out_spec[key] = parsing_ops.FixedLenSequenceFeature(parsing.shape,
                                                            parsing.dtype,
                                                            True,
                                                            parsing.default_value)
      else:
        out_spec[key] = parsing
    return out_spec

  def create_state(self, state_manager):
    return self.dense_column.create_state(state_manager)

  def transform_feature(self, transformation_cache, state_manager):
    """Transforms underlying `categorical_column`."""
    return transformation_cache.get(self.dense_column, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    return inputs.get(self.dense_column)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return self.dense_column.variable_shape

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.dense_column._variable_shape

  def output_shape(self, inputs):
    """See `DenseColumn` base class."""
    if self._variable_shape is not None:
      num_elements = self.variable_shape.num_elements()
    else:
      num_elements = array_ops.shape(inputs)[-1]
    return (-1, self.sequence_length, num_elements)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _output_shape(self, inputs):
    if self._variable_shape is not None:
      num_elements = self._variable_shape.num_elements()
    else:
      num_elements = array_ops.shape(inputs)[-1]
    return (-1, self.sequence_length, num_elements)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns tensor after doing the embedding lookup.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Embedding lookup tensor.
    """
    return self.dense_column.get_dense_tensor(
        transformation_cache,
        state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return self.dense_column._get_dense_tensor(  # pylint: disable=protected-access
        inputs=inputs,
        weight_collections=weight_collections,
        trainable=trainable)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.dense_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['dense_column'] = serialize_feature_column(
        self.dense_column)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dense_column'] = deserialize_feature_column(
        config['dense_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


def _raise_shared_embedding_column_error():
  raise ValueError('SharedEmbeddingColumns are not supported in '
                   '`linear_model` or `input_layer`. Please use '
                   '`DenseFeatures` or `LinearModel` instead.')


class SequenceMultiHashEmbeddingColumn(
    DenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SequenceMultiHashEmbeddingColumn',
        ('dense_column', 'sequence_length'))):
  """See `sequence_embedding_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.dense_column, FeatureColumn) and
            self.dense_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_sequence'.format(self.dense_column.name)

  @property
  def categorical_column(self):
    return self.dense_column.categorical_column

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    out_spec = {}
    dense_spec = self.dense_column.parse_example_spec
    for key, parsing in dense_spec.items():
      if isinstance(parsing, parsing_ops.FixedLenFeature):
        out_spec[key] = parsing_ops.FixedLenSequenceFeature(parsing.shape,
                                                            parsing.dtype,
                                                            True,
                                                            parsing.default_value)
      else:
        out_spec[key] = parsing
    return out_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    out_spec = {}
    dense_spec = self.dense_column._parse_example_spec
    for key, parsing in dense_spec.items():
      if isinstance(parsing, parsing_ops.FixedLenFeature):
        out_spec[key] = parsing_ops.FixedLenSequenceFeature(parsing.shape,
                                                            parsing.dtype,
                                                            True,
                                                            parsing.default_value)
      else:
        out_spec[key] = parsing
    return out_spec

  def transform_feature(self, transformation_cache, state_manager):
    """Transforms underlying `categorical_column`."""
    return transformation_cache.get(self.dense_column, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    return inputs.get(self.dense_column)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return self.dense_column.variable_shape

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return self.dense_column._variable_shape

  def output_shape(self, inputs):
    """See `DenseColumn` base class."""
    if self._variable_shape is not None:
      num_elements = self.variable_shape.num_elements()
    else:
      num_elements = array_ops.shape(inputs)[-1]
    return (-1, self.sequence_length, num_elements)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _output_shape(self, inputs):
    if self._variable_shape is not None:
      num_elements = self._variable_shape.num_elements()
    else:
      num_elements = array_ops.shape(inputs)[-1]
    return (-1, self.sequence_length, num_elements)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns tensor after doing the embedding lookup.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Embedding lookup tensor.
    """
    return self.dense_column.get_dense_tensor(
        transformation_cache,
        state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return self.dense_column._get_dense_tensor(  # pylint: disable=protected-access
        inputs=inputs,
        weight_collections=weight_collections,
        trainable=trainable)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.dense_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['dense_column'] = serialize_feature_column(
        self.dense_column)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dense_column'] = deserialize_feature_column(
        config['dense_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


class SharedEmbeddingColumnCreator(tracking.AutoTrackable):

  def __init__(self,
               dimension,
               initializer,
               ckpt_to_load_from,
               tensor_name_in_ckpt,
               num_buckets,
               trainable,
               name='shared_embedding_column_creator'):
    self._dimension = dimension
    self._initializer = initializer
    self._ckpt_to_load_from = ckpt_to_load_from
    self._tensor_name_in_ckpt = tensor_name_in_ckpt
    self._num_buckets = num_buckets
    self._trainable = trainable
    self._name = name
    # Map from graph keys to embedding_weight variables.
    self._embedding_weights = {}

  def __call__(self,
               categorical_column,
               combiner,
               max_norm,
               coalesced_scope=None):
    return SharedEmbeddingColumn(categorical_column,
                                 self,
                                 combiner,
                                 max_norm,
                                 coalesced_scope)

  @property
  def embedding_weights(self):
    key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    if key not in self._embedding_weights:
      embedding_shape = (self._num_buckets, self._dimension)
      var = variable_scope.get_variable(
          name=self._name,
          shape=embedding_shape,
          dtype=dtypes.float32,
          initializer=self._initializer,
          trainable=self._trainable)

      if self._ckpt_to_load_from is not None:
        to_restore = var
        if isinstance(to_restore, variables.PartitionedVariable):
          to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
        checkpoint_utils.init_from_checkpoint(
            self._ckpt_to_load_from, {self._tensor_name_in_ckpt: to_restore})
      self._embedding_weights[key] = var
    return self._embedding_weights[key]

  @property
  def dimension(self):
    return self._dimension

  @property
  def name(self):
    return self._name


class SharedEmbeddingColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SharedEmbeddingColumn',
        ('categorical_column', 'shared_embedding_column_creator', 'combiner',
         'max_norm', 'coalesced_scope'))):
  """See `embedding_column`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_shared_embedding'.format(self.categorical_column.name)

  @property
  def var_scope_name(self):
    if self.coalesced_scope:
      return self.coalesced_scope.get_coalesced_name_by_column(self)
    else:
      return self.name

  @property
  def embedding_name(self):
    return self.shared_embedding_column_creator.name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  def _parse_example_spec(self):
    return _raise_shared_embedding_column_error()

  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class."""
    return transformation_cache.get(self.categorical_column, state_manager)

  def _transform_feature(self, inputs):
    return _raise_shared_embedding_column_error()

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(
        [self.shared_embedding_column_creator.dimension])

  @property
  def _variable_shape(self):
    return _raise_shared_embedding_column_error()

  def create_state(self, state_manager):
    """Creates the embedding lookup variable."""
    if self.coalesced_scope:
      self.coalesced_scope.create_state_by_column(
        self)
    else:
      super(SharedEmbeddingColumn, self).create_state(state_manager)

  def _get_dense_tensor_internal(self, transformation_cache, state_manager):
    """Private method that follows the signature of _get_dense_tensor."""
    # This method is called from a variable_scope with name _var_scope_name,
    # which is shared among all shared embeddings. Open a name_scope here, so
    # that the ops for different columns have distinct names.
    with ops.name_scope(None, default_name=self.name):
      # Get sparse IDs and weights.
      sparse_tensors = self.categorical_column.get_sparse_tensors(
          transformation_cache, state_manager)
      sparse_ids = sparse_tensors.id_tensor
      sparse_weights = sparse_tensors.weight_tensor

      embedding_weights = self.shared_embedding_column_creator.embedding_weights

      # Return embedding lookup result.
      return embedding_ops.safe_embedding_lookup_sparse(
          embedding_weights=embedding_weights,
          sparse_ids=sparse_ids,
          sparse_weights=sparse_weights,
          combiner=self.combiner,
          name='%s_weights' % self.name,
          max_norm=self.max_norm)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns the embedding lookup result."""
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column_v2(
          self, transformation_cache, state_manager)
    return self._get_dense_tensor_internal(transformation_cache, state_manager)

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return _raise_shared_embedding_column_error()

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    if self.coalesced_scope:
      raise NotImplementedError(
          'get_sequence_dense_tensor function not implemented on coalesced mod')
    dense_tensor = self._get_dense_tensor_internal(transformation_cache,
                                                   state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    return _raise_shared_embedding_column_error()

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    raise NotImplementedError()

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    raise NotImplementedError()


def _check_shape(shape, key):
  """Returns shape if it's valid, raises error otherwise."""
  assert shape is not None
  if not nest.is_sequence(shape):
    shape = [shape]
  shape = tuple(shape)
  for dimension in shape:
    if not isinstance(dimension, int):
      raise TypeError('shape dimensions must be integer. '
                      'shape: {}, key: {}'.format(shape, key))
    if dimension < 1:
      raise ValueError('shape dimensions must be greater than 0. '
                       'shape: {}, key: {}'.format(shape, key))
  return shape


class SharedMultiHashEmbeddingColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'EmbeddingColumn',
        ('categorical_column', 'dimension', 'shared_name','hash_combiner',
         'combiner', 'initializer', 'ckpt_to_load_from', 'tensor_name_in_ckpt',
         'max_norm', 'trainable', 'coalesced_scope'))):
  """See `embedding_column`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_shared_multi_hash_embedding'.format(self.categorical_column.name)

  @property
  def var_scope_name(self):
    return self.shared_name

  @property
  def embedding_name(self):
    return self.shared_name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  def _parse_example_spec(self):
    return _raise_shared_embedding_column_error()

  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class."""
    return transformation_cache.get(self.categorical_column, state_manager)

  def _transform_feature(self, inputs):
    return _raise_shared_embedding_column_error()

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape([self.dimension])

  @property
  def _variable_shape(self):
    return _raise_shared_embedding_column_error()

  def get_embedding(self):
    shared_embedding_collection = ops.get_collection(self.embedding_name)
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError(
            'Collection {} can only contain one variable. '
            'Suggested fix A: Choose a unique name for this collection. '
            'Suggested fix B: Do not add any variables to this collection. '
            'The feature_column library already adds a variable under the '
            'hood.'.format(shared_embedding_collection))
      embedding_weights = shared_embedding_collection[0]
      return embedding_weights
    else:
      raise ValueError("Embedding not created yet.")

  def create_state(self, state_manager):
    """Creates the embedding lookup variable."""
    shared_embedding_collection = ops.get_collection(self.embedding_name)
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError(
            'Collection {} can only contain one variable. '
            'Suggested fix A: Choose a unique name for this collection. '
            'Suggested fix B: Do not add any variables to this collection. '
            'The feature_column library already adds a variable under the '
            'hood.'.format(shared_embedding_collection))
    else:
      num_buckets = getattr(self.categorical_column, 'num_buckets',
                            self.categorical_column._num_buckets)  # pylint: disable=protected-access
      embedding_shape = (num_buckets, self.dimension)
      state_manager.create_variable(
          self,
          name='embedding_weights',
          shape=embedding_shape,
          dtype=dtypes.float32,
          trainable=self.trainable,
          use_resource=True,
          initializer=self.initializer)
      embedding_weights = state_manager.get_variable(
          self, name='embedding_weights')
      ops.add_to_collection(self.embedding_name,
                            embedding_weights)

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    if self.ckpt_to_load_from is not None:
      to_restore = embedding_weights
      if isinstance(to_restore, variables.PartitionedVariable):
        to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
      checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {
          self.tensor_name_in_ckpt: to_restore
      })

    # Return embedding lookup result.
    return embedding_ops.safe_embedding_lookup_multi_dim(
        embedding_weights=embedding_weights,
        sparse_ids=sparse_ids,
        sparse_weights=sparse_weights,
        combiners=[self.combiner, self.hash_combiner],
        name='%s_weights' % self.name,
        max_norm=self.max_norm,
        weight_axis=-2)

  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    """Private method that follows the signature of get_dense_tensor."""
    embedding_weights = self.get_embedding()
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns the embedding lookup result."""
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column_v2(
          self, transformation_cache, state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return _raise_shared_embedding_column_error()

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    dense_tensor = self._get_dense_tensor_internal(transformation_cache,
                                                   state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    return _raise_shared_embedding_column_error()

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    raise NotImplementedError()

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    raise NotImplementedError()


_global_coalesced_scopes = []

@tf_export('feature_column.coalesced_embedding_scope')
@contextlib.contextmanager
def coalesced_embedding_scope(name=None, num_partitions=None):
  global _global_coalesced_scopes
  scope = CoalescedEmbeddingScope(name, num_partitions)
  _global_coalesced_scopes.append(scope)
  yield scope
  scope = _global_coalesced_scopes.pop()
  scope.build()

def current_coalesced_scope():
  global _global_coalesced_scopes
  return None if len(_global_coalesced_scopes) == 0 else \
         _global_coalesced_scopes[-1]

@tf_export('feature_column.CoalescedEmbeddingScope')
class CoalescedEmbeddingScope(coalesced_utils.CoalescedScopeBase):
  def __init__(self, name=None, num_partitions=None):
    if name is None:
      name = 'CoalescedEmbedding'
    self._num_partitions = num_partitions
    super(CoalescedEmbeddingScope, self).__init__(name)

  def allowed_column_types(self):
    return CoalescedEmbeddingColumn._COALESCING_TYPES

  def build(self):
    if self._built:
      return
    cluster = collections.defaultdict(list)
    for name, column in self._columns.items():
      h = coalesced_utils.make_cluster_signature(column)
      cluster[h].append((name, column))

    for h, names_and_columns in cluster.items():
      names_and_columns.sort(key=lambda x: x[1].name)

    for h, names_and_columns in cluster.items():
      names, columns = zip(*names_and_columns)
      coalesced_name = self.get_name()
      coalesced_column = CoalescedEmbeddingColumn(
          columns, coalesced_name, self._num_partitions)
      for name in names:
        self._coalesced_map[name] = coalesced_column
    self._built = True

class CoalescedEmbeddingColumn(object):
  """Coalescing _EmbeddingColumns into one according to signature.

  Args:

  Raises:
  """

  _COALESCING_TYPES = (
      EmbeddingColumn,
      SharedEmbeddingColumn,
      SharedEmbeddingColumnV2,
      SharedMultiHashEmbeddingColumn,
  )

  def __init__(self, columns, name, num_partitions=None):
    for i, c in enumerate(columns):
      if not isinstance(c, CoalescedEmbeddingColumn._COALESCING_TYPES):
        raise ValueError('columns must be a list of EmbeddingColumns, ',
                         'Given {} at index {}'.format(c, i))

    if len(columns) == 0:
      raise ValueError('columns cannot be empty')

    coalesced_utils.check_coalesced_columns_compatible(columns)

    for c in columns:
      if c not in coalesced_utils.get_embedding_signature():
        raise ValueError('signature not found for column: {}'.format(c))

    if num_partitions is not None:
      self._partitioner = \
          partitioned_variables.fixed_size_partitioner(num_partitions)
    else:
      self._partitioner = variable_scope.get_variable_scope().partitioner
    if self._partitioner is None:
      logging.log(logging.WARN, 'No partitioner found in outer variable'
                  ' scopes, use default: fixed_size_partitioner(1)')
      self._partitioner = partitioned_variables.fixed_size_partitioner(1)

    self._columns = columns
    self._runtime_columns = collections.defaultdict(list)
    self._runtime_col_multi_hash = {}
    self.build_runtime_columns(columns)
    self._name = name
    self._default_attr = coalesced_utils.get_embedding_signature()[columns[0]]

    self._unique_columns, self._indices_map = \
        coalesced_utils.deduplicate_shared_embedding(self._columns)
    save_slice_infos, tensor_slices, total_size = \
        coalesced_utils.build_slice_info(self._unique_columns, self._partitioner)
    self._save_slice_infos = save_slice_infos
    self._tensor_slices = tensor_slices

    self._local_offsets = []
    for save_slice_list in save_slice_infos:
      offset = 0
      offset_list = []
      for save_slice in save_slice_list:
        offset_list.append(offset)
        offset += save_slice.var_shape[0]
      self._local_offsets.append(offset_list)

    self._global_offsets = [[0 for j in range(len(tensor_slices))]
                               for i in range(len(save_slice_infos))]
    offset = 0
    for i in range(len(tensor_slices)):
      for j in range(len(save_slice_infos)):
        self._global_offsets[j][i] = offset
        ts = tensor_slices[i][j]
        offset += ts[0].stop - ts[0].start

    self._has_replace_var_name = False
    self._var_name_replace_maps = []
    self._total_bucket_size = total_size

  def _get_attr_from_signature(self, sig, attr):
    sig = json.loads(sig)
    return sig.get(attr)

  def build_runtime_columns(self, columns):
    for i, column in enumerate(columns):
      h = coalesced_utils._make_runtime_signature(column)
      self._runtime_columns[h].append((i, column))
      self._runtime_col_multi_hash[h] = self._get_attr_from_signature(h, 'hash_combiner')

  @property
  def name(self):
    return self._name

  @property
  def columns(self):
    return self._columns

  def _get_unique_index(self, column):
    if hasattr(column, 'embedding_name'):
      name = column.embedding_name
    else:
      name = column.name
    if name not in self._indices_map:
      raise ValueError('column {} not coalesced'.format(name))
    return self._indices_map[name]

  def encode(self, data, index):
    if not isinstance(data, sparse_tensor_lib.SparseTensor):
      raise ValueError('data should be a SparseTensor, Given {}'.format(data))
    values = gen_feature_column_ops.coalesced_bucketized_embedding_encode(
        math_ops.cast(data.values, dtype=dtypes.int64),
        self._local_offsets[index],
        self._global_offsets[index])
    return sparse_tensor_lib.SparseTensor(indices=data.indices,
                                          values=values,
                                          dense_shape=data.dense_shape)

  def make_sparse_inputs(self, transformation_cache, state_manager):
    result_list = []
    for h, runtime_columns in self._runtime_columns.items():
      ids_list = []
      weights_list = []
      weight_type = None
      for c in runtime_columns:
        sparse_tensors = c[1].categorical_column.get_sparse_tensors(
            transformation_cache, state_manager)
        ids, weights = fc_utils.parse_sparse_data(sparse_tensors)
        if ids is None:
          raise ValueError('sparse ids cannot be None')
        index = self._get_unique_index(c[1])
        ids_list.append(self.encode(ids, index))
        weights_list.append(weights)
        if weights is not None:
          if weight_type is None:
            weight_type = weights.dtype
          elif weight_type != weights.dtype:
            raise ValueError('all weights should have same dtype, but got '
                             '{} and {}'.format(weight_type, weights.dtype))
      if weight_type is None:
        weight_type = dtypes.float32
      format_rank = 3 if self._runtime_col_multi_hash[h] else 2
      result = coalesced_utils.coalesce_sparse_data(ids_list, weights_list, weight_type, format_rank=format_rank)
      result_list.append(result)
    return result_list


  def _check_weight_slice_compatible(self, weight):
    if isinstance(weight, variables.PartitionedVariable):
      parts = list(weight)
      if len(parts) != len(self._tensor_slices):
        raise ValueError('Variable parts num not equal to tensor slices num')
      weight_slice_offset = []
      offset = 0
      for part in parts:
        weight_slice_offset.append(offset)
        offset += int(part.shape[0])
      for i in range(len(parts)):
        if weight_slice_offset[i] != self._global_offsets[0][i]:
          raise ValueError('Variable parts not equal to tensor slices')
    else:
      if len(self._tensor_slices) != 1:
        raise ValueError('Coalesced weights is not partitioned, but '
                         'tensor slices has {} parts'.format(
                             len(self._tensor_slices)))

  def _get_embedding_name(self):
    if not hasattr(self, '_embedding_weights'):
      raise ValueError('get_embedding_name function should be called after create embeddings.')
    if isinstance(self._embedding_weights, variables.PartitionedVariable):
      name = list(self._embedding_weights)[0].name
      offset = len(name.split('/')[-1]) + 1
    else:
      name = self._embedding_weights.name
      offset = len(name.split(':')[-1]) + 1
    name = name[:-offset]
    return name

  def _generate_shared_column_replace_map(self):
    for i, column in enumerate(self._unique_columns):
      if isinstance(column, SharedEmbeddingColumn):
        var_scope_name = variable_scope.get_variable_scope().name
        var_name = column.embedding_name
        shared_emb_name = var_scope_name + "/" + var_name if len(var_scope_name) > 0 else var_name
        self._var_name_replace_maps.append(
            (self._get_embedding_name(), shared_emb_name))
      else:
        self._var_name_replace_maps.append((self._name, column.embedding_name))

  def _replace_slot_name(self, embedding_weights):
    if self._has_replace_var_name:
      return
    self._generate_shared_column_replace_map() 
    if isinstance(embedding_weights, variables.PartitionedVariable):
      # set save_slice_info
      parts = list(embedding_weights)
      for i, part in enumerate(parts):
        tensor_slices = self._tensor_slices[i]
        save_slices = [slice_list[i] for slice_list in self._save_slice_infos]
        raw_info = part._save_slice_info
        for j, save_slice in enumerate(save_slices):
          save_slice.full_name = raw_info.full_name.replace(
              self._var_name_replace_maps[j][0], self._var_name_replace_maps[j][1])
        save_info = coalesced_utils.CoalescedSaveSliceInfo(
            raw_info.full_name, raw_info.full_shape, raw_info.var_offset,
            raw_info.var_shape, raw_info.var_full_name, save_slices,
            tensor_slices)
        part._set_save_slice_info(save_info)
    else:
      # TODO
      pass
    self._has_replace_var_name = True

  def get_or_create_embedding_weights(self):
    if not hasattr(self, '_embedding_weights'):
      dimension = self._default_attr.dimension
      dtype = self._default_attr.dtype
      initializer = self._default_attr.initializer
      trainable = self._default_attr.trainable
  
      embedding_weights = variable_scope.get_variable(
          "embedding_weights",
          shape=[self._total_bucket_size, dimension],
          dtype=dtype,
          initializer=initializer,
          trainable=trainable,
          partitioner=self._partitioner)
      # check slice info match with global offsets
      self._check_weight_slice_compatible(embedding_weights)
      self._embedding_weights = embedding_weights
    return self._embedding_weights

  def _embedding_lookup_sparse(self, embedding_weights, ids, weights, combiner, hash_combiner=''):
    if not hash_combiner:
      return embedding_ops.safe_embedding_lookup_sparse(
        embedding_weights=embedding_weights,
        sparse_ids=ids,
        sparse_weights=weights,
        combiner=combiner)
    else:
      return embedding_ops.safe_embedding_lookup_multi_dim(
          embedding_weights=embedding_weights,
          sparse_ids=ids,
          sparse_weights=weights,
          combiners=[combiner, hash_combiner],
          weight_axis=-2)

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return _raise_shared_embedding_column_error()

  def get_dense_tensor(self, transformation_cache, state_manager):
    embedding_weights = self.get_or_create_embedding_weights()
    lookup_input_list = self.make_sparse_inputs(
        transformation_cache, state_manager)
    self._replace_slot_name(embedding_weights)
    # Return embedding lookup result.
    embedding_outputs = []
    for lookup_input, runtime_pair in zip(*(lookup_input_list,
                                       self._runtime_columns.items())):
      hash_combiner = self._runtime_col_multi_hash[runtime_pair[0]]
      cids_and_columns = runtime_pair[1]
      cids, columns = zip(*cids_and_columns)
      embeddings = self._embedding_lookup_sparse(
        embedding_weights,
        lookup_input[0],
        lookup_input[1],
        coalesced_utils.get_signature_attributes(columns[0]).combiner,
        hash_combiner)
      values = array_ops.split(embeddings, lookup_input[2])
      results = []
      for value, origin_shape, col in zip(values, lookup_input[3], columns):
        origin_rank = array_ops.size(origin_shape)
        if coalesced_utils.get_signature_attributes(col).combiner == 'tile':
          real_dim = array_ops.gather(origin_shape, origin_rank - 1) * self._default_attr.dimension
          value = array_ops.slice(value,
                                 [0, 0],
                                 [-1, real_dim])
        value = array_ops.reshape(
            value,
            array_ops.concat([
                array_ops.slice(origin_shape, [0], [origin_rank - 1]),
                array_ops.slice(array_ops.shape(value), [1], [-1])
            ], 0))
        results.append(value)
      embedding_outputs.extend(zip(cids, results))
    return list(zip(*sorted(embedding_outputs)))[1]


def _check_shape(shape, key):
  """Returns shape if it's valid, raises error otherwise."""
  assert shape is not None
  if not nest.is_sequence(shape):
    shape = [shape]
  shape = tuple(shape)
  for dimension in shape:
    if not isinstance(dimension, int):
      raise TypeError('shape dimensions must be integer. '
                      'shape: {}, key: {}'.format(shape, key))
    if dimension < 1:
      raise ValueError('shape dimensions must be greater than 0. '
                       'shape: {}, key: {}'.format(shape, key))
  return shape


class HashedCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('HashedCategoricalColumn',
                           ('key', 'hash_bucket_size', 'dtype'))):
  """see `categorical_column_with_hash_bucket`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    """Hashes the values in the feature_column."""
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseColumn input must be a SparseTensor.')

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    if self.dtype == dtypes.string:
      sparse_values = input_tensor.values
    else:
      sparse_values = string_ops.as_string(input_tensor.values)

    sparse_id_values = string_ops.string_to_hash_bucket_fast(
        sparse_values, self.hash_bucket_size, name='lookup')
    return sparse_tensor_lib.SparseTensor(
        input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    """Hashes the values in the feature_column."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)

class EmbeddingCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('HashedCategoricalColumn',
                           ('key', 'dtype', 'partition_num', 'ev_option'))):

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    """Hashes the values in the feature_column."""
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseColumn input must be a SparseTensor.')

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    if self.dtype == dtypes.string:
        max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
        sparse_id_values = string_ops.string_to_hash_bucket_fast(
                               input_tensor.values, max_value)
    else:
        sparse_id_values = input_tensor.values
    return sparse_tensor_lib.SparseTensor(
        input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    """Hashes the values in the feature_column."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return 1

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return 1

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class AdaptiveEmbeddingCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('AdaptiveEmbeddingCategoricalColumn',
                           ('key', 'hash_bucket_size', 'dtype',
                            'partition_num', 'ev_option',
                            #'adaptive_mask_tensor', 'hash_ev_ids'))):
                            #'hash_ev_ids'))):
                            ))):

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def set_adaptive_mask_tensor(self, adaptive_mask_tensor):
    self.adaptive_mask_tensor = adaptive_mask_tensor

  def _transform_input_tensor(self, input_tensor):
    flat_ids = array_ops.reshape(input_tensor.values, [-1])
    original_indices = math_ops.range(array_ops.size(flat_ids))
    parts = data_flow_ops.dynamic_partition(original_indices, self.adaptive_mask_tensor, 2)
    spids_part = data_flow_ops.dynamic_partition(flat_ids, self.adaptive_mask_tensor, 2)

    if self.dtype == dtypes.string:
      hash_ids = string_ops.string_to_hash_bucket_fast(
        spids_part[0], self.hash_bucket_size, name="lookup_hash")
      ev_ids = string_ops.string_to_hash_bucket_fast(
        spids_part[1], np.iinfo(dtypes.int64.as_numpy_dtype).max)
      self.hash_ev_ids = string_ops.string_to_hash_bucket_fast(
        spids_part[1], self.hash_bucket_size, name="lookup_hash_ev")
      sparse_id_values = data_flow_ops.dynamic_stitch(parts, [hash_ids, ev_ids])
    else:
      hash_ids = string_ops.string_to_hash_bucket_fast(
              string_ops.as_string(spids_part[0]), self.hash_bucket_size, name="lookup_hash")
      self.hash_ev_ids = string_ops.string_to_hash_bucket_fast(
              string_ops.as_string(spids_part[1]), self.hash_bucket_size, name="lookup_hash_ev")
      ev_ids = spids_part[1]
      sparse_id_values = data_flow_ops.dynamic_stitch(parts, [hash_ids, ev_ids])
    return sparse_tensor_lib.SparseTensor(input_tensor.indices, sparse_id_values,
                                         input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    """Hashes the values in the feature_column."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class HashOnlyCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('HashOnlyCategoricalColumn',
                           ('key', 'hash_type', 'allow_neg', 'dtype'))):
  """see `categorical_column_with_hash`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    """Hashes the values in the feature_column."""
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseColumn input must be a SparseTensor.')

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    if self.dtype == dtypes.string:
      sparse_values = input_tensor.values
    else:
      sparse_values = string_ops.as_string(input_tensor.values)
    sparse_id_values = string_ops.string_to_hash(sparse_values,
                                                 hash_type=self.hash_type,
                                                 allow_neg=self.allow_neg)
    return sparse_tensor_lib.SparseTensor(
        input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    """Hashes the values in the feature_column."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    raise ValueError("categorical_column_with_hash does not has attr `_num_buckets`"
        " if you want to look up embedding with embedding column, "
        "please use hashtable feature column.")

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class MultiHashedCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('MultiHashedCategoricalColumn',
                           ('key', 'hash_bucket_size', 'hash_types', 'dtype'))):
  """see `categorical_column_with_multi_hash_bucket`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    """Hashes the values in the feature_column."""
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseColumn input must be a SparseTensor.')

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    if self.dtype == dtypes.string:
      sparse_values = input_tensor.values
    else:
      sparse_values = string_ops.as_string(input_tensor.values)
    sparse_values = string_ops.string_to_hash(sparse_values,
                                              hash_type='farm',
                                              allow_neg=False,
                                              num_buckets=self.hash_bucket_size)
    sparse_values = string_ops.as_string(sparse_values)
    hash_sparse_tensors = []
    new_dense_shape = array_ops.concat([input_tensor.dense_shape, [1]], axis=0)
    for i, hash_type in enumerate(self.hash_types):
      sparse_bucket_values = string_ops.string_to_hash(sparse_values,
                                                       hash_type=hash_type,
                                                       allow_neg=False,
                                                       num_buckets=self.hash_bucket_size)
      new_sp = sparse_tensor_lib.SparseTensor(input_tensor.indices, sparse_bucket_values, input_tensor.dense_shape)
      hash_sparse_tensors.append(sparse_ops.sparse_reshape(new_sp, shape=new_dense_shape))
    output_tensor = sparse_ops.sparse_concat(sp_inputs=hash_sparse_tensors, axis=-1, name="multihash_concat")
    return output_tensor

  def transform_feature(self, transformation_cache, state_manager):
    """Hashes the values in the feature_column."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class VocabularyFileCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('VocabularyFileCategoricalColumn',
                           ('key', 'vocabulary_file', 'vocabulary_size',
                            'num_oov_buckets', 'dtype', 'default_value'))):
  """See `categorical_column_with_vocabulary_file`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor, state_manager=None):
    """Creates a lookup table for the vocabulary."""
    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    key_dtype = self.dtype
    if input_tensor.dtype.is_integer:
      # `index_table_from_file` requires 64-bit integer keys.
      key_dtype = dtypes.int64
      input_tensor = math_ops.cast(input_tensor, dtypes.int64)

    name = '{}_lookup'.format(self.key)
    table = lookup_ops.index_table_from_file(
        vocabulary_file=self.vocabulary_file,
        num_oov_buckets=self.num_oov_buckets,
        vocab_size=self.vocabulary_size,
        default_value=self.default_value,
        key_dtype=key_dtype,
        name=name)
    if state_manager is not None:
      state_manager.add_resource(self, name, table)
    return table.lookup(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    """Creates a lookup table for the vocabulary."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.vocabulary_size + self.num_oov_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class VocabularyListCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'VocabularyListCategoricalColumn',
        ('key', 'vocabulary_list', 'dtype', 'default_value', 'num_oov_buckets'))
):
  """See `categorical_column_with_vocabulary_list`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor, state_manager=None):
    """Creates a lookup table for the vocabulary list."""
    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    key_dtype = self.dtype
    if input_tensor.dtype.is_integer:
      # `index_table_from_tensor` requires 64-bit integer keys.
      key_dtype = dtypes.int64
      input_tensor = math_ops.cast(input_tensor, dtypes.int64)

    name = '{}_lookup'.format(self.key)
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=tuple(self.vocabulary_list),
        default_value=self.default_value,
        num_oov_buckets=self.num_oov_buckets,
        dtype=key_dtype,
        name=name)
    if state_manager is not None:
      state_manager.add_resource(self, name, table)
    return table.lookup(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    """Creates a lookup table for the vocabulary list."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return len(self.vocabulary_list) + self.num_oov_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class IdentityCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('IdentityCategoricalColumn',
                           ('key', 'number_buckets', 'default_value'))):

  """See `categorical_column_with_identity`."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(dtypes.int64)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    """Returns a SparseTensor with identity values."""
    if not input_tensor.dtype.is_integer:
      raise ValueError(
          'Invalid input, not integer. key: {} dtype: {}'.format(
              self.key, input_tensor.dtype))

    values = math_ops.cast(input_tensor.values, dtypes.int64, name='values')
    num_buckets = math_ops.cast(
        self.num_buckets, dtypes.int64, name='num_buckets')
    zero = math_ops.cast(0, dtypes.int64, name='zero')
    if self.default_value is None:
      # Fail if values are out-of-range.
      assert_less = check_ops.assert_less(
          values,
          num_buckets,
          data=(values, num_buckets),
          message='Bucket index for categorical column '
          '"{}" exceeds number of buckets'.format(self.name),
          name='assert_less_than_num_buckets')
      assert_greater = check_ops.assert_greater_equal(
          values,
          zero,
          data=(values,),
          message='Negative bucket index for categorical column "{}"'.format(
              self.name),
          name='assert_greater_or_equal_0')
      with ops.control_dependencies((assert_less, assert_greater)):
        values = array_ops.identity(values)
    else:
      # Assign default for out-of-range values.
      values = array_ops.where_v2(
          math_ops.logical_or(
              values < zero, values >= num_buckets, name='out_of_range'),
          array_ops.fill(
              dims=array_ops.shape(values),
              value=math_ops.cast(self.default_value, dtypes.int64),
              name='default_values'), values)

    return sparse_tensor_lib.SparseTensor(
        indices=input_tensor.indices,
        values=values,
        dense_shape=input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    """Returns a SparseTensor with identity values."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.number_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    return dict(zip(self._fields, self))

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    return cls(**kwargs)


class WeightedCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'WeightedCategoricalColumn',
        ('categorical_column', 'weight_feature_key', 'dtype'))):
  """See `weighted_categorical_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_weighted_by_{}'.format(
        self.categorical_column.name, self.weight_feature_key)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    config = self.categorical_column.parse_example_spec
    if self.weight_feature_key in config:
      raise ValueError('Parse config {} already exists for {}.'.format(
          config[self.weight_feature_key], self.weight_feature_key))
    config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
    return config

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    config = self.categorical_column._parse_example_spec  # pylint: disable=protected-access
    if self.weight_feature_key in config:
      raise ValueError('Parse config {} already exists for {}.'.format(
          config[self.weight_feature_key], self.weight_feature_key))
    config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
    return config

  @property
  def num_buckets(self):
    """See `DenseColumn` base class."""
    return self.categorical_column.num_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.categorical_column._num_buckets  # pylint: disable=protected-access

  def _transform_weight_tensor(self, weight_tensor):
    if weight_tensor is None:
      raise ValueError('Missing weights {}.'.format(self.weight_feature_key))
    weight_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        weight_tensor)
    if self.dtype != weight_tensor.dtype.base_dtype:
      raise ValueError('Bad dtype, expected {}, but got {}.'.format(
          self.dtype, weight_tensor.dtype))
    if not isinstance(weight_tensor, sparse_tensor_lib.SparseTensor):
      # The weight tensor can be a regular Tensor. In this case, sparsify it.
      weight_tensor = _to_sparse_input_and_drop_ignore_values(
          weight_tensor, ignore_value=0.0)
    if not weight_tensor.dtype.is_floating:
      weight_tensor = math_ops.cast(weight_tensor, dtypes.float32)
    return weight_tensor

  def transform_feature(self, transformation_cache, state_manager):
    """Applies weights to tensor generated from `categorical_column`'."""
    weight_tensor = transformation_cache.get(self.weight_feature_key,
                                             state_manager)
    weight_tensor = self._transform_weight_tensor(weight_tensor)
    return (transformation_cache.get(self.categorical_column, state_manager),
            weight_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    """Applies weights to tensor generated from `categorical_column`'."""
    weight_tensor = inputs.get(self.weight_feature_key)
    weight_tensor = self._transform_weight_tensor(weight_tensor)
    return (inputs.get(self.categorical_column), weight_tensor)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    tensors = transformation_cache.get(self, state_manager)
    return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    tensors = inputs.get(self)
    return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column, self.weight_feature_key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class WeightedMultiHashedCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'WeightedMultiHashedCategoricalColumn',
        ('categorical_column', 'weight_feature_key', 'dtype'))):
  """See `weighted_multi_hash_categorical_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_weighted_by_{}'.format(
        self.categorical_column.name, self.weight_feature_key)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    config = self.categorical_column.parse_example_spec
    if self.weight_feature_key in config:
      raise ValueError('Parse config {} already exists for {}.'.format(
          config[self.weight_feature_key], self.weight_feature_key))
    config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
    config[self.weight_feature_key].weighted_key = True
    return config

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    config = self.categorical_column._parse_example_spec  # pylint: disable=protected-access
    if self.weight_feature_key in config:
      raise ValueError('Parse config {} already exists for {}.'.format(
          config[self.weight_feature_key], self.weight_feature_key))
    config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
    config[self.weight_feature_key].weighted_key = True
    return config

  @property
  def num_buckets(self):
    """See `DenseColumn` base class."""
    return self.categorical_column.num_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.categorical_column._num_buckets  # pylint: disable=protected-access

  def _transform_weight_tensor(self, weight_tensor):
    if weight_tensor is None:
      raise ValueError('Missing weights {}.'.format(self.weight_feature_key))
    weight_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        weight_tensor)
    if self.dtype != weight_tensor.dtype.base_dtype:
      raise ValueError('Bad dtype, expected {}, but got {}.'.format(
          self.dtype, weight_tensor.dtype))
    if not isinstance(weight_tensor, sparse_tensor_lib.SparseTensor):
      # The weight tensor can be a regular Tensor. In this case, sparsify it.
      weight_tensor = _to_sparse_input_and_drop_ignore_values(
          weight_tensor, ignore_value=0.0)
    if not weight_tensor.dtype.is_floating:
      weight_tensor = math_ops.to_float(weight_tensor)
    new_dense_shape = array_ops.concat([weight_tensor.dense_shape, [1]], axis=0)
    weight_tensor = sparse_ops.sparse_reshape(weight_tensor, shape=new_dense_shape)
    weight_tensors = [weight_tensor] * len(self.categorical_column.hash_types)
    weight_tensor = sparse_ops.sparse_concat(sp_inputs=weight_tensors, axis=-1, name="multihash_weighted_concat")
    return weight_tensor

  def transform_feature(self, transformation_cache, state_manager):
    """Applies weights to tensor generated from `categorical_column`'."""
    weight_tensor = transformation_cache.get(self.weight_feature_key,
                                             state_manager)
    weight_tensor = self._transform_weight_tensor(weight_tensor)
    return (transformation_cache.get(self.categorical_column, state_manager),
            weight_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    """Applies weights to tensor generated from `categorical_column`'."""
    weight_tensor = inputs.get(self.weight_feature_key)
    weight_tensor = self._transform_weight_tensor(weight_tensor)
    return (inputs.get(self.categorical_column), weight_tensor)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    tensors = transformation_cache.get(self, state_manager)
    return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    tensors = inputs.get(self)
    return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column, self.weight_feature_key]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


class CrossedColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('CrossedColumn',
                           ('keys', 'hash_bucket_size', 'hash_key'))):
  """See `crossed_column`."""

  @property
  def _is_v2_column(self):
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, six.string_types):
        continue
      if not isinstance(key, FeatureColumn):
        return False
      if not key._is_v2_column:  # pylint: disable=protected-access
        return False
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    feature_names = []
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, (FeatureColumn, fc_old._FeatureColumn)):  # pylint: disable=protected-access
        feature_names.append(key.name)
      else:  # key must be a string
        feature_names.append(key)
    return '_X_'.join(sorted(feature_names))

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    config = {}
    for key in self.keys:
      if isinstance(key, FeatureColumn):
        config.update(key.parse_example_spec)
      elif isinstance(key, fc_old._FeatureColumn):  # pylint: disable=protected-access
        config.update(key._parse_example_spec)  # pylint: disable=protected-access
      else:  # key must be a string
        config.update({key: parsing_ops.VarLenFeature(dtypes.string)})
    return config

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.parse_example_spec

  def transform_feature(self, transformation_cache, state_manager):
    """Generates a hashed sparse cross from the input tensors."""
    feature_tensors = []
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, six.string_types):
        feature_tensors.append(transformation_cache.get(key, state_manager))
      elif isinstance(key, (fc_old._CategoricalColumn, CategoricalColumn)):  # pylint: disable=protected-access
        ids_and_weights = key.get_sparse_tensors(transformation_cache,
                                                 state_manager)
        if ids_and_weights.weight_tensor is not None:
          raise ValueError(
              'crossed_column does not support weight_tensor, but the given '
              'column populates weight_tensor. '
              'Given column: {}'.format(key.name))
        feature_tensors.append(ids_and_weights.id_tensor)
      else:
        raise ValueError('Unsupported column type. Given: {}'.format(key))
    return sparse_ops.sparse_cross_hashed(
        inputs=feature_tensors,
        num_buckets=self.hash_bucket_size,
        hash_key=self.hash_key)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    """Generates a hashed sparse cross from the input tensors."""
    feature_tensors = []
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, six.string_types):
        feature_tensors.append(inputs.get(key))
      elif isinstance(key, (CategoricalColumn, fc_old._CategoricalColumn)):  # pylint: disable=protected-access
        ids_and_weights = key._get_sparse_tensors(inputs)  # pylint: disable=protected-access
        if ids_and_weights.weight_tensor is not None:
          raise ValueError(
              'crossed_column does not support weight_tensor, but the given '
              'column populates weight_tensor. '
              'Given column: {}'.format(key.name))
        feature_tensors.append(ids_and_weights.id_tensor)
      else:
        raise ValueError('Unsupported column type. Given: {}'.format(key))
    return sparse_ops.sparse_cross_hashed(
        inputs=feature_tensors,
        num_buckets=self.hash_bucket_size,
        hash_key=self.hash_key)

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    """See `CategoricalColumn` base class."""
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return list(self.keys)

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['keys'] = tuple([serialize_feature_column(fc) for fc in self.keys])
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['keys'] = tuple([
        deserialize_feature_column(c, custom_objects, columns_by_name)
        for c in config['keys']
    ])
    return cls(**kwargs)


def _collect_leaf_level_keys(cross):
  """Collects base keys by expanding all nested crosses.

  Args:
    cross: A `CrossedColumn`.

  Returns:
    A list of strings or `CategoricalColumn` instances.
  """
  leaf_level_keys = []
  for k in cross.keys:
    if isinstance(k, CrossedColumn):
      leaf_level_keys.extend(_collect_leaf_level_keys(k))
    else:
      leaf_level_keys.append(k)
  return leaf_level_keys


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


class IndicatorColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple('IndicatorColumn', ('categorical_column'))):
  """Represents a one-hot column for use in deep networks.

  Args:
    categorical_column: A `CategoricalColumn` which is created by
      `categorical_column_with_*` function.
  """

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_indicator'.format(self.categorical_column.name)

  def _transform_id_weight_pair(self, id_weight_pair):
    id_tensor = id_weight_pair.id_tensor
    weight_tensor = id_weight_pair.weight_tensor

    # If the underlying column is weighted, return the input as a dense tensor.
    if weight_tensor is not None:
      weighted_column = sparse_ops.sparse_merge(
          sp_ids=id_tensor,
          sp_values=weight_tensor,
          vocab_size=int(self._variable_shape[-1]))
      # Remove (?, -1) index.
      weighted_column = sparse_ops.sparse_slice(weighted_column, [0, 0],
                                                weighted_column.dense_shape)
      # Use scatter_nd to merge duplicated indices if existed,
      # instead of sparse_tensor_to_dense.
      return array_ops.scatter_nd(weighted_column.indices,
                                  weighted_column.values,
                                  weighted_column.dense_shape)

    dense_id_tensor = sparse_ops.sparse_tensor_to_dense(
        id_tensor, default_value=-1)

    # One hot must be float for tf.concat reasons since all other inputs to
    # input_layer are float32.
    one_hot_id_tensor = array_ops.one_hot(
        dense_id_tensor,
        depth=self._variable_shape[-1],
        on_value=1.0,
        off_value=0.0)

    # Reduce to get a multi-hot per example.
    return math_ops.reduce_sum(one_hot_id_tensor, axis=[-2])

  def transform_feature(self, transformation_cache, state_manager):
    """Returns dense `Tensor` representing feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Transformed feature `Tensor`.

    Raises:
      ValueError: if input rank is not known at graph building time.
    """
    id_weight_pair = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._transform_id_weight_pair(id_weight_pair)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    id_weight_pair = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    return self._transform_id_weight_pair(id_weight_pair)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  @property
  def variable_shape(self):
    """Returns a `TensorShape` representing the shape of the dense `Tensor`."""
    if isinstance(self.categorical_column, FeatureColumn):
      return tensor_shape.TensorShape([1, self.categorical_column.num_buckets])
    else:
      return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])  # pylint: disable=protected-access

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])  # pylint: disable=protected-access

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns dense `Tensor` representing feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.

    Raises:
      ValueError: If `categorical_column` is a `SequenceCategoricalColumn`.
    """
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by transform_feature.
    return transformation_cache.get(self, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    if isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must not be of type _SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by transform_feature.
    return inputs.get(self)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by transform_feature.
    dense_tensor = transformation_cache.get(self, state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    # Do nothing with weight_collections and trainable since no variables are
    # created in this function.
    del weight_collections
    del trainable
    if not isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must be of type _SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by _transform_feature.
    dense_tensor = inputs.get(self)
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


def _verify_static_batch_size_equality(tensors, columns):
  """Verify equality between static batch sizes.

  Args:
    tensors: iterable of input tensors.
    columns: Corresponding feature columns.

  Raises:
    ValueError: in case of mismatched batch sizes.
  """
  # bath_size is a Dimension object.
  expected_batch_size = None
  for i in range(0, len(tensors)):
    batch_size = tensor_shape.Dimension(tensor_shape.dimension_value(
        tensors[i].shape[0]))
    if batch_size.value is not None:
      if expected_batch_size is None:
        bath_size_column_index = i
        expected_batch_size = batch_size
      elif not expected_batch_size.is_compatible_with(batch_size):
        raise ValueError(
            'Batch size (first dimension) of each feature must be same. '
            'Batch size of columns ({}, {}): ({}, {})'.format(
                columns[bath_size_column_index].name, columns[i].name,
                expected_batch_size, batch_size))


class SequenceCategoricalColumn(
    CategoricalColumn,
    fc_old._SequenceCategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('SequenceCategoricalColumn',
                           ('categorical_column'))):
  """Represents sequences of categorical data."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class."""
    return self.categorical_column.transform_feature(transformation_cache,
                                                     state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    return self.categorical_column._transform_feature(inputs)  # pylint: disable=protected-access

  @property
  def num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.categorical_column.num_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    return self.categorical_column._num_buckets  # pylint: disable=protected-access

  def _get_sparse_tensors_helper(self, sparse_tensors):
    id_tensor = sparse_tensors.id_tensor
    weight_tensor = sparse_tensors.weight_tensor
    # Expands third dimension, if necessary so that embeddings are not
    # combined during embedding lookup. If the tensor is already 3D, leave
    # as-is.
    shape = array_ops.shape(id_tensor)
    # Compute the third dimension explicitly instead of setting it to -1, as
    # that doesn't work for dynamically shaped tensors with 0-length at runtime.
    # This happens for empty sequences.
    target_shape = [shape[0], shape[1], math_ops.reduce_prod(shape[2:])]
    id_tensor = sparse_ops.sparse_reshape(id_tensor, target_shape)
    if weight_tensor is not None:
      weight_tensor = sparse_ops.sparse_reshape(weight_tensor, target_shape)
    return CategoricalColumn.IdWeightPair(id_tensor, weight_tensor)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    """Returns an IdWeightPair.

    `IdWeightPair` is a pair of `SparseTensor`s which represents ids and
    weights.

    `IdWeightPair.id_tensor` is typically a `batch_size` x `num_buckets`
    `SparseTensor` of `int64`. `IdWeightPair.weight_tensor` is either a
    `SparseTensor` of `float` or `None` to indicate all weights should be
    taken to be 1. If specified, `weight_tensor` must have exactly the same
    shape and indices as `sp_ids`. Expected `SparseTensor` is same as parsing
    output of a `VarLenFeature` which is a ragged matrix.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_sparse_tensors_helper(sparse_tensors)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    return self._get_sparse_tensors_helper(sparse_tensors)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def _get_config(self):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    return config

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


def _check_config_keys(config, expected_keys):
  """Checks that a config has all expected_keys."""
  if set(config.keys()) != set(expected_keys):
    raise ValueError('Invalid config: {}, expected keys: {}'.format(
        config, expected_keys))


def _standardize_and_copy_config(config):
  """Returns a shallow copy of config with lists turned to tuples.

  Keras serialization uses nest to listify everything.
  This causes problems with the NumericColumn shape, which becomes
  unhashable. We could try to solve this on the Keras side, but that
  would require lots of tracking to avoid changing existing behavior.
  Instead, we ensure here that we revive correctly.

  Args:
    config: dict that will be used to revive a Feature Column

  Returns:
    Shallow copy of config with lists turned to tuples.
  """
  kwargs = config.copy()
  for k, v in kwargs.items():
    if isinstance(v, list):
      kwargs[k] = tuple(v)

  return kwargs
