# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as backend
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils


class DeNormalization(base_preprocessing_layer.PreprocessingLayer):
    """
    Kears custom DeNormalization layer for data normalized to a standard Gaussian distribution.
    The function takes input data with mean ~ 0 and variance ~ 1, and de-normalize them with the provided values
    of mean and variance:

    x = z * sqrt(variance) + mean

    where z are the input (normalized) data and x are the (de-normalized) output data.

    Most of the methods in this class were taken from the Keras normalization layer on Github from the script: 
    Keras--> preprocessing --> normalization.py
    https://github.com/keras-team/keras/blob/master/keras/layers/preprocessing/normalization.py

    However, many methods of this class have not been tested.

    The call is what differs this class from the Normalization class:
    
    inputs * tf.maximum(tf.sqrt(self.variance), backend.epsilon()) + self.mean

    In case of variance approaching 0, the epsilon value is used instead.
    """
    def __init__(self, axis=-1, mean=None, variance=None, **kwargs):


        super().__init__(**kwargs)
        
        # Standardize axis to a tuple
        if axis is None:
            axis = ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)
        self.axis = axis

        # Set `mean` and `variance` if passed.
        if isinstance(mean, tf.Variable):
            raise ValueError(
                "Normalization does not support passing a Variable "
                "for the `mean` init arg."
            )
        if isinstance(variance, tf.Variable):
            raise ValueError(
                "Normalization does not support passing a Variable "
                "for the `variance` init arg."
            )
        if (mean is not None) != (variance is not None):
            raise ValueError(
                "When setting values directly, both `mean` and `variance` "
                "must be set. Got mean: {} and variance: {}".format(
                    mean, variance
                )
            )
        self.input_mean     = mean
        self.input_variance = variance
        
    def build(self, input_shape):
        super().build(input_shape)

        if isinstance(input_shape, (list, tuple)) and all(
            isinstance(shape, tf.TensorShape) for shape in input_shape
        ):
            raise ValueError(
                "Normalization only accepts a single input. If you are "
                "passing a python list or tuple as a single input, "
                "please convert to a numpy array or `tf.Tensor`."
            )

        input_shape = tf.TensorShape(input_shape).as_list()
        ndim = len(input_shape)

        if any(a < -ndim or a >= ndim for a in self.axis):
            raise ValueError(
                "All `axis` values must be in the range [-ndim, ndim). "
                "Found ndim: `{}`, axis: {}".format(ndim, self.axis)
            )

        # Axes to be kept, replacing negative values with positive equivalents.
        # Sorted to avoid transposing axes.
        self._keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        # All axes to be kept should have known shape.
        for d in self._keep_axis:
            if input_shape[d] is None:
                raise ValueError(
                    "All `axis` values to be kept must have known shape. "
                    "Got axis: {}, "
                    "input shape: {}, with unknown axis at index: {}".format(
                        self.axis, input_shape, d
                    )
                )
        # Axes to be reduced.
        self._reduce_axis = [d for d in range(ndim) if d not in self._keep_axis]
        # 1 if an axis should be reduced, 0 otherwise.
        self._reduce_axis_mask = [
            0 if d in self._keep_axis else 1 for d in range(ndim)
        ]
        # Broadcast any reduced axes.
        self._broadcast_shape = [
            input_shape[d] if d in self._keep_axis else 1 for d in range(ndim)
        ]
        mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)

        if self.input_mean is None:
            self.adapt_mean = self.add_weight(
                name="mean",
                shape=mean_and_var_shape,
                dtype=self.compute_dtype,
                initializer="zeros",
                trainable=False,
            )
            self.adapt_variance = self.add_weight(
                name="variance",
                shape=mean_and_var_shape,
                dtype=self.compute_dtype,
                initializer="ones",
                trainable=False,
            )
            self.count = self.add_weight(
                name="count",
                shape=(),
                dtype=tf.int64,
                initializer="zeros",
                trainable=False,
            )
            self.finalize_state()
        else:
            # In the no adapt case, make constant tensors for mean and variance
            # with proper broadcast shape for use during call.
            mean = self.input_mean * np.ones(mean_and_var_shape)
            variance = self.input_variance * np.ones(mean_and_var_shape)
            mean = tf.reshape(mean, self._broadcast_shape)
            variance = tf.reshape(variance, self._broadcast_shape)
            self.mean = tf.cast(mean, self.compute_dtype)
            self.variance = tf.cast(variance, self.compute_dtype)

    # We override this method solely to generate a docstring.
    def adapt(self, data, batch_size=None, steps=None):
        """Computes the mean and variance of values in a dataset.
        Calling `adapt()` on a `Normalization` layer is an alternative to
        passing in `mean` and `variance` arguments during layer construction. A
        `Normalization` layer should always either be adapted over a dataset or
        passed `mean` and `variance`.
        During `adapt()`, the layer will compute a `mean` and `variance`
        separately for each position in each axis specified by the `axis`
        argument. To calculate a single `mean` and `variance` over the input
        data, simply pass `axis=None`.
        In order to make `Normalization` efficient in any distribution context,
        the computed mean and variance are kept static with respect to any
        compiled `tf.Graph`s that call the layer. As a consequence, if the layer
        is adapted a second time, any models using the layer should be
        re-compiled. For more information see
        `tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt`.
        `adapt()` is meant only as a single machine utility to compute layer
        state.  To analyze a dataset that cannot fit on a single machine, see
        [Tensorflow Transform](
        https://www.tensorflow.org/tfx/transform/get_started)
        for a multi-machine, map-reduce solution.
        Arguments:
          data: The data to train on. It can be passed either as a
              `tf.data.Dataset`, or as a numpy array.
          batch_size: Integer or `None`.
              Number of samples per state update.
              If unspecified, `batch_size` will default to 32.
              Do not specify the `batch_size` if your data is in the
              form of datasets, generators, or `keras.utils.Sequence` instances
              (since they generate batches).
          steps: Integer or `None`.
              Total number of steps (batches of samples)
              When training with input tensors such as
              TensorFlow data tensors, the default `None` is equal to
              the number of samples in your dataset divided by
              the batch size, or 1 if that cannot be determined. If x is a
              `tf.data` dataset, and 'steps' is None, the epoch will run until
              the input dataset is exhausted. When passing an infinitely
              repeating dataset, you must specify the `steps` argument. This
              argument is not supported with array inputs.
        """
        super().adapt(data, batch_size=batch_size, steps=steps)

    # Use data to compute mean/var.
    def update_state(self, data):
        if self.input_mean is not None:
            raise ValueError(
                "Cannot `adapt` a Normalization layer that is initialized with "
                "static `mean` and `variance`, "
                "you passed mean {} and variance {}.".format(
                    self.input_mean, self.input_variance
                )
            )

        if not self.built:
            raise RuntimeError("`build` must be called before `update_state`.")

        data = self._standardize_inputs(data)
        data = tf.cast(data, self.adapt_mean.dtype)
        batch_mean, batch_variance = tf.nn.moments(data, axes=self._reduce_axis)
        batch_shape = tf.shape(data, out_type=self.count.dtype)
        if self._reduce_axis:
            batch_reduce_shape = tf.gather(batch_shape, self._reduce_axis)
            batch_count = tf.reduce_prod(batch_reduce_shape)
        else:
            batch_count = 1

        total_count = batch_count + self.count
        batch_weight = tf.cast(batch_count, dtype=self.compute_dtype) / tf.cast(
            total_count, dtype=self.compute_dtype
        )
        existing_weight = 1.0 - batch_weight

        total_mean = (
            self.adapt_mean * existing_weight + batch_mean * batch_weight
        )
        # The variance is computed using the lack-of-fit sum of squares
        # formula (see
        # https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
        total_variance = (self.adapt_variance + (self.adapt_mean - total_mean) ** 2) * existing_weight + \
                              (batch_variance + (batch_mean - total_mean) ** 2) * batch_weight
        self.adapt_mean.assign(total_mean)
        self.adapt_variance.assign(total_variance)
        self.count.assign(total_count)

    def reset_state(self):
        if self.input_mean is not None or not self.built:
            return

        self.adapt_mean.assign(tf.zeros_like(self.adapt_mean))
        self.adapt_variance.assign(tf.ones_like(self.adapt_variance))
        self.count.assign(tf.zeros_like(self.count))

    def finalize_state(self):
        if self.input_mean is not None or not self.built:
            return

        # In the adapt case, we make constant tensors for mean and variance with
        # proper broadcast shape and dtype each time `finalize_state` is called.
        self.mean = tf.reshape(self.adapt_mean, self._broadcast_shape)
        self.mean = tf.cast(self.mean, self.compute_dtype)
        self.variance = tf.reshape(self.adapt_variance, self._broadcast_shape)
        self.variance = tf.cast(self.variance, self.compute_dtype)


    def call(self, inputs):
        inputs = self._standardize_inputs(inputs)
        # The base layer automatically casts floating-point inputs, but we
        # explicitly cast here to also allow integer inputs to be passed
        inputs = tf.cast(inputs, self.compute_dtype)
        return inputs * tf.maximum(tf.sqrt(self.variance), backend.epsilon()) + self.mean
        
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_spec):
        return input_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "mean": utils.listify_tensors(self.input_mean),
                "variance": utils.listify_tensors(self.input_variance),
            }
        )
        return config

    def _standardize_inputs(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if inputs.dtype != self.compute_dtype:
            inputs = tf.cast(inputs, self.compute_dtype)
        return inputs


    



if __name__ == '__main__':

    print("Keras custom de-normalization layer ")

    import matplotlib.pyplot as plt
    
    # Test custom layer
    n, m = 1000, 2
    X = tf.cast(tf.linspace(-10.0, 10.0, n), dtype=tf.float32)
    X = tf.reshape(tensor=X, shape=(n,1))
    X2 = tf.cast(tf.linspace(-3, 3, n), dtype=tf.float32)
    X2 = tf.reshape(tensor=X2, shape=(n,1))
    X = tf.concat(values=(X, X2), axis=1)
    y = 5.5 * tf.math.sin(X[:, 0]) + 0.5 * tf.math.sin(2.5 * X[:, 1]) + 2.0 + 0.2 * tf.random.normal(shape=(n,))

    X_mean = tf.math.reduce_mean(input_tensor=X, axis=0)
    X_var  = tf.math.reduce_variance(input_tensor=X, axis=0)
    y_mean = tf.math.reduce_mean(input_tensor=y)
    y_var  = tf.math.reduce_variance(input_tensor=y)

    inputs = keras.Input(shape=X.shape[1:])    
    x = inputs
    x = layers.Normalization(mean = X_mean, variance = X_var)(inputs)
    x = layers.Dense(40, activation='tanh')(x)
    x = layers.Dense(5, activation='tanh')(x)
    x = layers.Dense(1)(x)
    x = DeNormalization(mean=y_mean, variance=y_var)(x)

    model = keras.Model(inputs, x)
    model.compile(optimizer='rmsprop', loss='mse', metrics='mse')

    model.fit(X, y, batch_size=100, epochs=1000, validation_split=0.3)

    yhat = model.predict(X)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y, 'o')
    ax.plot(yhat, 'ro')
    plt.show()