from __future__ import print_function
import keras
from keras.layers import Input # input layer for sendind patch operator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
import tensorflow as tf
import numpy as np


class ExpMapFixed(Layer):
    """
    """

    def __init__(self, exp_map, batch_size, nv, nrings, ndirs, **kwargs):
        super(ExpMapFixed, self).__init__(**kwargs)
        y = np.expand_dims(exp_map, axis=0)
        y = np.repeat(y, repeats=batch_size, axis=0)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        b = np.zeros((self.nbatch, self.nv, self.nrings, self.ndirs), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :, :] += i
        y = np.stack([b, y], axis=-1)
        self.exp_map = tf.convert_to_tensor(y, dtype=tf.int32)



    def build(self, input_shape):

        self.built = True

        super(ExpMapFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return self.exp_map

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv, self.nrings, self.ndirs, 2)


class FrameTransporterFixed(Layer):
    """
    """

    def __init__(self, exp_map, transport, batch_size, nv, nrings, ndirs, **kwargs):
        super(FrameTransporterFixed, self).__init__(**kwargs)
        e = np.expand_dims(exp_map, axis=0)
        e = np.repeat(e, repeats=batch_size, axis=0)
        f = np.expand_dims(transport, axis=0)
        f = np.repeat(f, repeats=batch_size, axis=0)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        b = np.zeros((self.nbatch, self.nv, self.nrings, self.ndirs), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :, :] += i
        t = np.stack([b, e, f], axis=-1)
        self.transport = tf.convert_to_tensor(t, dtype=tf.int32)

    def build(self, input_shape):

        self.built = True

        super(FrameTransporterFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        return self.transport

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv, self.nrings, self.ndirs, 3)


class GeodesicFieldFixed(Layer):
    """
    """

    def __init__(self, exp_map, transport, batch_size, nv, nrings, ndirs, **kwargs):
        super(GeodesicFieldFixed, self).__init__(**kwargs)
        e = np.expand_dims(exp_map, axis=0)
        e = np.repeat(e, repeats=batch_size, axis=0)
        f = np.expand_dims(transport, axis=0)
        f = np.repeat(f, repeats=batch_size, axis=0)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        b = np.zeros((self.nbatch, self.nv, self.nrings, self.ndirs), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :, :] += i
        s = np.zeros((batch_size, nv, nrings, ndirs), dtype=np.int32)
        for i in range(ndirs):
            s[:, :, :, i] += i
        f = f + s
        f = np.remainder(f, self.ndirs)
        t = np.stack([b, e, f], axis=-1)
        self.transport = tf.convert_to_tensor(t, dtype=tf.int32)

    def build(self, input_shape):

        self.built = True

        super(GeodesicFieldFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        return self.transport

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv, self.nrings, self.ndirs, 3)


class PatchOperatorFixed(Layer):
    """
    """

    def __init__(self, contributors, weights, angles, batch_size, **kwargs):
        super(PatchOperatorFixed, self).__init__(**kwargs)

        shape = np.shape(contributors)
        self.nbatch = batch_size
        self.nv = shape[0]
        self.nrings = shape[1]
        self.ndirs = shape[2]

        """
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1, 1))
        b = tf.tile(batch_idx, (1, self.nv, self.nrings, self.ndirs, 3))
        """
        c = tf.convert_to_tensor(contributors, dtype=tf.int32)
        c = K.expand_dims(c, axis=0)
        self.c = K.repeat_elements(c, rep=self.nbatch, axis=0)
        w = tf.convert_to_tensor(weights, dtype=tf.float32)
        w = K.expand_dims(w, axis=0)
        self.w = K.repeat_elements(w, rep=self.nbatch, axis=0)
        a = tf.convert_to_tensor(angles, dtype=tf.float32)
        a = K.expand_dims(a, axis=0)
        self.a = K.repeat_elements(a, rep=self.nbatch, axis=0)

    def build(self, input_shape):

        self.built = True

        super(PatchOperatorFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return [self.c, self.w, self.a]

    def compute_output_shape(self, input_shape):
        shape = (self.nbatch, self.nv, self.nrings, self.ndirs, 3)
        return [shape, shape, shape]

