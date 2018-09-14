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
from custom_layers import cyclic_shift
import tensorflow as tf
import numpy as np
from sampling import downsample_mesh_sync, downsample_mesh_async
from sampling import upsample_mesh_sync, upsample_mesh_async


class PoolSurfaceFixed(Layer):
    """
    """

    def __init__(self, pool_points, batch_size, nv2, **kwargs):
        super(PoolSurfaceFixed, self).__init__(**kwargs)
        y = np.expand_dims(pool_points, axis=0)
        y = np.repeat(y, repeats=batch_size, axis=0)
        self.nbatch = batch_size
        self.nv2 = nv2

        b = np.zeros((self.nbatch, self.nv2), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :] += i
        y = np.stack([b, y], axis=2)
        self.pool = tf.convert_to_tensor(y, dtype=tf.int32)

    def build(self, input_shape):

        self.built = True

        super(PoolSurfaceFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return self.pool

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv2, 2)


class PoolFrameBundleFixed(Layer):
    """
    """

    def __init__(self, pool_points, pool_frames, batch_size, nv2, ndirs1, ndirs2, **kwargs):
        super(PoolFrameBundleFixed, self).__init__(**kwargs)

        p = np.expand_dims(pool_points, axis=1)
        p = np.repeat(p, repeats=ndirs2, axis=1)
        p = np.expand_dims(p, axis=0)
        p = np.repeat(p, repeats=batch_size, axis=0)

        s = np.expand_dims(pool_frames, axis=1)
        s = np.repeat(s, repeats=ndirs2, axis=1)
        s = np.expand_dims(s, axis=0)
        s = np.repeat(s, repeats=batch_size, axis=0)

        self.nbatch = batch_size
        self.nv2 = nv2
        self.ndirs2 = ndirs2
        self.ndirs1 = ndirs1

        b = np.zeros((self.nbatch, self.nv2, self.ndirs2), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :] += i
        s_ = np.zeros((self.nbatch, self.nv2, self.ndirs2), dtype=np.float32)
        for i in range(ndirs2):
            s_[:, :, i] += (1.*i*self.ndirs1)/(1.*self.ndirs2)
        # see tf.round and tf.cast
        s_ = np.around(s_)
        s_ = s_.astype(int)
        s = s + s_
        s = np.remainder(s, self.ndirs1)
        pool = np.stack([b, p, s], axis=3)
        self.pool = tf.convert_to_tensor(pool, dtype=tf.int32)

    def build(self, input_shape):

        self.built = True

        super(PoolFrameBundleFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        return self.pool

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv2, self.ndirs2, 3)


class PoolingOperatorFixed(Layer):
    """
    """
    def __init__(self, parents, angular_shifts, batch_size, **kwargs):
        self.nbatch = batch_size
        self.nv = np.shape(parents)[0]
        p = tf.convert_to_tensor(parents, dtype=tf.int32)
        p = K.expand_dims(p, axis=0)
        self.p = K.repeat_elements(p, rep=self.nbatch, axis=0)
        a = tf.convert_to_tensor(angular_shifts, dtype=tf.float32)
        a = K.expand_dims(a, axis=0)
        self.a = K.repeat_elements(a, rep=self.nbatch, axis=0)
        super(PoolingOperatorFixed, self).__init__(**kwargs)

    def build(self, input_shape):

        self.built = True

        super(PoolingOperatorFixed, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return [self.p, self.a]

    def compute_output_shape(self, input_shape):
        shape = (self.nbatch, self.nv)
        return [shape, shape]


class Pooling(Layer):
    """
    """

    def __init__(self, **kwargs):
        super(Pooling, self).__init__(**kwargs)

    def build(self, input_shape):

        self.built = True

        super(Pooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if K.ndim(inputs[0]) == 3:
            return downsample_mesh_async(inputs[0], inputs[1])
        else:
            return downsample_mesh_sync(inputs[0], inputs[1], inputs[2])

    def compute_output_shape(self, input_shape):
        if len(input_shape[0]) == 3:
            return (input_shape[0][0], input_shape[1][1], input_shape[0][2])
        else:
            return (input_shape[0][0], input_shape[1][1], input_shape[0][2], input_shape[0][3])


class Shape(Layer):
    """
    """

    def __init__(self, axis=None, **kwargs):
        self.axis = axis
        super(Shape, self).__init__(**kwargs)

    def build(self, input_shape):

        self.built = True

        super(Shape, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.axis is None:
            return K.shape(inputs)
        else:
            return K.shape(inputs)[self.axis]

    def compute_output_shape(self, input_shape):
        if self.axis is None:
            return (1, )
        else:
            return (len(input_shape), )


class TransposedPooling(Layer):
    """
    """

    def __init__(self, new_nv, new_ndirs=None, **kwargs):
        self.new_nv = new_nv
        self.new_ndirs = new_ndirs
        super(TransposedPooling, self).__init__(**kwargs)

    def build(self, input_shape):

        self.built = True

        super(TransposedPooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.new_nv is None:
            new_nv = inputs[-1]
        else:
            new_nv = self.new_nv

        if K.ndim(inputs[0]) == 3:
            return upsample_mesh_async(inputs[0], inputs[1], new_nv)
        else:
            if self.new_ndirs is None:
                raise ValueError('Unspecified number of directional channels')
            return upsample_mesh_sync(inputs[0], inputs[1], inputs[2], new_nv, self.new_ndirs)

    def compute_output_shape(self, input_shape):
        if len(input_shape[0]) == 3:
            return (input_shape[0][0], self.new_nv, input_shape[0][2])
        else:
            return (input_shape[0][0], self.new_nv, self.new_ndirs, input_shape[0][-1])


class LocalAverage(Layer):

    def __init__(self, **kwargs):
        super(LocalAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape[0]) == 4:
            self.ndirs = input_shape[0][2]

        super(LocalAverage, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        y = inputs[0]

        if K.ndim(y) == 3:
            exp_map = inputs[1]
            y = tf.gather_nd(y, exp_map)
            y = K.sum(y, axis=2, keepdims=False)
            y = K.sum(y, axis=2, keepdims=False)
            y = y + inputs[0]
            y /= (K.shape(exp_map)[2]*K.shape(exp_map)[3] + 1.)
            return y
        elif K.ndim(y) == 4:
            geod_field = inputs[1]
            y = tf.gather_nd(y, geod_field)
            y = K.sum(y, axis=2, keepdims=False)
            y = y + inputs[0]
            y /= (K.shape(geod_field)[2] + 1.)
            return y

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ShrinkPatchOpDisc(Layer):

    def __init__(self, nrings, **kwargs):
        self.nrings = nrings
        super(ShrinkPatchOpDisc, self).__init__(**kwargs)

    def build(self, input_shape):

        super(ShrinkPatchOpDisc, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        Exp = inputs[0]
        Frame = inputs[1]
        n = self.nrings
        return [Exp[:, :, 0:n, :, :], Frame[:, :, 0:n, :, :]]

    def compute_output_shape(self, input_shape):
        inp0 = input_shape[0]
        inp1 = input_shape[1]
        return [(inp0[0], inp0[1], self.nrings, inp0[3], 2), (inp1[0], inp1[1], self.nrings, inp1[3], 3)]


class MaxPooling(Layer):

    def __init__(self, r=-1, take_max=False, **kwargs):
        self.r = r
        self.take_max = take_max
        super(MaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape[0]) == 4:
            self.ndirs = input_shape[0][2]

        super(MaxPooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        y = inputs[0]

        if K.ndim(y) == 3:
            exp_map = inputs[1]
            if self.r > 0:
                exp_map = exp_map[:, :, :self.r, :, :]
            y = tf.gather_nd(y, exp_map)
            y = K.max(y, axis=2, keepdims=False)
            y = K.max(y, axis=2, keepdims=False)
            y = K.maximum(y, inputs[0])
            return y
        elif K.ndim(y) == 4:
            frame_transporter = inputs[1]
            if self.r > 0:
                frame_transporter = frame_transporter[:, :, :self.r, :, :]
            # frame transporter is a (nbatch, nv, nrings, ndirs, 3) tensor
            def conv_cond(l, f, Y):
                return tf.less(l, self.ndirs)

            def conv_body(l, f, Y):
                z = tf.gather(f, cyclic_shift(l, self.ndirs, reverse=False), axis=2)
                z = tf.gather_nd(z, frame_transporter)
                z = K.max(z, axis=2, keepdims=False)
                z = K.max(z, axis=2, keepdims=False)
                z = K.maximum(z, tf.gather(f, l, axis=2))
                Y = Y.write(l, z)
                return (l + 1, f, Y)

            Y_ = tf.TensorArray(size=self.ndirs, dtype=tf.float32)
            conv_loop_inp = (0, y, Y_)
            Y_ = tf.while_loop(cond=conv_cond, body=conv_body, loop_vars=conv_loop_inp, parallel_iterations=10)[2]
            y = Y_.stack()
            y = K.permute_dimensions(y, (1, 2, 0, 3))
            if self.take_max:
                y = K.max(y, axis=2, keepdims=False)
            return y

    def compute_output_shape(self, input_shape):
        if len(input_shape[0]) == 4:
            if self.take_max:
                return (input_shape[0][0], input_shape[0][1], input_shape[0][-1])
        return input_shape[0]


class AngularMaxPooling(Layer):

    def __init__(self, r=1, take_max=False, **kwargs):
        self.take_max = take_max
        self.r = r
        super(AngularMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 4:
            self.ndirs = input_shape[2]
        super(AngularMaxPooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, y):
        if K.ndim(y) == 3:
            return y
        elif K.ndim(y) == 4:
            if self.take_max:
                y = K.max(y, axis=2, keepdims=False)
            else:
                n = 2*self.r + 1
                a = self.ndirs-self.r
                b = self.r
                y = K.concatenate([y[:, :, a:self.ndirs, :], y, y[:, :, 0:b, :]], axis=2)
                # y = K.concatenate([y, y[:, :, 0:n-1, :]], axis=2)
                # y = keras.layers.MaxPool2D(pool_size=(1, n), strides=(1, 1), padding='valid', data_format=None)(y)
                y = keras.backend.pool2d(y, pool_size=(1, n), strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
                # y = y[:, :, self.r:self.r+self.ndirs, :]
                print(y.get_shape())
            return y

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            if self.take_max:
                return (input_shape[0], input_shape[1], input_shape[-1])
        return input_shape


class AngularAveragePooling(Layer):

    def __init__(self, r=1, take_average=False, **kwargs):
        self.take_average = take_average
        self.r = r
        super(AngularAveragePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 4:
            self.ndirs = input_shape[2]
        super(AngularAveragePooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, y):
        if K.ndim(y) == 3:
            return y
        elif K.ndim(y) == 4:
            if self.take_average:
                print('uuuuuu')
                print(K.shape(y)[2])
                # y /= K.shape(y)[2]
                y = K.sum(y, axis=2, keepdims=False)
            else:
                n = 2*self.r + 1
                a = self.ndirs-self.r
                b = self.r
                y = K.concatenate([y[:, :, a:self.ndirs, :], y, y[:, :, 0:b, :]], axis=2)
                # y = K.concatenate([y, y[:, :, 0:n-1, :]], axis=2)
                y = keras.backend.pool2d(y, pool_size=(1, n), strides=(1, 1), padding='valid', data_format=None,
                                         pool_mode='avg')
                # y = y[:, :, self.r:self.r+self.ndirs, :]
                print(y.get_shape())
            return y

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            if self.take_average:
                return (input_shape[0], input_shape[1], input_shape[-1])
        return input_shape


class PoolDir(Layer):

    def __init__(self, dir=0, take_max=False, **kwargs):
        self.dir = dir
        super(PoolDir, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PoolDir, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, y):
        if K.ndim(y) == 3:
            return y
        elif K.ndim(y) == 4:
            y = y[:, :, self.dir, :]
            return y

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            return (input_shape[0], input_shape[1], input_shape[-1])
        return input_shape


class SelectDir(Layer):

    def __init__(self, nbatch, nv, ndirs, dir=0, take_max=False, **kwargs):
        self.dir = dir
        self.ndirs = ndirs
        self.nv = nv
        self.nbatch = nbatch

        super(SelectDir, self).__init__(**kwargs)

    def build(self, input_shape):
        a = np.zeros(shape=(self.nbatch, self.nv, self.ndirs, 1), dtype=np.float32)
        a[:, :, self.dir, :] = 1.0
        self.a = tf.convert_to_tensor(a, dtype=tf.float32)
        super(SelectDir, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, y):
        if K.ndim(y) == 3:
            return y
        elif K.ndim(y) == 4:
            y = K.prod([y, self.a])
            return y

    def compute_output_shape(self, input_shape):
        return input_shape


class AngularPooling2d(Layer):

    def __init__(self, r=1, pool='max', full=False, **kwargs):
        self.pool = pool
        self.full = full
        self.r = r
        super(AngularPooling2d, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 5:
            self.ndirs = input_shape[3]
        super(AngularPooling2d, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, y):
        if K.ndim(y) == 4:
            return y
        elif K.ndim(y) == 5:
            if self.full:
                if self.pool is 'max':
                    y = K.max(y, axis=3, keepdims=False)
                else:
                    y = K.sum(y, axis=3, keepdims=False) / self.ndirs
            else:
                n = 2*self.r + 1
                a = self.ndirs-self.r
                b = self.r
                y = K.concatenate([y[:, :, a:self.ndirs, :], y, y[:, :, 0:b, :]], axis=2)
                # y = K.concatenate([y, y[:, :, 0:n-1, :]], axis=2)
                # y = keras.layers.MaxPool2D(pool_size=(1, n), strides=(1, 1), padding='valid', data_format=None)(y)
                y = keras.backend.pool2d(y, pool_size=(1, n), strides=(1, 1), padding='valid', data_format=None, pool_mode=self.pool)
                # y = y[:, :, self.r:self.r+self.ndirs, :]
                print(y.get_shape())
            return y

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 5:
            if self.full:
                return (input_shape[0], input_shape[1], input_shape[2], input_shape[-1])
        return input_shape


