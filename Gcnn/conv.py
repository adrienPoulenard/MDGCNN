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
from sampling import bilinear_sampler
import tensorflow as tf
import numpy as np
import math
from sampling import window_interpolation_sync, window_interpolation_async, window_interpolation


def gcnn_conv(inputs, kernel, nbatch, nv, nrings, ndirs, nfilters, nchannels):
    y = inputs

    if nv is None:
        y = y[0, :, :, :, :]
        y = K.concatenate([y, y[:, :, :-1, :]], axis=2)
        y = K.conv2d(y, kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))
        y = tf.squeeze(y, [1])
        y = K.expand_dims(y, axis=0)
    elif nbatch is None:
        y = K.concatenate([y, y[:, :, :, :-1, :]], axis=3)
        conv = lambda x: K.conv2d(x, kernel,
                                  strides=(1, 1),
                                  padding='valid',
                                  data_format='channels_last',
                                  dilation_rate=(1, 1))
        y = tf.map_fn(fn=conv, elems=y, dtype=tf.float32)
        y = tf.squeeze(y, [2])
    else:
        y = K.reshape(y, (nbatch * nv, nrings, ndirs, nchannels))
        # pad it along the dirs axis so that conv2d produces circular
        # convolution along that dimension
        # shape = (nbatch, nv, ndirs, nchannel)
        y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding

        y = K.conv2d(y, kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))

        y = K.reshape(y, (nbatch, nv, 1, ndirs, nfilters))

        y = tf.squeeze(y, [2])

    return y

class GeodesicConv(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs, nrings=16,
                 ntraindirs=None,
                 sync_mode='radial_sync',
                 take_max=True,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 weights=None,
                 **kwargs):
        super(GeodesicConv, self).__init__(**kwargs)

        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
        if ntraindirs is None:
            self.ntraindirs = ndirs
        else:
            self.ntraindirs = ntraindirs
        k_idx = np.zeros(ndirs)
        for i in range(ndirs):
            k_idx[i] = i*self.ntraindirs / ndirs + 0.001
        k_idx = np.around(k_idx)

        k_idx = k_idx.astype(int)
        k_idx = np.remainder(k_idx, self.ntraindirs)

        print('k_idx')
        print(k_idx)

        self.k_idx = tf.convert_to_tensor(k_idx, dtype=tf.int32)
        # self.BatchNormLayer = LL.BatchNormLayer()
        # self.take_max = take_max

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=self.rank + 2)
        self.nv = nv
        self.take_max = take_max
        self.sync_mode = sync_mode
        self.constweights = False
        if weights is not None:
            self.constweights = True
            self.use_bias = True
            #self.k = tf.convert_to_tensor(weights[0][1:nrings+1, :, :, :], dtype=tf.float32)
            #c_k = weights[0][1, :, :, :]
            #self.center_k = tf.convert_to_tensor(c_k, dtype=tf.float32)
            self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
            self.c = tf.convert_to_tensor(weights[1], dtype=tf.float32)
            self.b = tf.convert_to_tensor(weights[2], dtype=tf.float32)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # see def of Conv_ at https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
        # and https://keras.io/initializers/
        # and https://keras.io/regularizers/

        if self.nv != input_shape[0][1]:
            raise ValueError('Unexpected number of vertexes')

        if input_shape[0][-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        nchannels = input_shape[0][-1]
        kernel_shape = (self.nrings, self.ntraindirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            self.kernel = self.k
            self.center_kernel = self.c
            self.bias = self.b
        else:
            self.kernel_weights = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.ntraindirs is not self.ndirs:
                self.kernel = tf.gather(self.kernel_weights, self.k_idx, axis=1)
            else:
                self.kernel = self.kernel_weights

            self.center_kernel = self.add_weight(shape=(nchannels, self.nfilters),
                                                 initializer=self.kernel_initializer,
                                                 name='center_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.nfilters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None

        # Set input spec.
        # self.input_spec = InputSpec(ndim=self.rank + 2,
        #                             axes={channel_axis: input_dim})
        # build frame transporter pullback
        # y = tf.convert_to_tensor(self.connectivity, dtype=tf.int32)
        # self.patches = K.reshape(y, (self.nv, self.nrings, self.ndirs))

        self.built = True

        super(GeodesicConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs[0]
        contributors = inputs[1]
        weights = inputs[2]
        angles = inputs[3]

        if self.sync_mode is None or self.sync_mode is 'radial_sync':

            if K.ndim(y) == 3:
                y = K.expand_dims(y, axis=2)
                y = K.repeat_elements(y, rep=self.ndirs, axis=2)

            y_ = y


            nbatch = K.shape(y)[0]
            nchannels = K.shape(y)[-1]

            # synchronize with sync field

            y = window_interpolation_sync(y, contributors, weights, angles)

            # circular convolution

            y = gcnn_conv(y, self.kernel, nbatch, self.nv, self.nrings, self.ndirs, self.nfilters, nchannels)

            # add contribution of central vertex

            y += K.dot(y_, self.center_kernel)

            if self.use_bias:
                y = K.bias_add(y, self.bias, data_format=None)

            #y = y + center

            if self.activation is not None:
                y = self.activation(y)

            if self.take_max:
                y = K.max(y, axis=2, keepdims=False)
        elif self.sync_mode is 'async':

            if K.ndim(y) == 4:
                y = K.max(y, axis=2, keepdims=False)
            y_ = y

            nbatch = K.shape(y)[0]
            nchannels = K.shape(y)[-1]

            # pull back the input to the fiber product of the tangent bundle by the frame bundle
            # by the frame transporter

            y = window_interpolation_async(y, contributors, weights)

            y = gcnn_conv(y, self.kernel, nbatch, self.nv, self.nrings, self.ndirs, self.nfilters, nchannels)

            # add contribution of central vertex
            y_ = K.dot(y_, self.center_kernel)
            y_ = K.expand_dims(y_, axis=2)
            y_ = K.repeat_elements(y_, rep=self.ndirs, axis=2)
            y += y_

            if self.use_bias:
                y = K.bias_add(y, self.bias, data_format=None)

            if self.activation is not None:
                y = self.activation(y)

            if self.take_max:
                y = K.max(y, axis=2, keepdims=False)
        return y

    def compute_output_shape(self, input_shape):
        if self.take_max:
            return (input_shape[0][0], input_shape[0][1], self.nfilters)
        else:
            return (input_shape[0][0], input_shape[0][1], self.ndirs, self.nfilters)


def cyclic_shift(shift, n, reverse=False):
    a = np.arange(n, dtype=np.int32)
    if reverse:
        return (a-(shift+n)) % n
    else:
        return (a+shift) % n


def local3d_conv(ndirs, w, X, window_kernel, position_kernel=None):
    # w is the (nbatch, nv, nrings, ndirs, 3) (interpoled) 3d window tensor

    pos = X
    # computing the rotated frames
    X = tf.expand_dims(X, axis=2)
    x = tf.subtract(w[:, :, 0, :, :], X)
    x = tf.nn.l2_normalize(x, axis=-1)
    y = K.concatenate([tf.expand_dims(x[:, :, -1, :], axis=2), x[:, :, :-1, :]], axis=2)
    z = tf.cross(x, y)
    z = tf.reduce_sum(z, axis=2, keepdims=True)
    z = tf.nn.l2_normalize(z, axis=-1)
    z = tf.tile(z, [1, 1, ndirs, 1])
    x_z = tf.expand_dims(tf.einsum('ijkl,ijkl->ijk', x, z), axis=3)
    x = x - tf.multiply(x_z, x)
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.cross(z, x)
    frames = tf.stack([x, y, z], axis=4)

    cyclic_shifts = np.zeros((ndirs, ndirs), dtype=np.int32)

    for i in range(ndirs):
        cyclic_shifts[i, :] = cyclic_shift(i, ndirs, reverse=False)

    cyclic_shifts = tf.convert_to_tensor(cyclic_shifts, dtype=tf.int32)
    # window_kernel = tf.gather(window_kernel, cyclic_shifts, axis=1)


    ######### window convolution #########
    # apply frames f_ijklm to window w_ijnol
    # 'ijklm,ijnol->ijknom'
    # apply window kernel shifts K_nkomp to the rotated windows W_ijknom
    # 'nkomp,ijknom->ijkp'

    # Y = tf.einsum('nkomp,ijklm,ijnol->ijkp', window_kernel, frames, w)

    ######### central point contribution #########
    # apply rotated frames f_ijklm to postion vector P_ijl
    # 'ijklm,ijl->ijkm'
    # apply position kernel K_kmn to rotated position vectors P_ijkm
    # 'kmn,ijkm->ijkn'

    if position_kernel is not None:
        # computing position vector
        barycenter = tf.reduce_mean(pos, axis=1, keepdims=True)
        pos = tf.subtract(pos, barycenter)
        # pos = tf.nn.l2_normalize(pos, axis=-1)
        position_kernel = tf.expand_dims(position_kernel, axis=0)
        position_kernel = tf.tile(position_kernel, [ndirs, 1, 1])
        print('frames shape')
        print(K.int_shape(frames))
        print('pos shape')
        print(K.int_shape(pos))
        print('pos_kernel_shape')
        print(K.int_shape(position_kernel))
        Y = tf.einsum('kmn,ijklm,ijl->ijkn', position_kernel, frames, pos)
        print('y_shape')
        print(K.int_shape(Y))
        return Y
    return Y


def vector_from_barycenter(x):
    barycenter = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.subtract(x, barycenter)


class Local3dGeodesicConv(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs=8, nrings=2,
                 use_global_context=False,
                 ntraindirs=None,
                 take_max=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 weights=None,
                 **kwargs):
        super(Local3dGeodesicConv, self).__init__(**kwargs)

        self.use_global_context = use_global_context
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
        if ntraindirs is None:
            self.ntraindirs = ndirs
        else:
            self.ntraindirs = ntraindirs
        k_idx = np.zeros(ndirs)
        for i in range(ndirs):
            k_idx[i] = i*self.ntraindirs / ndirs + 0.001
        k_idx = np.around(k_idx)

        k_idx = k_idx.astype(int)
        k_idx = np.remainder(k_idx, self.ntraindirs)

        print('k_idx')
        print(k_idx)

        self.k_idx = tf.convert_to_tensor(k_idx, dtype=tf.int32)
        # self.BatchNormLayer = LL.BatchNormLayer()
        # self.take_max = take_max

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=self.rank + 2)
        self.nv = nv
        self.take_max = take_max

        self.constweights = False
        if weights is not None:
            self.constweights = True
            self.use_bias = True
            #self.k = tf.convert_to_tensor(weights[0][1:nrings+1, :, :, :], dtype=tf.float32)
            #c_k = weights[0][1, :, :, :]
            #self.center_k = tf.convert_to_tensor(c_k, dtype=tf.float32)
            if self.use_global_context:
                self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
                self.c = tf.convert_to_tensor(weights[1], dtype=tf.float32)
                self.b = tf.convert_to_tensor(weights[2], dtype=tf.float32)
            else:
                self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
                self.c = None
                self.b = tf.convert_to_tensor(weights[1], dtype=tf.float32)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # see def of Conv_ at https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
        # and https://keras.io/initializers/
        # and https://keras.io/regularizers/

        if self.nv != input_shape[0][1]:
            raise ValueError('Unexpected number of vertexes')

        if input_shape[0][-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        if input_shape[0][-1] is not 3:
            raise ValueError('Unexpected number of input channels')

        kernel_shape = (self.nrings, self.ntraindirs, 3, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            self.window_kernel = self.k
            self.position_kernel = self.c
            self.bias = self.b
        else:
            """
            self.kernel_weights = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='local_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            """
            if self.ntraindirs is not self.ndirs:
                self.window_kernel = tf.gather(self.kernel_weights, self.k_idx, axis=1)
            else:
                # self.window_kernel = self.kernel_weights
                self.window_kernel = None

            if self.use_global_context:
                self.position_kernel = self.add_weight(shape=(3, self.nfilters),
                                                     initializer=self.kernel_initializer,
                                                     name='global_kernel',
                                                     regularizer=self.kernel_regularizer,
                                                     constraint=self.kernel_constraint)
            else:
                self.position_kernel = None

            if self.use_bias:
                self.bias = self.add_weight(shape=(self.nfilters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None

        # Set input spec.
        # self.input_spec = InputSpec(ndim=self.rank + 2,
        #                             axes={channel_axis: input_dim})
        # build frame transporter pullback
        # y = tf.convert_to_tensor(self.connectivity, dtype=tf.int32)
        # self.patches = K.reshape(y, (self.nv, self.nrings, self.ndirs))

        self.built = True

        super(Local3dGeodesicConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs[0]
        contributors = inputs[1]
        weights = inputs[2]

        if K.int_shape(y)[-1] is not 3:
            print(K.int_shape(y)[-1])
            raise ValueError('Unexpected number of input channels')

        # pull back the input to the fiber product of the tangent bundle by the frame bundle
        # by the frame transporter

        w = window_interpolation_async(y, contributors, weights)

        y = local3d_conv(ndirs=self.ndirs, window_kernel=self.window_kernel, w=w, X=y,
                         position_kernel=self.position_kernel)

        if self.use_bias:
            y = K.bias_add(y, self.bias, data_format=None)

        if self.activation is not None:
            y = self.activation(y)

        if self.take_max:
            y = K.max(y, axis=2, keepdims=False)

        return y

    def compute_output_shape(self, input_shape):
        if self.take_max:
            return (input_shape[0][0], input_shape[0][1], self.nfilters)
        else:
            return (input_shape[0][0], input_shape[0][1], self.ndirs, self.nfilters)


def grid_exp_map(y, x, axis, stride, radius, nrings, ndirs, r, t):
    if axis == 0:
        return 0.5 + x*stride + radius*r*np.cos(2.*t*np.pi/ndirs)/nrings
    else:
        return 0.5 + y*stride + radius * r * np.sin(2. * t * np.pi / ndirs) / nrings


class GeodesicConvImg(Layer):
    """
    """

    def __init__(self, nbatch, nfilters, ndirs, nrings=2,
                 radius=2.,
                 shape=(32, 32),
                 strides=(1, 1),
                 pool=False,
                 ntraindirs=16,
                 sync_mode='radial_sync',
                 take_max=True,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 weights=None,
                 random_perturb=0.0,
                 **kwargs):
        super(GeodesicConvImg, self).__init__(**kwargs)

        self.nbatch = nbatch
        self.radius = radius
        self.shape_x = shape[0]
        self.shape_y = shape[1]
        self.stride_x = strides[0]
        self.stride_y = strides[1]
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
        self.ntraindirs = ntraindirs
        self.pool = pool
        if not self.pool:
            k_idx = np.zeros(ndirs)
            for i in range(ndirs):
                k_idx[i] = i*ntraindirs / ndirs + 0.001
            k_idx = np.around(k_idx)

            k_idx = k_idx.astype(int)
            k_idx = np.remainder(k_idx, ntraindirs)

            print('k_idx')
            print(k_idx)

            self.k_idx = tf.convert_to_tensor(k_idx, dtype=tf.int32)
        # self.BatchNormLayer = LL.BatchNormLayer()
        # self.take_max = take_max

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=self.rank + 2)
        self.take_max = take_max
        self.sync_mode = sync_mode
        self.constweights = False
        if weights is not None:
            self.constweights = True
            self.use_bias = True
            #self.k = tf.convert_to_tensor(weights[0][1:nrings+1, :, :, :], dtype=tf.float32)
            #c_k = weights[0][1, :, :, :]
            #self.center_k = tf.convert_to_tensor(c_k, dtype=tf.float32)
            self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
            self.c = tf.convert_to_tensor(weights[1], dtype=tf.float32)
            self.b = tf.convert_to_tensor(weights[2], dtype=tf.float32)
        self.random_perturb = random_perturb


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # see def of Conv_ at https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
        # and https://keras.io/initializers/
        # and https://keras.io/regularizers/

        nbatch = self.nbatch
        self.sz_x = int(math.ceil(self.shape_x / self.stride_x))
        self.sz_y = int(math.ceil(self.shape_y / self.stride_y))
        self.e_x = None
        self.e_y = None
        if not self.pool:
            e_x = np.fromfunction(lambda y, x, r, theta: grid_exp_map(y, x, 0, self.stride_x, self.radius,
                                                                   self.nrings, self.ndirs, r, theta),
                                  shape=(self.sz_y, self.sz_x, self.nrings, self.ndirs), dtype=np.float32)
            e_x += self.random_perturb*self.radius*np.random.randn(self.sz_y, self.sz_x, self.nrings, self.ndirs)
            e_x = np.expand_dims(e_x, axis=0)
            e_x = np.repeat(e_x, repeats=nbatch, axis=0)
            self.e_x = tf.convert_to_tensor(e_x.astype(np.float32), dtype=tf.float32)
            e_y = np.fromfunction(lambda y, x, r, theta: grid_exp_map(y, x, 1, self.stride_y,
                                                                   self.radius, self.nrings, self.ndirs, r, theta),
                                  shape=(self.sz_y, self.sz_x, self.nrings, self.ndirs), dtype=np.float32)
            e_y += self.random_perturb * self.radius * np.random.randn(self.sz_y, self.sz_x, self.nrings, self.ndirs)
            e_y = np.expand_dims(e_y, axis=0)
            e_y = np.repeat(e_y, repeats=nbatch, axis=0)
            self.e_y = tf.convert_to_tensor(e_y.astype(np.float32), dtype=tf.float32)

        x = np.fromfunction(lambda j, i: 0.5 + i*self.stride_x, shape=(self.sz_y, self.sz_x), dtype=np.float32)
        x += self.random_perturb*self.radius*np.random.randn(self.sz_y, self.sz_x)
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, repeats=nbatch, axis=0)
        self.x = tf.convert_to_tensor(x.astype(np.float32), dtype=tf.float32)

        y = np.fromfunction(lambda j, i: 0.5 + j * self.stride_y, shape=(self.sz_y, self.sz_x), dtype=np.float32)
        y += self.random_perturb * self.radius * np.random.randn(self.sz_y, self.sz_x)
        y = np.expand_dims(y, axis=0)
        y = np.repeat(y, repeats=nbatch, axis=0)
        print('y shape')
        print(np.shape(y))
        self.y = tf.convert_to_tensor(y.astype(np.float32), dtype=tf.float32)

        # if self.sync_mode is not 'async':


        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        nchannels = input_shape[-1]
        kernel_shape = (self.nrings, self.ntraindirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            if not self.pool:
                self.kernel = self.k
                self.center_kernel = self.c
                self.bias = self.b
            else:
                self.kernel = None
                self.center_kernel = self.c
                self.bias = self.b
        else:
            if not self.pool:
                self.kernel_weights = self.add_weight(shape=kernel_shape,
                                              initializer=self.kernel_initializer,
                                              name='kernel',
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)
                if self.ntraindirs is not self.ndirs:
                    self.kernel = tf.gather(self.kernel_weights, self.k_idx, axis=1)
                else:
                    self.kernel = self.kernel_weights
            else:
                self.kernel = None

            self.center_kernel = self.add_weight(shape=(nchannels, self.nfilters),
                                                 initializer=self.kernel_initializer,
                                                 name='center_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.nfilters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None

        # Set input spec.
        # self.input_spec = InputSpec(ndim=self.rank + 2,
        #                             axes={channel_axis: input_dim})
        # build frame transporter pullback
        # y = tf.convert_to_tensor(self.connectivity, dtype=tf.int32)
        # self.patches = K.reshape(y, (self.nv, self.nrings, self.ndirs))

        self.built = True

        super(GeodesicConvImg, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs

        if self.sync_mode is None or self.sync_mode is 'radial_sync':

            if K.ndim(y) == 4:
                y = K.expand_dims(y, axis=3)
                y = K.repeat_elements(y, rep=self.ndirs, axis=3)

            y_ = bilinear_sampler(y, self.x, self.y, self.nrings, self.ndirs)


            nbatch = K.shape(y)[0]
            nchannels = K.shape(y)[-1]

            if not self.pool:
                # synchronize with sync field

                y = bilinear_sampler(y, self.e_x, self.e_y, self.nrings, self.ndirs)

                # prepare circular convolution

                y = K.reshape(y, (nbatch*self.sz_y*self.sz_x, self.nrings, self.ndirs, nchannels))
                # pad it along the dirs axis so that conv2d produces circular
                # convolution along that dimension
                # shape = (nbatch, nv, ndirs, nchannel)
                y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

                # output is N x outmaps x 1 x nrays if filter size is the same as
                # input image size prior padding

                y = K.conv2d(y, self.kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))

                y = K.reshape(y, (nbatch, self.sz_y, self.sz_x, 1, self.ndirs, self.nfilters))

                y = tf.squeeze(y, [3])

                # add contribution of central vertex

                y += K.dot(y_, self.center_kernel)
            else:
                y = K.dot(y_, self.center_kernel)

            if self.use_bias:
                y = K.bias_add(y, self.bias, data_format=None)

            #y = y + center

            if self.activation is not None:
                y = self.activation(y)

            if self.take_max:
                y = K.max(y, axis=2, keepdims=False)

        elif self.sync_mode is 'async':
            if K.ndim(y) == 5:
                y = K.max(y, axis=3, keepdims=False)
            y_ = bilinear_sampler(y, self.x, self.y, self.nrings, self.ndirs)
            if not self.pool:

                nbatch = K.shape(y)[0]
                nchannels = K.shape(y)[-1]

                # pull back the input to the fiber product of the tangent bundle by the frame bundle
                # by the frame transporter

                y = bilinear_sampler(y, self.e_x, self.e_y, self.nrings, self.ndirs)

                y = K.reshape(y, (nbatch * self.sz_y*self.sz_x, self.nrings, self.ndirs, nchannels))
                # pad it along the dirs axis so that conv2d produces circular
                # convolution along that dimension
                # shape = (nbatch, nv, ndirs, nchannel)
                y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

                # output is N x outmaps x 1 x nrays if filter size is the same as
                # input image size prior padding

                y = K.conv2d(y, self.kernel, strides=(1, 1), padding='valid', data_format='channels_last',
                             dilation_rate=(1, 1))

                y = K.reshape(y, (nbatch, self.sz_y, self.sz_x, 1, self.ndirs, self.nfilters))
                # y = K.max(y, axis=2, keepdims=False)
                y = tf.squeeze(y, [3])

                # add contribution of central vertex

                y_ = K.dot(y_, self.center_kernel)
                y_ = K.expand_dims(y_, axis=3)
                y_ = K.repeat_elements(y_, rep=self.ndirs, axis=3)
                y += y_
            else:
                y_ = K.dot(y_, self.center_kernel)
                y_ = K.expand_dims(y_, axis=3)
                y_ = K.repeat_elements(y_, rep=self.ndirs, axis=3)
                y = y_

            if self.use_bias:
                y = K.bias_add(y, self.bias, data_format=None)

            if self.activation is not None:
                y = self.activation(y)

            if self.take_max:
                y = K.max(y, axis=2, keepdims=False)
        return y

    def compute_output_shape(self, input_shape):
        if self.take_max:
            return (input_shape[0], self.sz_y, self.sz_x, self.nfilters)
        else:
            return (input_shape[0], self.sz_y, self.sz_x, self.ndirs, self.nfilters)