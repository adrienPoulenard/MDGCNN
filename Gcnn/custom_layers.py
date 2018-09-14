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


class Max(Layer):

    def __init__(self, axis=None, keepdims=False, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super(Max, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Max, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.max(x, self.axis, self.keepdims)

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        if self.keepdims:
            lst[self.axis] = 1
        else:
            lst.pop(self.axis)

        shape = tuple(lst)
        return shape


class FrameBundlePullBack(Layer):

    def __init__(self, ndirs, **kwargs):
        self.ndirs = ndirs
        super(FrameBundlePullBack, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FrameBundlePullBack, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        y = K.expand_dims(inputs, axis=2)
        y = K.repeat_elements(y, rep=self.ndirs, axis=2)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.ndirs, input_shape[2])


class RepeatAxis(Layer):

    def __init__(self, rep, axis, **kwargs):
        self.rep = rep
        self.axis = axis
        super(RepeatAxis, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RepeatAxis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return K.repeat_elements(inputs, rep=self.rep, axis=self.axis)

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[self.axis] *= self.rep
        shape = tuple(lst)
        print('rep axis output shape: ')
        print(shape)
        return shape

def exponentialMapPullBack(E, nv, nrings, ndirs):
    e = np.reshape(E, (nv, nrings, ndirs))
    return tf.convert_to_tensor(e, dtype=tf.int32)

def frameTransporterPullBack(E, F, nv, nrings, ndirs):
    e = np.reshape(E, (nv, nrings, ndirs))
    e = e[:, :, :, np.newaxis]
    f = np.reshape(F, (nv, nrings, ndirs))
    f = f[:, :, :, np.newaxis]
    y = np.concatenate((e, f), 3)
    res = tf.convert_to_tensor(y, dtype=tf.int32)
    return res


def cyclic_shift(shift, n, reverse=False):
    a = np.arange(n, dtype=np.int32)
    if reverse:
        return (a-(shift+n)) % n
    else:
        return (a+shift) % n


class ExponentialMap(Layer):
    """
    """
    def __init__(self, nv, nrings, ndirs, connectivity, **kwargs):
        super(ExponentialMap, self).__init__(**kwargs)
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        self.connectivity = connectivity

    def build(self, input_shape):
        self.exp_map = exponentialMapPullBack(self.connectivity,
                                              self.nv, self.nrings, self.ndirs)

        self.built = True

        super(ExponentialMap, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        return self.exp_map

    def compute_output_shape(self, input_shape):
        return (self.nv, self.nrings, self.ndirs)


class FrameTransporter(Layer):
    """
    """

    def __init__(self, nv, nrings, ndirs, connectivity, transport, **kwargs):
        super(FrameTransporter, self).__init__(**kwargs)
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        self.connectivity = connectivity
        self.transport = transport

    def build(self, input_shape):
        self.frame_transporter = frameTransporterPullBack(self.connectivity, self.transport,
                                                          self.nv, self.nrings, self.ndirs)

        self.built = True

        super(FrameTransporter, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        return self.frame_transporter

    def compute_output_shape(self, input_shape):
        return (self.nv, self.nrings, self.ndirs, 2)


class SyncConv(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs, nrings,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SyncConv, self).__init__(**kwargs)

        self.rank = 2
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
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

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # see def of Conv_ at https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
        # and https://keras.io/initializers/
        # and https://keras.io/regularizers/

        if self.nv != input_shape[0][1]:
            raise ValueError('Unexpected number of vertexes')

        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        nchannels = input_shape[0][-1]
        kernel_shape = (self.nrings, self.ndirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # k = tf.gather(params=self.kernel, indices=self.cyclic_permutation, axis=1)

        # k = K.permute_dimensions(k, (1, 0, 2, 3, 4))

        # self.kernel_rotations = K.reshape(k, (self.ndirs, self.nrings * self.ndirs * nchannels, self.nfilters))

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
        # self.frame_transporter = frameTransporterPullBack(self.connectivity, self.transport,
        #                                                  self.nv, self.nrings, self.ndirs)

        self.built = True

        super(SyncConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs[0]
        frame_transporter = inputs[1]

        print('y shape 11: ')
        print(y.get_shape())

        nbatch = K.shape(y)[0]
        nchannels = K.shape(y)[-1]

        y = K.permute_dimensions(y, (1, 2, 3, 0))

        # pull back the input to the fiber product of the tangent bundle by the frame bundle
        # by the frame transporter
        # shape = np.array([nbatch, self.nv, self.nrings, self.ndirs, nchannels, self.nfilters])

        def conv_cond(l, f, Y):
            return tf.less(l, self.ndirs)

        def conv_body(l, f, Y):
            z = tf.gather(f, cyclic_shift(l, self.ndirs, reverse=False), axis=1)
            z = tf.gather_nd(z, frame_transporter)
            z = K.permute_dimensions(z, (0, 4, 1, 2, 3))
            z = K.reshape(z,  (self.nv * nbatch, self.nrings * self.ndirs * nchannels))
            k = tf.gather(self.kernel, cyclic_shift(l, self.ndirs, reverse=True), axis=1)
            k = K.reshape(k, (self.nrings * self.ndirs * nchannels, self.nfilters))
            z = tf.matmul(z, k)
            z = K.reshape(z, (self.nv, nbatch, self.nfilters))
            Y = Y.write(l, z)
            # z = K.reshape(z, (nv * nb, nf))
            # z = tf.expand_dims(z, axis=0)
            return (l+1, f, Y)

        Y_ = tf.TensorArray(size=self.ndirs, dtype=tf.float32)

        conv_loop_inp = (0, y, Y_)

        Y_ = tf.while_loop(cond=conv_cond, body=conv_body, loop_vars=conv_loop_inp)[2]

        y = Y_.stack()

        #y = K.reshape(y, (self.ndirs, self.nv, nbatch, self.nfilters))

        y = K.permute_dimensions(y, (2, 1, 0, 3))

        if self.use_bias:
            y = K.bias_add(y, self.bias, data_format=None)

        print('y shape 55: ')
        print(y.get_shape())

        if self.activation is not None:
            return self.activation(y)

        return y

    def compute_output_shape(self, input_shape):
        print('SGCL input shape: ')
        print(input_shape)
        return (input_shape[0][0], input_shape[0][1], self.ndirs, self.nfilters)
        # return (input_shapes[0][0], input_shapes[0][1], 16, 1)

class AsyncConv(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs, nrings,
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
                 **kwargs):
        super(AsyncConv, self).__init__(**kwargs)

        self.rank = 2
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
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
        kernel_shape = (self.nrings, self.ndirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
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

        super(AsyncConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs[0]
        exp_map = inputs[1]

        nbatch = K.shape(y)[0]
        nchannels = K.shape(y)[-1]

        # pull back the input to the fiber product of the tangent bundle by the frame bundle
        # by the frame transporter

        y = tf.gather(y, exp_map, axis=1)

        y = K.reshape(y, (nbatch*self.nv, self.nrings, self.ndirs, nchannels))
        # pad it along the dirs axis so that conv2d produces circular
        # convolution along that dimension
        # shape = (nbatch, nv, ndirs, nchannel)
        y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding

        y = K.conv2d(y, self.kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))

        y = K.reshape(y, (nbatch, self.nv, 1, self.ndirs, self.nfilters))
        # y = K.max(y, axis=2, keepdims=False)
        y = tf.squeeze(y, [2])

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


def cyclic_rotations_3d(ndirs):
    lst = []
    for i in range(ndirs):
        c = np.cos((2.*i*np.pi)/ndirs)
        s = np.sin((2.*i*np.pi)/ndirs)
        r = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], dtype=np.float32)
        lst.append(r)
    res = np.stack(lst, axis=0)
    return tf.convert_to_tensor(res, dtype=tf.float32)


def cyclic_rotation_3d(i, ndirs):
    c = np.cos((2.*i*np.pi)/ndirs)
    s = np.sin((2.*i*np.pi)/ndirs)
    r = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], dtype=np.float32)
    return tf.convert_to_tensor(r, dtype=tf.float32)


class Input3d(Layer):
    """
    """

    def __init__(self, batch_size, nv, nrings, ndirs, nfilters, take_max=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Input3d, self).__init__(**kwargs)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        self.nfilters = nfilters
        self.use_bias = use_bias
        self.take_max = take_max


        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self.rotations = cyclic_rotations_3d(self.ndirs)

        """
        self.rotations = tf.TensorArray(size=self.ndirs, dtype=tf.float32)

        rot_tmp = cyclic_rotations_3d(self.ndirs)
        for i in range(self.ndirs):
            self.rotations.write(i, rot_tmp[i])
        """

        kernel_shape = (self.nrings, self.ndirs, 3, self.nfilters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
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

        self.built = True

        super(Input3d, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        basepoints = inputs[0]
        nbatch = K.shape(basepoints)[0]
        expmap = inputs[2]
        local_frames = inputs[1]


        # pull back to tangent patches
        y = tf.gather_nd(basepoints, expmap)
        # center local 3d coordinates
        bp = K.expand_dims(basepoints, 2)
        bp = K.expand_dims(bp, 2)

        bp = K.repeat_elements(bp, self.nrings, axis=2)
        bp = K.repeat_elements(bp, self.ndirs, axis=3)

        #y -= bp
        y = tf.subtract(y, bp)

        # rotate local 3d coordinates
        y = tf.einsum('ijkl,ijabl->ijabk', local_frames, y)

        # compute convolution with kernel
        y = K.reshape(y, (nbatch * self.nv, self.nrings, self.ndirs, 3))
        # pad it along the dirs axis so that conv2d produces circular
        # convolution along that dimension
        # shape = (nbatch, nv, ndirs, nchannel)
        y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding

        y = K.conv2d(y, self.kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))

        y = K.reshape(y, (nbatch, self.nv, 1, self.ndirs, self.nfilters))
        y = tf.squeeze(y, [2])

        if self.use_bias:
            y = K.bias_add(y, self.bias, data_format=None)

        if self.activation is not None:
            y = self.activation(y)

        if self.take_max:
            y = K.max(y, axis=2, keepdims=False)

        return y

    def compute_output_shape(self, input_shape):
        if self.take_max:
            return (self.nbatch, self.nv, self.nfilters)
        else:
            return (self.nbatch, self.nv, self.ndirs, self.nfilters)


def reverse_permutation(s):
    """
    print('uu')
    print(K.shape(s))
    rs = tf.nn.top_k(s, K.shape(s)[0], sorted=True)[1]
    rs = tf.reverse(rs, axis=0)
    """
    return s


class ExpMapBis(Layer):
    """
    """

    def __init__(self, batch_size, nv, nrings, ndirs, **kwargs):
        super(ExpMapBis, self).__init__(**kwargs)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs

    def build(self, input_shape):

        # build batch index tensor
        # B_ijkl = i
        b = np.zeros((self.nbatch, self.nv, self.nrings, self.ndirs), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :, :] += i

        self.b = tf.convert_to_tensor(b, tf.int32)
        self.built = True

        super(ExpMapBis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        e = inputs
        return tf.stack([self.b, e], axis=4)

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv, self.nrings, self.ndirs, 2)


class FrameTransporterBis(Layer):
    """
    """

    def __init__(self, batch_size, nv, nrings, ndirs, **kwargs):
        super(FrameTransporterBis, self).__init__(**kwargs)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs

    def build(self, input_shape):

        # build batch index tensor
        # B_ijkl = i
        b = np.zeros((self.nbatch, self.nv, self.nrings, self.ndirs), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :, :] += i

        self.b = tf.convert_to_tensor(b, tf.int32)
        # self.b = tf.expand_dims(self.b, axis=4)
        self.built = True

        super(FrameTransporterBis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # e = tf.expand_dims(inputs[0], axis=4)
        # f = tf.expand_dims(inputs[1], axis=4)
        e = inputs[0]
        f = inputs[1]
        return tf.stack([self.b, e, f], axis=4)

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv, self.nrings, self.ndirs, 3)


class ShuffleVertices(Layer):
    """
    """
    def __init__(self, batch_size, nv, nrings, ndirs, shuffle=True, **kwargs):
        super(ShuffleVertices, self).__init__(**kwargs)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        self.shuffle = shuffle

    def build(self, input_shape):

        self.s = tf.convert_to_tensor(np.arange(self.nv, dtype=np.int32), dtype=tf.int32)

        self.built = True
        super(ShuffleVertices, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        s = tf.random_shuffle(self.s)
        rs = reverse_permutation(s)
        e = inputs[1]
        f = inputs[2]
        y = inputs[0]
        if self.shuffle:
            e = tf.gather(e, s, axis=1)
            e = tf.gather(rs, e, axis=0)
            f = tf.gather(f, s, axis=1)
            y = tf.gather(y, s, axis=1)
        return [y, e, f, s, rs]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], self.nv, self.nv]


class DeShuffleOutput(Layer):
    """
    """
    def __init__(self, batch_size, nv, nrings, ndirs, shuffle=True, **kwargs):
        super(DeShuffleOutput, self).__init__(**kwargs)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        self.shuffle = shuffle

    def build(self, input_shape):
        self.built = True
        super(DeShuffleOutput, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        y = inputs[0]
        rs = inputs[1]
        if self.shuffle:
            y = tf.gather(y, rs, axis=1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ReshapeBis(Layer):
    """
    """

    def __init__(self, shape,
                 **kwargs):
        self.shape = shape
        super(ReshapeBis, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

        super(ReshapeBis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        return K.reshape(inputs, self.shape)

    def compute_output_shape(self, input_shape):
        return self.shape


class SyncConvBis(Layer):
    """
    """

    def __init__(self, nv, nrings, ndirs, nfilters,
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
        super(SyncConvBis, self).__init__(**kwargs)

        self.rank = 2
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
        # self.BatchNormLayer = LL.BatchNormLayer()
        self.take_max = take_max

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
        self.constweights = False
        if weights is not None:
            self.constweights = True
            self.use_bias = True
            self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
            self.b = tf.convert_to_tensor(weights[1], dtype=tf.float32)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # see def of Conv_ at https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
        # and https://keras.io/initializers/
        # and https://keras.io/regularizers/

        if self.nv != input_shape[0][1]:
            raise ValueError('Unexpected number of vertexes')

        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        nchannels = input_shape[0][-1]
        kernel_shape = (self.nrings*self.ndirs*nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            # should test self.k and self.b shapes
            self.kernel = self.k
            self.bias = self.b
        else:
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
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

        self.built = True

        super(SyncConvBis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs[0]

        if K.ndim(y) == 3:
            y = K.expand_dims(y, axis=2)
            y = K.repeat_elements(y, rep=self.ndirs, axis=2)

        frame_transporter = inputs[1]
        # frame transporter is a (nbatch, nv, nrings, ndirs, 3) tensor

        nbatch = K.shape(y)[0]
        nchannels = K.shape(y)[-1]

        # y = K.permute_dimensions(y, (1, 2, 3, 0))

        # pull back the input to the fiber product of the tangent bundle by the frame bundle
        # by the frame transporter
        # shape = np.array([nbatch, self.nv, self.nrings, self.ndirs, nchannels, self.nfilters])

        def conv_cond(l, f, Y):
            return tf.less(l, self.ndirs)

        def conv_body(l, f, Y):
            z = tf.gather(f, cyclic_shift(l, self.ndirs, reverse=False), axis=2)
            z = tf.gather_nd(z, frame_transporter)
            z = tf.gather(z, cyclic_shift(l, self.ndirs, reverse=False), axis=3)
            z = K.reshape(z,  (nbatch * self.nv, self.nrings * self.ndirs * nchannels))
            # k = tf.gather(self.kernel, cyclic_shift(l, self.ndirs, reverse=True), axis=1)
            k = self.kernel
            # k = K.reshape(k, (self.nrings * self.ndirs * nchannels, self.nfilters))
            z = tf.matmul(z, k)
            z = K.reshape(z, (nbatch, self.nv, self.nfilters))
            Y = Y.write(l, z)
            # z = K.reshape(z, (nv * nb, nf))
            # z = tf.expand_dims(z, axis=0)
            return (l+1, f, Y)

        Y_ = tf.TensorArray(size=self.ndirs, dtype=tf.float32)

        conv_loop_inp = (0, y, Y_)

        Y_ = tf.while_loop(cond=conv_cond, body=conv_body, loop_vars=conv_loop_inp, parallel_iterations=1)[2]

        y = Y_.stack()

        # y = K.reshape(y, (self.ndirs, self.nv, nbatch, self.nfilters))

        y = K.permute_dimensions(y, (1, 2, 0, 3))

        # add contribution of central vertex
        """
        center = Dense(units=self.nfilters,
                       use_bias=False,
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       bias_constraint=self.bias_constraint)(inputs[0])
        """
        #y = y + center

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


class AsyncConvBis(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs, nrings,
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
        super(AsyncConvBis, self).__init__(**kwargs)

        self.rank = 2
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
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
            self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
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
        nchannels = input_shape[0][-1]
        kernel_shape = (self.nrings, self.ndirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            # should test self.k and self.b shapes
            self.kernel = self.k
            self.bias = self.b
        else:
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

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

        super(AsyncConvBis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y_ = inputs[0]

        if K.ndim(y_) == 4:
            y_ = K.max(y_, axis=2, keepdims=False)

        y = y_

        exp_map = inputs[1]

        nbatch = K.shape(y)[0]
        nchannels = K.shape(y)[-1]

        # pull back the input to the fiber product of the tangent bundle by the frame bundle
        # by the frame transporter

        y = tf.gather_nd(y, exp_map)

        """
        def conv_cond(l, f, Y):
            return tf.less(l, self.ndirs)

        def conv_body(l, f, Y):
            z = tf.gather(f, cyclic_shift(l, self.ndirs, reverse=False), axis=3)
            print('z shape')
            print(z.get_shape())
            z = K.reshape(z,  (nbatch * self.nv, self.nrings * self.ndirs * nchannels))
            # k = tf.gather(self.kernel, cyclic_shift(l, self.ndirs, reverse=True), axis=1)
            k = self.kernel
            k = K.reshape(k, (self.nrings * self.ndirs * nchannels, self.nfilters))
            z = tf.matmul(z, k)
            z = K.reshape(z, (nbatch, self.nv, self.nfilters))
            Y = Y.write(l, z)
            # z = K.reshape(z, (nv * nb, nf))
            # z = tf.expand_dims(z, axis=0)
            return (l+1, f, Y)

        Y_ = tf.TensorArray(size=self.ndirs, dtype=tf.float32)

        conv_loop_inp = (0, y, Y_)

        Y_ = tf.while_loop(conv_cond, conv_body, conv_loop_inp)[2]

        y = Y_.stack()

        # y = K.reshape(y, (self.ndirs, self.nv, nbatch, self.nfilters))

        y = K.permute_dimensions(y, (1, 2, 0, 3))
        """

        y = K.reshape(y, (nbatch*self.nv, self.nrings, self.ndirs, nchannels))
        # pad it along the dirs axis so that conv2d produces circular
        # convolution along that dimension
        # shape = (nbatch, nv, ndirs, nchannel)
        y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding

        y = K.conv2d(y, self.kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))

        y = K.reshape(y, (nbatch, self.nv, 1, self.ndirs, self.nfilters))
        # y = K.max(y, axis=2, keepdims=False)
        y = tf.squeeze(y, [2])

        # add contribution of central vertex

        y += K.dot(y_, self.center_kernel)

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




"""
class SpatialMaxPooling(Layer):

    def __init__(self, nrings, take_max=False, **kwargs):
        self.nrings = nrings
        self.take_max = take_max
        super(SpatialMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape[0]) == 4:
            self.ndirs = input_shape[0][2]

        super(SpatialMaxPooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        y = inputs[0]

        if K.ndim(y) == 3:
            exp_map = inputs[1]
            y = tf.gather_nd(y, exp_map)
            y = K.max(y, axis=2, keepdims=False)
            y = K.max(y, axis=2, keepdims=False)
            y = K.maximum(y, inputs[0])
            return y
        elif K.ndim(y) == 4:
            sync_field = inputs[1]
            # sync_field is a (nbatch, nv, nrings, ndirs, 3) tensor
            z = tf.gather_nd(z, frame_transporter)
            z = K.max(z, axis=2, keepdims=False)
            z = K.max(z, axis=2, keepdims=False)
            z = K.maximum(z, tf.gather(f, l, axis=2))

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
            Y_ = tf.while_loop(cond=conv_cond, body=conv_body, loop_vars=conv_loop_inp, parallel_iterations=1)[2]
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
"""

def geodesic_field_shift(batch_size, nv, nrings, ndirs):
    s = np.zeros((batch_size, nv, nrings, ndirs), dtype=np.int32)
    for i in range(ndirs):
        s[:, :, :, i] += i
    s = tf.convert_to_tensor(s, tf.int32)
    return s


class GeodesicField(Layer):
    """
    """

    def __init__(self, batch_size, nv, nrings, ndirs, **kwargs):
        super(GeodesicField, self).__init__(**kwargs)
        self.nbatch = batch_size
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs

    def build(self, input_shape):

        # build batch index tensor
        # B_ijkl = i
        b = np.zeros((self.nbatch, self.nv, self.nrings, self.ndirs), dtype=np.int32)
        for i in range(self.nbatch):
            b[i, :, :, :] += i

        self.b = tf.convert_to_tensor(b, tf.int32)
        # self.b = tf.expand_dims(self.b, axis=4)
        self.built = True

        self.s = geodesic_field_shift(self.nbatch, self.nv, self.nrings, self.ndirs)

        super(GeodesicField, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # e = tf.expand_dims(inputs[0], axis=4)
        # f = tf.expand_dims(inputs[1], axis=4)
        e = inputs[0]
        f = inputs[1] + self.s
        f = tf.mod(f, self.ndirs)
        return tf.stack([self.b, e, f], axis=4)

    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.nv, self.nrings, self.ndirs, 3)


class FrameFiberConv(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs,
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
        super(FrameFiberConv, self).__init__(**kwargs)

        self.rank = 2
        self.nfilters = nfilters
        self.ndirs = ndirs
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
            self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
            self.b = tf.convert_to_tensor(weights[1], dtype=tf.float32)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # see def of Conv_ at https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
        # and https://keras.io/initializers/
        # and https://keras.io/regularizers/

        if self.nv != input_shape[1]:
            raise ValueError('Unexpected number of vertexes')

        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        nchannels = input_shape[-1]
        kernel_shape = (self.ndirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            # should test self.k and self.b shapes
            self.kernel = self.k
            self.bias = self.b
        else:
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
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

        super(FrameFiberConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        y = inputs

        nbatch = K.shape(y)[0]
        nchannels = K.shape(y)[-1]

        y = K.reshape(y, (nbatch*self.nv, self.ndirs, nchannels))
        # pad it along the dirs axis so that conv2d produces circular
        # convolution along that dimension
        # shape = (nbatch, nv, ndirs, nchannel)
        y = K.concatenate([y, y[:, :-1, :]], axis=1)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding

        y = K.conv1d(y, self.kernel, strides=1, padding='valid', data_format='channels_last', dilation_rate=1)

        y = K.reshape(y, (nbatch, self.nv, self.ndirs, self.nfilters))

        if self.use_bias:
            y = K.bias_add(y, self.bias, data_format=None)

        if self.activation is not None:
            y = self.activation(y)

        if self.take_max:
            y = K.max(y, axis=2, keepdims=False)

        return y

    def compute_output_shape(self, input_shape):
        if self.take_max:
            return (input_shape[0], input_shape[1], self.nfilters)
        else:
            return (input_shape[0], input_shape[1], self.ndirs, self.nfilters)


class SyncGeodesicConv(Layer):
    """
    """

    def __init__(self, nfilters, nv, ndirs, nrings,
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
        super(SyncGeodesicConv, self).__init__(**kwargs)

        self.rank = 2
        self.nfilters = nfilters
        # shape of the convolution kernel
        # self.kernel_shape = (nrings, ndirs, self.input_shape[0][2], nfilters)
        # self.kernel_size = conv_utils.normalize_tuple((nrings , ndirs), self.rank, 'kernel_size')
        self.nrings = nrings
        self.ndirs = ndirs
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
            self.k = tf.convert_to_tensor(weights[0], dtype=tf.float32)
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
        nchannels = input_shape[0][-1]
        kernel_shape = (self.nrings, self.ndirs, nchannels, self.nfilters)#self.kernel_size + (input_dim, self.nfilters)

        if self.constweights:
            # should test self.k and self.b shapes
            #print('zzzz')
            #print(tf.shape(self.k))
            self.kernel = self.k
            #print('vvv')
            #print(tf.shape(self.center_k))
            #self.center_kernel = self.center_k
            self.bias = self.b
        else:
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

            self.center_kernel = self.add_weight(shape=(nchannels, self.nfilters),
                                                 initializer=self.kernel_initializer,
                                                 name='center_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)


            """
            self.center_kernel = self.add_weight(shape=(self.ndirs, nchannels, self.nfilters),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            """
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

        super(SyncGeodesicConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # y, M = inputs

        y = inputs[0]

        sync_field = inputs[1]

        if K.ndim(y) == 3:
            y = K.expand_dims(y, axis=2)
            y = K.repeat_elements(y, rep=self.ndirs, axis=2)

        y_ = y


        nbatch = K.shape(y)[0]
        nchannels = K.shape(y)[-1]

        # synchronize with sync field

        y = tf.gather_nd(y, sync_field)

        # prepare circular convolution

        y = K.reshape(y, (nbatch*self.nv, self.nrings, self.ndirs, nchannels))
        # pad it along the dirs axis so that conv2d produces circular
        # convolution along that dimension
        # shape = (nbatch, nv, ndirs, nchannel)
        y = K.concatenate([y, y[:, :, :-1, :]], axis=2)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding

        y = K.conv2d(y, self.kernel, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1))

        y = K.reshape(y, (nbatch, self.nv, 1, self.ndirs, self.nfilters))

        y = tf.squeeze(y, [2])

        # add contribution of central vertex

        y += K.dot(y_, self.center_kernel)

        if self.use_bias:
            y = K.bias_add(y, self.bias, data_format=None)

        #y = y + center

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


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


class ConstantTensor(Layer):
    """
    """

    def __init__(self, const, batch_size, dtype='float', **kwargs):
        super(ConstantTensor, self).__init__(**kwargs)

        self.shape = np.shape(const)
        self.nbatch = batch_size

        if dtype == 'int':
            t = tf.convert_to_tensor(const, dtype=tf.int32)
        else:
            t = tf.convert_to_tensor(const, dtype=tf.float32)
        t = K.expand_dims(t, axis=0)
        t = K.repeat_elements(t, rep=self.nbatch, axis=0)
        self.t = t

    def build(self, input_shape):

        self.built = True

        super(ConstantTensor, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return self.t

    def compute_output_shape(self, input_shape):
        return (self.nbatch, ) + totuple(self.shape)


class DenseStack(Layer):
    """
    """

    def __init__(self, units,
                 slices,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseStack, self).__init__(**kwargs)
        self.units = units
        self.slices = slices
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.slices, input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.slices, self.units),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})


        self.built = True

        super(DenseStack, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        output = tf.tensordot(inputs, self.kernel, [[-1], [1]])
        if self.use_bias:
            output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.slices
        output_shape.append(self.units)
        return tuple(output_shape)


