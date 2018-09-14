import keras
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Reshape, Dropout, Concatenate
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from custom_layers import RepeatAxis, Max, AsyncConv, SyncConv, FrameTransporter, ExponentialMap
from custom_layers import Input3d, SyncConvBis, AsyncConvBis, SyncGeodesicConv
from conv import GeodesicConv
from custom_layers import ExpMapBis, FrameTransporterBis, DenseStack
from pooling import AngularMaxPooling
from custom_regularizers import L1L2_circ
from keras.layers import BatchNormalization, Activation




class ConvOperator(object):
    """
    """

    def __init__(self, conv_op, patch_op, nv, nrings, ndirs, nfilters=1,
                 take_max=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None):

        self.conv_op = conv_op
        self.patch_op = patch_op
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

    def __call__(self, inputs):
        return self.conv_op(nv=self.nv, nrings=self.nrings, ndirs=self.ndirs, nfilters=self.nfilters,
                            take_max=self.take_max, activation=self.activation, use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            activity_regularizer=self.activity_regularizer,
                            kernel_constraint=self.kernel_constraint,
                            bias_constraint=self.bias_constraint)([inputs, self.patch_op])

    def set_take_max(self, take_max):
        self.take_max = take_max

    def get_take_max(self):
        return self.take_max

    def get_nfilters(self):
        return self.nfilters

    def set_nfilters(self, nfilters):
        self.nfilters = nfilters

"""
def Map(f, x, concat=False, axis=-1):
    if isinstance(x, list):
        res = []
        for x_ in x:
            res.append(f(x_))
        if concat:
            return Concatenate(axis=axis)(res)
        else:
            return res
    else:
        return f(x)


def compose(f, x, n):
    res = x
    for i in range(n):
        res = f(res)
    return res


class Composer(object):
    def __init__(self, f, n):
        self.f = f
        self.n = n

    def __call__(self, x):
        return compose(self.f, x, self.n)

    def get_n(self):
        return self.n

    def set_n(self, n):
        self.n = n


def sparseConvBlock(inputs, conv, nfilters, nsplits=1, nlayers=1, take_max=False, concat=True):
    take_max_tmp = conv.get_take_max()
    nfilters_tmp = conv.get_nfilters()
    conv.set_take_max(False)
    conv.set_nfilters(nfilters)
    cmp = Composer(conv, nlayers)
    branches = []

    if isinstance(inputs, list):
        print('uuuuuuu')
        branches = inputs
    else:
        for i in range(nsplits):
            branches.append(inputs)

    res = Map(f=cmp, x=branches, concat=concat)

    conv.set_take_max(take_max_tmp)
    conv.set_nfilters(nfilters_tmp)

    if take_max:
        Map(f=Max(axis=2, keepdims=False), x=res)
    else:
        return res



"""


#model.add(Lambda(lambda x: x ** 2))

# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

class Map(object):
    """
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return tf.map_fn(self.f, x)


class Split(object):
    """
    """
    def __init__(self, num_or_size_splits, axis):
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def __call__(self, x):
        return tf.split(value=x, num_or_size_splits=self.num_or_size_splits, axis=self.axis)


class LinkLayer(object):
    """
    """
    def __init__(self, f, n_units, n_in, n_out):
        self.f = f
        self.n_units = n_units
        self.n_in = n_in
        self.n_out = n_out
        self.split = Split(num_or_size_splits=int(n_units/n_in), axis=-1)
        self.map = Map(f)
    """
    def get_n_in(self):
        return self.n_in

    def set_n_in(self, n_in):
        self.n_in = n_in

    def get_n_out(self):
        return self.n_out

    def set_n_out(self, n_out):
        self.n_out = n_out
    """

    def __call__(self, x):
        x = Dense(units=self.n_units, activation='relu')(x)
        y = self.split(x)
        y = self.map(y)
        return K.concatenate(y, axis=-1)


def gcnn_resnet_layer(inputs,
                      contributors,
                      weights,
                      angles,
                      n_v,
                      n_rings=3,
                      n_dirs=16,
                      num_filters=16,
                      sync_mode='radial_sync',
                      take_max=False,
                      activation='relu',
                      batch_normalization=True,
                      conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    # ang_reg = 1e-2
    ang_reg = 0.
    conv = GeodesicConv(nfilters=num_filters,
                        nv=n_v,
                        ndirs=n_dirs,
                        nrings=n_rings,
                        sync_mode=sync_mode,
                        take_max=False,
                        activation=None,
                        kernel_initializer='he_normal',
                        kernel_regularizer=L1L2_circ(l1=0., l2=1e-4, l1_d=ang_reg, l2_d=0.))


    x = inputs
    if conv_first:
        x = conv([x, contributors, weights, angles])
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv([x, contributors, weights, angles])
    if take_max:
        x = AngularMaxPooling(r=1, take_max=True)(x)
    return x

def quadratic_layer(inputs,
                    units,
                 slices,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None):

    x = []
    if type(inputs) == type(str()):
        if len(inputs) == 2:
            x.append(inputs[0])
            x.append(inputs[1])
        elif len(inputs) == 1:
            x.append(inputs[0])
            x.append(inputs[0])
        else:
            raise ValueError('Unexpected number of inputs')
    else:
        x.append(inputs)
        x.append(inputs)

    y0 = DenseStack(units=units,
                    slices=slices,
                    activation=None,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(x[0])

    y1 = DenseStack(units=units,
                    slices=slices,
                    activation=None,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(x[1])

    n = K.ndim(y0)
    if n == 4:
        y = tf.einsum('ijkl,ijkl->ijk', y0, y1)
    elif n == 5:
        y = tf.einsum('ijklm,ijklm->ijkl', y0, y1)
    else:
        raise ValueError('Unexpected dimension')

    return y


