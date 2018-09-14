import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    nrings = 0
    ndirs = 0
    if K.ndim(x) == 5:
        nrings = shape[3]
        ndirs = shape[4]

    ndims = K.ndim(img)

    batch_idx = tf.range(0, batch_size)

    if nrings == 0 or ndirs == 0:
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))
        indices = tf.stack([b, y, x], 3)
    else:
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1, 1))
        b = tf.tile(batch_idx, (1, height, width, nrings, ndirs))
        if ndims == 4:
            indices = tf.stack([b, y, x], 5)
        else:
            t = tf.range(0, ndirs)
            t = tf.reshape(t, (ndirs, 1, 1, 1, 1))
            t = tf.transpose(t, perm=[4, 1, 2, 3, 0])
            t = tf.tile(t, (batch_size, height, width, nrings, 1))
            indices = tf.stack([b, y, x, t], 5)
    return tf.gather_nd(img, indices)

def get_vertex_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    nv = shape[1]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    b = tf.tile(batch_idx, (1, nv))
    indices = tf.stack([b, y, x], 2)

    return tf.gather_nd(img, indices)

def get_value(img, x, y):
    if K.ndim(x) == 2:
        return get_vertex_value(img, x, y)
    else:
        return get_pixel_value(img, x, y)


def bilinear_sampler(img, x, y, nrings, ndirs):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[-1]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    # x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_value(img, x0, y0)
    Ib = get_value(img, x0, y1)
    Ic = get_value(img, x1, y0)
    Id = get_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    """
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    """

    if K.ndim(img) == 5 and K.ndim(x) == 3:
        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        wa = K.repeat_elements(wa, rep=ndirs, axis=-1)
        wb = K.repeat_elements(wb, rep=ndirs, axis=-1)
        wc = K.repeat_elements(wc, rep=ndirs, axis=-1)
        wd = K.repeat_elements(wd, rep=ndirs, axis=-1)

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


class ImageSampling(Layer):

    def __init__(self, nbatch, uv=None, **kwargs):
        self.fixed = False
        self.nv = 0
        self.nbatch = nbatch
        if uv is not None:
            x = uv[:, 0]
            x = np.expand_dims(x, axis=0)
            x = np.repeat(x, repeats=self.nbatch, axis=0)
            y = uv[:, 1]
            y = np.expand_dims(y, axis=0)
            y = np.repeat(y, repeats=self.nbatch, axis=0)
            self.nv = np.shape(x)[1]
            self.x = tf.convert_to_tensor(x, dtype=tf.float32)
            self.y = tf.convert_to_tensor(y, dtype=tf.float32)
            self.fixed = True
        super(ImageSampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ImageSampling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.fixed:
            img = inputs
            return bilinear_sampler(img, self.x, self.y, 0, 0)
        else:
            img = inputs[0]
            uv = inputs[1]
            x = uv[:, :, 0]
            y = uv[:, :, 1]
        return bilinear_sampler(img, x, y, 0, 0)

    def compute_output_shape(self, input_shape):
        if self.fixed:
            return (input_shape[0], self.nv, input_shape[-1])
        else:
            return (input_shape[0][0], input_shape[0][1], input_shape[0][-1])


def window_interpolation_sync___(inputs, contributors, weights, angles):
    # max pooling
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    nv = shape[1]
    ndirs = shape[2]
    nrings = tf.shape(contributors)[2]

    a = tf.multiply(angles, tf.multiply(tf.cast(ndirs, 'float32'), (1./(2.*np.pi))))
    a = tf.floor(a)
    a = tf.cast(a, 'int32')

    dir_idx = tf.range(0, ndirs)
    dir_idx = tf.expand_dims(dir_idx, axis=-1)
    a = tf.add(a, dir_idx)
    a = tf.mod(a, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1, 1, 1))
    b = tf.tile(batch_idx, (1, nv, nrings, ndirs, 3))
    indices = tf.stack([b, contributors, a], -1)

    W = tf.gather_nd(inputs, indices)
    W = tf.reduce_max(W, axis=4)

    return W


def window_interpolation(inputs, contributors, weights, angles):
    # full interpolation
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    nv = shape[1]
    ndirs = shape[2]
    nrings = tf.shape(contributors)[2]

    a = tf.multiply(angles, tf.multiply(tf.cast(ndirs, 'float32'), (1. / (2. * np.pi))))

    a0 = tf.floor(a)
    aw1 = a - a0
    aw0 = -aw1 + 1.

    a0 = tf.cast(a0, 'int32')
    a1 = a0 + 1

    dir_idx = tf.range(0, ndirs)
    dir_idx = tf.expand_dims(dir_idx, axis=-1)
    a0 = tf.add(a0, dir_idx)
    a1 = tf.add(a1, dir_idx)
    a0 = tf.mod(a0, ndirs)
    a1 = tf.mod(a1, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1, 1, 1))
    b = tf.tile(batch_idx, (1, nv, nrings, ndirs, 3))
    indices0 = tf.stack([b, contributors, a0], -1)
    indices1 = tf.stack([b, contributors, a1], -1)

    W0 = tf.gather_nd(inputs, indices0)
    W1 = tf.gather_nd(inputs, indices1)

    aw0 = tf.expand_dims(aw0, axis=-1)
    aw1 = tf.expand_dims(aw1, axis=-1)
    W = tf.multiply(W0, aw0) + tf.multiply(W1, aw1)
    weights = tf.expand_dims(weights, axis=-1)
    W = tf.multiply(W, weights)
    W = tf.reduce_sum(W, axis=4)
    return W


def window_interpolation_sync(inputs, contributors, weights, angles):
    return window_interpolation(inputs, contributors, weights, angles)


def window_interpolation_sync_(inputs, contributors, weights, angles):
    # directional interpolation
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    nv = shape[1]
    ndirs = shape[2]
    nrings = tf.shape(contributors)[2]

    a = tf.multiply(angles[:, :, :, :, 0], tf.multiply(tf.cast(ndirs, 'float32'), (1./(2.*np.pi))))

    a0 = tf.floor(a)
    aw1 = a - a0
    aw0 = -aw1 + 1.

    a0 = tf.cast(a0, 'int32')
    a1 = a0 + 1

    dir_idx = tf.range(0, ndirs)
    a0 = tf.add(a0, dir_idx)
    a1 = tf.add(a1, dir_idx)
    a0 = tf.mod(a0, ndirs)
    a1 = tf.mod(a1, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1, 1))
    b = tf.tile(batch_idx, (1, nv, nrings, ndirs))
    c = contributors[:, :, :, :, 0]
    indices0 = tf.stack([b, c, a0], -1)
    indices1 = tf.stack([b, c, a1], -1)

    W0 = tf.gather_nd(inputs, indices0)
    W1 = tf.gather_nd(inputs, indices1)

    aw0 = tf.expand_dims(aw0, axis=-1)
    aw1 = tf.expand_dims(aw1, axis=-1)
    W = tf.multiply(W0, aw0) + tf.multiply(W1, aw1)
    return W


def window_interpolation_sync_u(inputs, contributors, weights, angles):
    # NN interpolation
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    nv = shape[1]
    ndirs = shape[2]
    nrings = tf.shape(contributors)[2]

    c = contributors[:, :, :, :, 0]

    a = angles[:, :, :, :, 0]
    a = tf.multiply(a, tf.multiply(tf.cast(ndirs, 'float32'), (1./(2.*np.pi))))
    a = tf.cast(a, 'int32')

    dir_idx = tf.range(0, ndirs)
    a = tf.add(a, dir_idx)
    a = tf.mod(a, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1, 1))
    b = tf.tile(batch_idx, (1, nv, nrings, ndirs))
    indices = tf.stack([b, c, a], -1)

    W = tf.gather_nd(inputs, indices)
    return W


def window_interpolation_async(inputs, contributors, weights):

    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    nv = shape[1]
    nrings = tf.shape(contributors)[2]
    ndirs = tf.shape(contributors)[3]

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1, 1, 1))
    b = tf.tile(batch_idx, (1, nv, nrings, ndirs, 3))
    indices = tf.stack([b, contributors], -1)

    W = tf.gather_nd(inputs, indices)

    weights = tf.expand_dims(weights, axis=-1)
    W = tf.multiply(W, weights)
    W = tf.reduce_sum(W, axis=4)
    return W


def downsample_mesh_sync(inputs, parents, angular_shifts):
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    new_nv = tf.shape(parents)[1]
    ndirs = shape[2]

    p = tf.expand_dims(parents, axis=-1)
    p = tf.tile(p, (1, 1, ndirs))

    a = tf.multiply(angular_shifts, tf.multiply(tf.cast(ndirs, 'float32'), (1. / (2. * np.pi))))
    a = tf.expand_dims(a, axis=-1)
    a = tf.tile(a, (1, 1, ndirs))

    a0 = tf.floor(a)
    aw1 = a - a0
    aw0 = -aw1 + 1.

    a0 = tf.cast(a0, 'int32')
    a1 = a0 + 1

    dir_idx = tf.range(0, ndirs)
    # dir_idx = tf.expand_dims(dir_idx, axis=-1)
    a0 = tf.add(a0, dir_idx)
    a1 = tf.add(a1, dir_idx)
    a0 = tf.mod(a0, ndirs)
    a1 = tf.mod(a1, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, new_nv, ndirs))
    indices0 = tf.stack([b, p, a0], -1)
    indices1 = tf.stack([b, p, a1], -1)

    out0 = tf.gather_nd(inputs, indices0)
    out1 = tf.gather_nd(inputs, indices1)

    aw0 = tf.expand_dims(aw0, axis=-1)
    aw1 = tf.expand_dims(aw1, axis=-1)

    out = tf.multiply(out0, aw0) + tf.multiply(out1, aw1)
    return out


def downsample_mesh_synco(inputs, parents, angular_shifts):
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    new_nv = tf.shape(parents)[1]
    ndirs = shape[2]

    p = tf.expand_dims(parents, axis=-1)
    p = tf.tile(p, (1, 1, ndirs))

    a = tf.multiply(angular_shifts, tf.multiply(tf.cast(ndirs, 'float32'), (1. / (2. * np.pi))))
    a = tf.expand_dims(a, axis=-1)
    a = tf.tile(a, (1, 1, ndirs))

    a = tf.cast(a, 'int32')

    dir_idx = tf.range(0, ndirs)
    # dir_idx = tf.expand_dims(dir_idx, axis=-1)
    a = tf.add(a, dir_idx)
    a = tf.mod(a, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, new_nv, ndirs))
    indices = tf.stack([b, p, a], -1)

    out = tf.gather_nd(inputs, indices)
    return out


def downsample_mesh_async(inputs, parents):
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    new_nv = tf.shape(parents)[1]

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1))
    b = tf.tile(batch_idx, (1, new_nv))
    indices = tf.stack([b, parents], -1)

    out = tf.gather_nd(inputs, indices)

    return out


def upsample_mesh_async(inputs, parents, new_nv):
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    old_nv = shape[1]

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1))
    b = tf.tile(batch_idx, (1, old_nv))
    indices = tf.stack([b, parents], -1)
    shape = tf.stack([B, new_nv, C])
    out = tf.scatter_nd(indices=indices, updates=inputs, shape=shape)
    return out


def upsample_mesh_sync(inputs, parents, angular_shifts, new_nv, new_ndirs):
    shape = tf.shape(inputs)
    C = shape[-1]
    B = shape[0]
    old_nv = shape[1]
    ndirs = new_ndirs

    p = tf.expand_dims(parents, axis=-1)
    p = tf.tile(p, (1, 1, ndirs))

    a = tf.multiply(angular_shifts, tf.multiply(tf.cast(ndirs, 'float32'), (-1. / (2. * np.pi))))
    a = tf.expand_dims(a, axis=-1)
    a = tf.tile(a, (1, 1, ndirs))

    a0 = tf.floor(a)
    aw1 = a - a0
    aw0 = -aw1 + 1.

    a0 = tf.cast(a0, 'int32')
    a1 = a0 + 1

    dir_idx = tf.range(0, ndirs)
    # dir_idx = tf.expand_dims(dir_idx, axis=-1)
    a0 = tf.add(a0, dir_idx)
    a1 = tf.add(a1, dir_idx)
    a0 = tf.mod(a0, ndirs)
    a1 = tf.mod(a1, ndirs)

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, old_nv, ndirs))
    indices0 = tf.stack([b, p, a0], -1)
    indices1 = tf.stack([b, p, a1], -1)

    print(B)
    print(new_nv)
    print(C)
    print(K.ndim(B))
    print(K.ndim(C))
    print(K.ndim(shape))
    print(K.shape(parents))
    shape = tf.stack([B, new_nv, new_ndirs, C])

    out0 = tf.scatter_nd(indices=indices0, updates=inputs, shape=shape)
    out1 = tf.scatter_nd(indices=indices1, updates=inputs, shape=shape)

    shape = tf.stack([B, new_nv, new_ndirs])
    aw0 = tf.scatter_nd(indices=indices0, updates=aw0, shape=shape)
    aw1 = tf.scatter_nd(indices=indices1, updates=aw1, shape=shape)

    aw0 = tf.expand_dims(aw0, axis=-1)
    aw1 = tf.expand_dims(aw1, axis=-1)

    out = tf.multiply(out0, aw0) + tf.multiply(out1, aw1)
    return out