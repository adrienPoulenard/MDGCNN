"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from custom_regularizers import L1L2_circ
from keras.engine.topology import Layer
from conv import GeodesicConvImg
from pooling import AngularPooling2d
import math
from plt_history import plt_history
import os
import ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Training parameters
batch_size = 20  # orig paper trained all networks with batch_size=128
epochs = 50

num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

# patch op
nrings = [2, 2, 2]
ndirs_ = 8
n_train_dirs_ = 8
ndirs = [ndirs_, ndirs_, ndirs_]
radius = 1.8
# pool_ratio = math.sqrt(2.)
pool_ratio = 2.
# sync = 'async'
sync = 'radial_sync'
random_perturb_ = 0.8/6.0



# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(np.shape(y_train))

"""
x_train = x_train[0:10000, :, :, :]
y_train = y_train[0:10000, :]
x_test = x_test[0:2000, :, :, :]
y_test = y_test[0:2000, :]
"""

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def resnet_layer(inputs,
                 nbatch,
                 shape,
                 n_rings,
                 radius,
                 n_dirs,
                 n_train_dirs,
                 num_filters=16,
                 strides=(1, 1),
                 sync_mode='radial_sync',
                 take_max=False,
                 pool=False,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 random_perturb=0.0):
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

    ang_reg = 0.0
    conv = GeodesicConvImg(nbatch=nbatch,
                           nfilters=num_filters,
                           ndirs=n_dirs,
                           ntraindirs=n_train_dirs,
                           nrings=n_rings,
                           radius=radius,
                           shape=shape,
                           strides=strides,
                           sync_mode=sync_mode,
                           pool=pool,
                           take_max=False,
                           activation=None,
                           kernel_initializer='he_normal',
                           kernel_regularizer=L1L2_circ(l1=0., l2=1e-4, l1_d=ang_reg, l2_d=0.),
                           random_perturb=random_perturb)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    if take_max:
        x = AngularPooling2d(r=1, full=True)(x)
    return x


def resnet_v1(input_shape,
              nbatch,
              n_rings,
              radius,
              pool_ratio,
              n_dirs,
              n_train_dirs,
              nstacks=None,
              nresblocks_per_stack=2,
              nfilters=16,
              sync_mode='radial_sync',
              num_classes=10,
              random_perturb=0.0):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """

    bn = False
    pool_ = True
    if nstacks is None:
        nstacks = len(n_rings)

    take_max = False
    if sync_mode is 'async':
        take_max = True

    num_filters = nfilters
    im_shape = (input_shape[0], input_shape[1])
    new_im_shape = im_shape

    inputs = Input(shape=input_shape)
    # x = resnet_layer(inputs=inputs)
    x = resnet_layer(inputs=inputs,
                     nbatch=nbatch,
                     shape=im_shape,
                     n_rings=n_rings[0],
                     radius=radius,
                     n_dirs=n_dirs[0],
                     n_train_dirs=n_train_dirs,
                     num_filters=num_filters,
                     strides=(1, 1),
                     sync_mode=sync_mode,
                     take_max=take_max,
                     batch_normalization=bn,
                     random_perturb=random_perturb)
    # Instantiate the stack of residual units
    for stack in range(nstacks):
        for res_block in range(nresblocks_per_stack):
            strides = (1, 1)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = (pool_ratio, pool_ratio)  # downsample
                new_im_shape = (int(math.ceil(im_shape[0]/pool_ratio)), int(math.ceil(im_shape[0]/pool_ratio)))
            y = resnet_layer(inputs=x,
                             nbatch=nbatch,
                             shape=im_shape,
                             n_rings=n_rings[stack],
                             radius=radius,
                             n_dirs=n_dirs[stack],
                             n_train_dirs=n_train_dirs,
                             num_filters=num_filters,
                             strides=strides,
                             sync_mode=sync_mode,
                             take_max=take_max,
                             batch_normalization=bn,
                             random_perturb=random_perturb)
            y = resnet_layer(inputs=y,
                             nbatch=nbatch,
                             shape=new_im_shape,
                             n_rings=n_rings[stack],
                             radius=radius,
                             n_dirs=n_dirs[stack],
                             n_train_dirs=n_train_dirs,
                             num_filters=num_filters,
                             strides=(1, 1),
                             sync_mode=sync_mode,
                             take_max=take_max,
                             activation=None,
                             batch_normalization=bn,
                             random_perturb=random_perturb)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 nbatch=nbatch,
                                 shape=im_shape,
                                 n_rings=n_rings[stack],
                                 radius=0.,
                                 n_dirs=n_dirs[stack],
                                 n_train_dirs=n_train_dirs,
                                 num_filters=num_filters,
                                 strides=strides,
                                 sync_mode=sync_mode,
                                 take_max=take_max,
                                 pool=True,
                                 activation=None,
                                 batch_normalization=False,
                                 random_perturb=random_perturb)
                im_shape = new_im_shape
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = int(math.ceil(num_filters*pool_ratio))

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AngularPooling2d(r=1, pool='max', full=True)(x)

    x = AveragePooling2D(pool_size=im_shape)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v1(input_shape=input_shape,
                  nbatch=batch_size,
                  n_rings=nrings,
                  radius=radius,
                  pool_ratio=pool_ratio,
                  n_dirs=ndirs,
                  n_train_dirs=n_train_dirs_,
                  nstacks=3,
                  nresblocks_per_stack=1,
                  nfilters=16,
                  sync_mode=sync,
                  num_classes=10,
                  random_perturb=random_perturb_)


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# plot training history
plt_history(history=history)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])




