from load_data import classificationDataset
import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras import backend as K

from custom_layers import Input3d, SyncConvBis, AsyncConvBis, SyncGeodesicConv

from pooling import AngularMaxPooling, MaxPooling, PoolFrameBundleFixed, PoolSurfaceFixed, Pooling, PoolDir, AngularAveragePooling
from pooling import SelectDir
from conv import GeodesicConv
from patch_operators import GeodesicFieldFixed, ExpMapFixed

import numpy as np
import models
from models import SGCNN_3D
import custom_losses
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from load_data import patch_op_ff, load_dense_tensor, load_dense_matrix
from conv import GeodesicConv
import os
from load_data import int_to_string, float_to_string, pool_op_ff, patch_op_ff, load_fixed_patch_op
from sampling import ImageSampling
from keras.datasets import cifar10
from custom_layers import ConstantTensor






"""

def dirac_test(dataset_path,
               name,
               ratio,
               radius,
               n_batch, n_v, n_rings, n_dirs,
               sync_mode,
               r=-1,
               dir=0):


    # create dirac data
    nv_id = 0
    dirac_idx = np.zeros(n_batch, dtype=np.int32)
    dirac_idx[0] = 39*19 + 19
    dirac_idx[1] = 39*19 + 19
    dirac = np.zeros(shape=(n_batch, n_v[nv_id], 1), dtype=np.float32)
    for i in range(n_batch):
        dirac[i, dirac_idx[i], 0] = 1.0
    dirac_dir = np.zeros(shape=(n_batch, n_v[nv_id], n_dirs[nv_id], 1), dtype=np.float32)
    for i in range(n_batch):
        dirac_dir[i, dirac_idx[i], 1, 0] = 1.0

    print(dirac_dir)


    contributors, weights, angles, parents, angular_shifts = load_fixed_patch_op(dataset_path=dataset_path,
                                                                                 name=name,
                                                                                 ratio=ratio,
                                                                                 radius=radius,
                                                                                 nv=n_v,
                                                                                 nrings=n_rings,
                                                                                 ndirs=n_dirs)

    inputs = Input(shape=(n_v[0], n_dirs[0], 1), batch_shape=(n_batch,) + (n_v[0], n_dirs[0], 1))

    # patch operator
    C = []
    W = []
    TA = []

    P = []
    AS = []

    for i in range(len(n_v) - 1):
        # pool_op = PoolingOperatorFixed(parents=parents[i], angular_shifts=angular_shifts[i], batch_size=n_batch)
        P.append(ConstantTensor(const=parents[i], batch_size=n_batch, dtype='int')([]))
        AS.append(ConstantTensor(const=angular_shifts[i], batch_size=n_batch, dtype='float')([]))

    for stack in range(len(n_v)):
        # patch_op = PatchOperatorFixed(contributors=contributors[stack],
        #                              weights=weights[stack],
        #                              angles=angles[stack])
        C.append(ConstantTensor(const=contributors[stack], batch_size=n_batch, dtype='int')([]))
        W.append(ConstantTensor(const=weights[stack], batch_size=n_batch, dtype='float')([]))
        TA.append(ConstantTensor(const=angles[stack], batch_size=n_batch, dtype='float')([]))

    k_weights = []
    kernel = np.zeros(shape=(n_rings[0], n_dirs[0], 1, 1), dtype=np.float32)
    kernel[r, dir, 0, 0] = 1.0
    bias = np.zeros((1, ), dtype=np.float32)
    center = np.zeros(shape=(1, 1), dtype=np.float32)
    k_weights.append(kernel)
    k_weights.append(center)
    k_weights.append(bias)

    conv1 = GeodesicConv(nfilters=1,
                 nv=n_v[0],
                 ndirs=n_dirs[0],
                 nrings=n_rings[0],
                 sync_mode=sync_mode,
                 take_max=False,
                 activation=None,
                 weights=k_weights)

    conv2 = GeodesicConv(nfilters=1,
                 nv=n_v[1],
                 ndirs=n_dirs[1],
                 nrings=n_rings[1],
                 sync_mode=sync_mode,
                 take_max=False,
                 activation=None,
                 weights=k_weights)

    max_pool = MaxPooling(r=3, take_max=False)

    mesh_pool = Pooling()

    ang_max = AngularMaxPooling(r=1, take_max=True)
    ang_max_partial = AngularMaxPooling(r=7, take_max=False)

    x = inputs
    x = ang_max_partial(x)
    # x = max_pool([x, t[0]])

    # x = select_dir(x)

    x = conv1([x, C[0], W[0], TA[0]])

    # x = mesh_pool([x, P[0], AS[0]])

    x = conv1([x, C[0], W[0], TA[0]])

    x = ang_max(x)

    # x = ang_avg(x)

    # x = pool_dir(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(), metrics=[])

    res = model.predict(dirac_dir, batch_size=n_batch, verbose=0)
    res = res.astype(np.float32)
    # res = np.reshape(res, (nv*3))
    save_dir = 'C:/Users/Adrien/Documents/Keras/Gcnn/unit tests'
    for j in range(1):
        res_ = res[j, :, :]
        # res_ = res_.squeeze(axis=0)
        np.savetxt(save_dir + '/dirac_' + int_to_string(j) + '_' + name + '.txt', res_, fmt='%f')


dirac_test(dataset_path='C:/Users/Adrien/Documents/Datasets/grid_mesh',
           name='grid_38x38', ratio=[1.000000, 0.500000], radius=[6.000000, 12.00000],
           n_batch=2, n_v=[1521, 400], n_rings=[4, 4], n_dirs=[8, 8],
           sync_mode='radial_sync',
           r=-1,
           dir=0)

"""



def uv_test(i, dataset, name, n_v):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    x_train = x_train[i, :, :, :]
    x_train = np.expand_dims(x_train, axis=0)

    uv_path = dataset + '/uv_coordinates'
    uv = load_dense_matrix(uv_path + '/' + name + '.txt', d_type=np.float32)
    uv = np.reshape(uv, newshape=(n_v, 2))
    inputs = Input(shape=(32, 32, 3), batch_shape=(1,) + (32, 32, 3))
    x = ImageSampling(nbatch=1, uv=uv)(inputs)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(), metrics=[])
    res = model.predict(x_train, batch_size=1, verbose=0)
    res = res.astype(np.float32)
    # res = np.reshape(res, (nv*3))
    save_dir = 'E:/Users/Adrien/Documents/Keras/Gcnn/unit tests'
    for j in range(1):
        res_ = res[j, :, :]
        res_ = res_.flatten()
        # res_ = res_.squeeze(axis=0)
        np.savetxt(save_dir + '/cifar10_' + int_to_string(i) + '_' + name + '.txt', res_, fmt='%f')


uv_test(2355, 'E:/Users/Adrien/Documents/Datasets/sphere', 'sphere_3002_w=32', 3002)








