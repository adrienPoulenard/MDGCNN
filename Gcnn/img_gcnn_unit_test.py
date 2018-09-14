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
from conv import GeodesicConv, GeodesicConvImg
from pooling import AngularPooling2d
import os
from load_data import int_to_string, float_to_string, pool_op_ff, patch_op_ff
from matplotlib import pyplot as plt


nbatch = 2
# sync = 'async'
sync = 'radial_sync'


radius = [12.5, 12.5]
ratio = [1.000000, 0.500000]
nrings = [2, 2]
ndirs_ = 16
ndirs = [ndirs_, ndirs_]
input_res_w = 32
input_res_h = 32

# nv = [2000, 501, 126]
# nv = [2000, 2000, 2000]
# radius = [0.062500, 0.125000, 0.250000]
# ratio = [1.000000, 0.250000, 0.062500]


# create dirac data
nv_id = 0
dirac_idx = np.zeros(nbatch, dtype=np.int32)
dh = 2
dw = 2
dx = int(input_res_w / 2)
dy = int(input_res_h / 2)
dirac_idx[0] = 0
dirac_idx[1] = 750
dirac = np.zeros(shape=(nbatch, input_res_h, input_res_w, 3), dtype=np.float32)
for i in range(nbatch):
    for x in range(dw):
        for y in range(dh):
            dirac[1, x + dx, y + dy, 1] = 1.0

"""
dirac_dir = np.zeros(shape=(nbatch, input_res_h, input_res_w, ndirs[nv_id], 3), dtype=np.float32)
for i in range(nbatch):
    dirac_dir[i, dirac_idx[i], 10, 1] = 1.0

print(dirac_dir)
"""

"""
nv = [2000, 1001, 501]
radius = [0.093750, 0.132583, 0.187500]
ratio = [1.000000, 0.500000, 0.250000]
"""


def dirac_test(n_batch, h, w, n_rings, n_dirs,
               radius,
               sync_mode,
               idx,
               r=-1,
               dir=0,
               random_perturb=0.0):

    inputs = Input(shape=(h, w, 3), batch_shape=(n_batch,) + (h, w, 3))

    weights = []
    kernel = np.zeros(shape=(n_rings[idx], n_dirs[idx], 3, 3), dtype=np.float32)
    kernel[r, dir, 1, 1] = 1.0
    bias = np.zeros((3, ), dtype=np.float32)
    center = np.zeros(shape=(3, 3), dtype=np.float32)
    weights.append(kernel)
    weights.append(center)
    weights.append(bias)

    conv1 = GeodesicConvImg(nbatch=n_batch,
                           nfilters=3,
                           radius=radius[idx],
                           shape=(h, w),
                           ndirs=n_dirs[idx],
                           nrings=n_rings[idx],
                           sync_mode=sync_mode,
                           take_max=False,
                           activation=None,
                           weights=weights,
                            random_perturb=random_perturb)

    conv2 = GeodesicConvImg(nbatch=n_batch,
                           nfilters=3,
                           radius=radius[idx],
                           shape=(h, w),
                           ndirs=n_dirs[idx],
                           nrings=n_rings[idx],
                           sync_mode=sync_mode,
                           take_max=False,
                           activation=None,
                           weights=weights,
                            random_perturb=random_perturb)





    # ang_avg = AngularAveragePooling(r=1, take_average=True)
    ang_max = AngularPooling2d(r=1, pool='max', full=True)

    x = inputs

    x = conv1(x)

    # x = mesh_pool([x, tp[0]])

    x = conv2(x)

    x = ang_max(x)

    # x = ang_avg(x)

    # x = pool_dir(x)
    return Model(inputs=inputs, outputs=x)


model = dirac_test(n_batch=nbatch, h=input_res_h, w=input_res_w, n_rings=nrings, n_dirs=ndirs,
                   sync_mode=sync,
                   idx=nv_id,
                   radius=radius,
                   r=-1,
                   dir=0,
                   random_perturb=0.8/6.0)


"""
model.compile(loss=custom_losses.categorical_cross_entropy_bis,
              optimizer=optim,
              metrics=[custom_losses.categorical_accuracy_bis])
"""
model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(), metrics=[])

print('dirac')
print(np.shape(dirac))
res = model.predict(dirac, batch_size=nbatch, verbose=0)
res = res.astype(np.float32)

img = res[1, :, :, :]

plt.imshow(img)
plt.show()
