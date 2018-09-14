from load_data import classificationDataset
import keras
from keras.models import Model
import numpy as np
import models
from models import SGCNN_3D
import custom_losses
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from models import gcnn_resnet_v1
from load_data import load_dense_tensor, load_dense_matrix, patch_op_ff, pool_op_ff
from load_data import int_to_string, float_to_string
from keras.datasets import cifar10
import os
import time
from plt_history import plt_history
from confusion_mat import plot_confusion_mat_

classes = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

subtract_pixel_mean = False
batch_norm = False

drive = 'C'
save_dir = drive + ':/Users/adrien/Documents/Keras/Gcnn/results'

name = 'sphere_3002'
epochs = 50
nbatch = 10

# sphere
nv = [3002, 752, 189]
radius = [0.176715, 0.353429, 0.706858]
# radius = [0.147262, 0.294524, 0.589049]
# radius = [0.245437, 2.*0.245437, 4.*0.245437]
ratio = [1.000000, 0.250000, 0.062500]


# grid
"""
nv = [1521, 400, 100]
radius = [1.800000, 3.600000, 7.200000]
ratio = [1.000000, 0.500000, 0.250000]
"""


"""
nv = [2000, 1001, 501]
radius = [0.093750, 0.132583, 0.187500]
ratio = [1.000000, 0.500000, 0.250000]
"""

ndirs_ = 8

nrings = [2, 2, 2]
ndirs = [ndirs_, ndirs_, ndirs_]
nclasses = 10
sync = 'radial_sync'
# sync = 'async'

model_name = name + '_' + sync + '_resnet'
#
dataset_path = drive + ':/Users/adrien/Documents/Datasets/sphere'

# load data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, nclasses)
y_test = keras.utils.to_categorical(y_test, nclasses)

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean


# load patch operators
print('loading patch operators')
# r = [0.161988, 0.323977, 0.647953]
# r = [0.323977, 0.647953]

# r = [0.176715, 0.353429]
contributors = []
weights = []
angles = []
for i in range(len(radius)):
    name_ = name + '_ratio=' + float_to_string(ratio[i]) + '_nv=' + int_to_string(nv[i])
    c_path = os.path.join(dataset_path + '/bin_contributors', patch_op_ff(name_, radius[i], nrings[i], ndirs[i]))
    contributors.append(load_dense_tensor(c_path, shape=(nv[i], nrings[i], ndirs[i], 3), d_type=np.int32))

    w_path = os.path.join(dataset_path + '/contributors_weights', patch_op_ff(name_, radius[i], nrings[i], ndirs[i]))
    weights.append(load_dense_tensor(w_path, (nv[i], nrings[i], ndirs[i], 3), d_type=np.float32))

    a_path = os.path.join(dataset_path + '/transported_angles', patch_op_ff(name_, radius[i], nrings[i], ndirs[i]))
    angles.append(load_dense_tensor(a_path, (nv[i], nrings[i], ndirs[i], 3), d_type=np.float32))

print('loading pooling operators')

angular_shifts = []
parent_vertices = []

for i in range(len(ratio)-1):
    a_path = os.path.join(dataset_path + '/angular_shifts',
                          pool_op_ff(name, ratio[i], ratio[i + 1], nv[i + 1]))
    angular_shifts.append(load_dense_matrix(a_path, d_type=np.float32))
    p_path = os.path.join(dataset_path + '/parent_vertices',
                          pool_op_ff(name, ratio[i], ratio[i + 1], nv[i + 1]))
    parent_vertices.append(load_dense_matrix(p_path, d_type=np.int32))

# loading uv coodrinates
uv = load_dense_matrix(dataset_path + '/uv_coordinates/' + name + '.txt', d_type=np.float32)
uv = np.reshape(uv, newshape=(nv[0], 2))

"""
model = simple_cifar10_test(n_batch=nbatch, n_v=nv, n_rings=nrings, n_dirs=ndirs,
                            exp_map=exp_map,
                            transport=transport,
                            pool_points=parent_vertices,
                            pool_frames=angular_shift,
                            sync=sync,
                            num_classes=nclasses)
"""

model = gcnn_resnet_v1(n_batch=nbatch, ratio=ratio, n_v=nv, n_rings=nrings, n_dirs=ndirs,
                       fixed_patch_op=True,
                       contributors=contributors,
                       weights=weights,
                       angles=angles,
                       parents=parent_vertices,
                       angular_shifts=angular_shifts,
                       batch_norm=batch_norm,
                       uv=uv,
                       nstacks=3,
                       nresblocks_per_stack=1,
                       nfilters=16,
                       sync_mode=sync,
                       num_classes=nclasses)


# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = 'adam'

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255.
# x_test /= 255.

history = model.fit(x_train, y_train,
                    batch_size=nbatch,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    verbose=1)
# plot training history

plt_history(history=history, save_path=os.path.join(save_dir, model_name + '_history'))

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1, batch_size=nbatch)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
weights_path = os.path.join(save_dir, model_name + '_weights.h5')

# Option 1: Save Weights + Architecture
model.save_weights(weights_path)
with open(model_path + '.json', 'w') as f:
    f.write(model.to_json())
model.save(model_path + '.h5')

print('Saved trained model at %s ' % model_path)

# confusion matrix

# plot confusion matrix
y_pred = model.predict(x_test, batch_size=nbatch, verbose=0)
plot_confusion_mat_(y_true=y_test, y_pred=y_pred, classes=classes,
                    save_path=os.path.join(save_dir, model_name + '_conf_mat'))




