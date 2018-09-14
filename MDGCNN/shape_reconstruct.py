from load_data import classificationDataset
import keras
from keras.models import Model
import numpy as np
import models
from models import SGCNN_3D
import custom_losses
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# load the data
# radius = 1.047198
# radius = 0.523599
radius = 0.100000

ndesc = 15
epochs = 150
nbatch = 1
nv = 6890
nrings = 3
ndirs = 16
ntest = 1
nlabels = 0
ntarsignals = 3

# set model
mymodel = models.test_reconstruct

# training params
# loss = keras.losses.mean_squared_error
loss = keras.losses.mean_absolute_error
optim = keras.optimizers.Adam()
# optim = keras.optimizers.Adadelta()
metrics = []
# metrics = [custom_losses.categorical_accuracy_bis]

"""
train_txt = 'C:/Users/Adrien/Documents/Datasets/spheres/train.txt'
test_txt = 'C:/Users/Adrien/Documents/Datasets/spheres/test.txt'
dataset_path = 'C:/Users/Adrien/Documents/Datasets/spheres'
signal_path = 'C:/Users/Adrien/Documents/Datasets/spheres/signals'
target_signal_path = 'C:/Users/Adrien/Documents/Datasets/spheres/basepoints'
connectivity_path = 'C:/Users/Adrien/Documents/Datasets/spheres/connectivity'
transport_path = 'C:/Users/Adrien/Documents/Datasets/spheres/transport'
"""

"""
train_txt = 'C:/Users/Adrien/Documents/Datasets/Faust/Faust_train.txt'
test_txt = 'C:/Users/Adrien/Documents/Datasets/Faust/Faust_test1.txt'
dataset_path = 'C:/Users/Adrien/Documents/Datasets/Faust'
signal_path = 'C:/Users/Adrien/Documents/Datasets/Faust/15wks_left_right'
target_signal_path = 'C:/Users/Adrien/Documents/Datasets/Faust/3d_reconstruct_70'
connectivity_path = 'C:/Users/Adrien/Documents/Datasets/Faust/connectivity'
transport_path = 'C:/Users/Adrien/Documents/Datasets/Faust/transport'
"""


train_txt = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/FAUST_train.txt'
test_txt = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/FAUST_test1.txt'
dataset_path = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted'
signal_path = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/signals'
target_signal_path = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/3d_reconstruct_70'
connectivity_path = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/connectivity'
transport_path = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/transport'
coord3d_path = ''  # 'C:/Users/Adrien/Documents/Datasets/Faust/coords3d'
labels_path = 'C:/Users/Adrien/Documents/Datasets/Faust non permuted/labels'


ds = classificationDataset(radius, nlabels, ndesc, ntarsignals, nv, nrings, ndirs,
                           train_txt, test_txt, dataset_path=dataset_path,
                           signals_path=signal_path, target_signal_path=target_signal_path)

# set inputs
train_inputs = [ds.get_train_signal(), ds.get_train_basepoints(), ds.get_train_local_frames(),
                ds.get_train_connectivity(), ds.get_train_transport()]
test_inputs = [ds.get_test_signal(), ds.get_test_basepoints(), ds.get_test_local_frames(),
               ds.get_test_connectivity(), ds.get_test_transport()]

print(np.shape(ds.get_train_basepoints()))
print(np.shape(ds.get_train_local_frames()))

# setup architecture
descs, input3d, input3dframes, inputExp, inputFrame, predictions = mymodel(nb_classes=ntarsignals, n_descs=ndesc,
                                                      n_batch=nbatch, n_v=nv, n_dirs=ndirs, n_rings=nrings,
                                                      shuffle_vertices=False)
# create the model
model = Model(inputs=[descs, input3d, input3dframes, inputExp, inputFrame], outputs=predictions)



"""
model.compile(loss=custom_losses.categorical_cross_entropy_bis,
              optimizer=optim,
              metrics=[custom_losses.categorical_accuracy_bis])
"""
model.compile(loss=loss, optimizer=optim, metrics=metrics)

# train the model
model.fit(x=train_inputs, y=ds.get_train_target_signal(), batch_size=nbatch, epochs=epochs, verbose=1)

res = model.predict(test_inputs, batch_size=1, verbose=0)
res = res.squeeze()
res = res.astype(np.float32)
res = np.reshape(res, (nv*3))
np.savetxt('C:/Users/Adrien/Documents/Datasets/Faust/reconstructed/reconstructed.txt', res, fmt='%f')
# res.tofile('C:/Users/Adrien/Documents/Datasets/spheres/reconstructed/reconstructed.txt')

print(np.shape(res))
print(res)
"""
# test/evaluate
# setup architecture
descs_, input3d_, input3dframes_, inputExp_, inputFrame_, predictions_ = mymodel(nb_classes=nlabels, n_descs=ndesc,
                                                          n_batch=1, n_v=nv, n_dirs=ndirs, n_rings=nrings,
                                                          shuffle_vertices=False)
# create the model
test_model = Model(inputs=[descs_, input3d_, input3dframes_, inputExp_, inputFrame_], outputs=predictions_)

test_model.compile(loss=loss, optimizer=optim, metrics=metrics)

test_model.set_weights(model.get_weights())


score = test_model.evaluate(test_inputs, ds.get_test_labels(), batch_size=1, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""