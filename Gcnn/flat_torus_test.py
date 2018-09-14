from load_data import load_patch_operator_as_triplets, load_mnist, load_mnist_test
import keras
from keras.layers import Input, Dense, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from custom_layers import SyncConvOrientedFixed, RepeatAxis, Max, twoInputsTest
import numpy as np

from models import SGCNN1_arch, GCNN0_arch, GCNN1_arch, SGCNN2_arch, GCNN2_arch

batch_size = 10
num_classes = 9
epochs = 1

# parameters

nv = 28*28
ndirs = 8
nrings = 2

# choose the number of samples
#ntrain = 1000
#ntest = 167
ntrain = 30000
ntest = 5000

# load the data
train_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/signals/flat_torus_28x28_training_30000.txt'
test_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/signals/flat_torus_28x28_test_5000.txt'
train_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/labels/training_labels_30000.txt'
test_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/labels/test_labels_5000.txt'

(x_train, y_train), (x_test, y_test) = load_mnist(train_signals_path,
                                                  test_signals_path,
                                                  train_labels_path,
                                                  test_labels_path,
                                                  nv, ntrain, ntest)

x_train = x_train.reshape(x_train.shape[0], nv, 1, 1)
x_test = x_test.reshape(x_test.shape[0], nv, 1, 1)
input_shape = (nv, 1, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(x_test.shape)


# load patch operator
patch_op_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/patch operators/flat_torus_28x28_ndir=8_nrings=2_rad=8x28_30_oriented=1.txt'


#patch_op_mat = load_patch_operator(patch_op_path, nv, ndirs, nrings)
patch_op_indices, patch_op_values, patch_op_shape = load_patch_operator_as_triplets(patch_op_path, nv, ndirs, nrings)

# setup architecture
data_input, predictions = SGCNN2_arch(nb_classes=num_classes, n_batch=batch_size, n_v=nv, n_dirs=ndirs, n_rings=nrings,
                                      indices=patch_op_indices, values=patch_op_values)

# create the model

model = Model(inputs=data_input, outputs=predictions)

# choose the optimizer
optim = keras.optimizers.Adam()
# optim = keras.optimizers.Adadelta()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optim,
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# test/evaluate
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print('Test loss same shape:', score[0])
print('Test accuracy same shape:', score[1])


def test_on_other_shape(pop_test_path, signal, labels, n_batch, n_v, n_dirs, n_rings):
    pop_test_indices, pop_test_values, pop_test_shape = load_patch_operator_as_triplets(pop_test_path, n_v, n_dirs,
                                                                                        n_rings)
    (x_val, y_val) = load_mnist_test(signal, labels, n_v, ntest)
    x_val = x_val.reshape(x_test.shape[0], n_v, 1, 1)
    # convert class vectors to binary class matrices
    y_val = keras.utils.to_categorical(y_val, num_classes)
    test_data, test_pred = SGCNN2_arch(nb_classes=num_classes, n_batch=n_batch, n_v=n_v, n_dirs=n_dirs, n_rings=n_rings,
                                       indices=pop_test_indices, values=pop_test_values)
    test_model = Model(inputs=test_data, outputs=test_pred)

    test_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=optim,
                       metrics=['accuracy'])

    test_model.set_weights(model.get_weights())

    my_score = test_model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    print('Test loss:', my_score[0])
    print('Test accuracy:', my_score[1])


print('test: ')
test_on_other_shape('C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/patch operators/flat_torus_42x42_ndir=8_nrings=2_rad=8x42_30_oriented=1.txt',
                    'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/signals/flat_torus_42x42_test_5000.txt',
                    'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/labels/test_labels_5000.txt',
                    n_batch=batch_size, n_v=42*42, n_dirs=ndirs, n_rings=nrings)

