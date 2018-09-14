from load_data import load_patch_operator_as_triplets, load_mnist, load_mnist_test
import keras
from keras.layers import Input, Dense, Reshape, Dropout
from keras.models import Model
from keras import backend as K
# from custom_layers import SyncConv, RepeatAxis, Max
import numpy as np

from load_data import load_dense_matrix
import custom_losses

from models import GCNN1, SGCNN1, SGCNN2, GCNN2, SGCNN2_sparse2, GCNN2_sparse2


def main_training(batch_size, num_classes, epochs,
                  ntrain, ntest, train_signals_path, train_labels_path, train_path_c, train_path_t,
                  test_signals_path, test_labels_path, test_path_c, test_path_t,
                  val_signal_path,
                  train_nv, test_nv, nrings, ndirs):


    # choose the number of samples
    # ntrain = 1000
    # ntest = 167


    # load the data

    (x_train, y_train), (x_test, y_test) = load_mnist(train_signals_path,
                                                      test_signals_path,
                                                      train_labels_path,
                                                      test_labels_path,
                                                      train_nv, ntrain, ntest)

    #x_train = x_train.reshape(x_train.shape[0], train_nv, 1, 1)
    #x_test = x_test.reshape(x_test.shape[0], train_nv, 1, 1)
    x_train = x_train.reshape(x_train.shape[0], train_nv)
    x_train = np.expand_dims(x_train, axis=2)
    # x_train = np.expand_dims(x_train, axis=3)
    x_test = x_test.reshape(x_test.shape[0], train_nv)
    x_test = np.expand_dims(x_test, axis=2)
    # x_test = np.expand_dims(x_test, axis=3)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(x_train.shape)
    print(x_test.shape)

    # test data
    (x_val, y_val) = load_mnist_test(val_signal_path, test_labels_path, test_nv, ntest)
    x_val = x_val.reshape(x_test.shape[0], test_nv)
    x_val = np.expand_dims(x_val, axis=2)
    # x_val = np.expand_dims(x_val, axis=3)

    # convert class vectors to binary class matrices
    y_val = keras.utils.to_categorical(y_val, num_classes)


    # load patch operators
    train_connectivity = load_dense_matrix(train_path_c)
    train_transport = load_dense_matrix(train_path_t)
    test_connectivity = load_dense_matrix(test_path_c)
    test_transport = load_dense_matrix(test_path_t)

    # setup architecture
    data_input, predictions = GCNN2_sparse2(nb_classes=num_classes, n_batch=batch_size, n_v=train_nv, n_dirs=ndirs, n_rings=nrings,
                                     connectivity=train_connectivity, transport=train_transport)

    # create the model

    model = Model(inputs=data_input, outputs=predictions)

    # choose the optimizer
    optim = keras.optimizers.Adam()
    loss = keras.losses.categorical_crossentropy
    # optim = keras.optimizers.Adadelta()

    model.compile(loss=loss,
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
    predict = model.predict(x_test, batch_size=batch_size, verbose=0)


    def test_on_other_shape(connectivity_, transport_, signal, labels, n_batch, n_v, n_dirs, n_rings):

        test_data, test_pred = GCNN2_sparse2(nb_classes=num_classes, n_batch=n_batch, n_v=n_v, n_dirs=n_dirs, n_rings=n_rings,
                                      connectivity=connectivity_, transport=transport_)
        test_model = Model(inputs=test_data, outputs=test_pred)

        test_model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optim,
                           metrics=['accuracy'])

        test_model.set_weights(model.get_weights())

        my_score = test_model.evaluate(signal, labels, batch_size=n_batch, verbose=0)
        print('Test loss:', my_score[0])
        print('Test accuracy:', my_score[1])

    print('test: ')
    test_on_other_shape(connectivity_=test_connectivity, transport_=test_transport, signal=x_val, labels=y_val,
                        n_batch=batch_size, n_v=test_nv, n_dirs=ndirs, n_rings=nrings)

    return 0
