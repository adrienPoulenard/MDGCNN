from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

import numpy
import pickle

"""
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""


def plt_history(history, save_path=None):
    # list all data in history
    print(history.keys())
    print('history_acc')
    print(history['acc'])
    print('history_val_acc')
    print(history['val_acc'])
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    if save_path is not None:
        with open(save_path, 'wb') as file_pi:
            pickle.dump(history, file_pi)


def plot_histories(histories, plt_train=False):
    methods = list(histories.keys())
    print(methods)
    for i in range(len(methods)):
        # summarize history for accuracy
        # plt.plot(histories[methods[i]]['acc'])
        print(histories[methods[i]]['val_acc'])
        plt.plot(histories[methods[i]]['val_acc'])
        plt.title('validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

    plt.legend(methods, loc='lower right')
    plt.show()

