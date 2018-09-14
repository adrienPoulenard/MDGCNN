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
# radius = 0.066666
# radius = 0.392699
radius = 0.314159
nlabels = 3
ndesc = 1
epochs = 100
nbatch = 25
nv = 2002
nrings = 3
ndirs = 16
ntest = 150

# set model
mymodel = models.shape_classification

# training params
loss = keras.losses.categorical_crossentropy
#loss = keras.losses.sparse_categorical_crossentropy
# loss = custom_losses.categorical_cross_entropy_bis
optim = keras.optimizers.Adam()
# optim = keras.optimizers.Adadelta()
metrics = ['accuracy']
# metrics = [custom_losses.categorical_accuracy_bis]

"""
train_txt = 'C:/Users/Adrien/Documents/Datasets/Faust/FAUST_train.txt'
test_txt = 'C:/Users/Adrien/Documents/Datasets/Faust/FAUST_test.txt'
dataset_path = 'C:/Users/Adrien/Documents/Datasets/Faust'
signal_path = 'C:/Users/Adrien/Documents/Datasets/Faust/matlab_descs_15wks'
connectivity_path = 'C:/Users/Adrien/Documents/Datasets/Faust/connectivity'
transport_path = 'C:/Users/Adrien/Documents/Datasets/Faust/transport'
coord3d_path = ''  # 'C:/Users/Adrien/Documents/Datasets/Faust/coords3d'
labels_path = 'C:/Users/Adrien/Documents/Datasets/Faust/labels_x=10_y=14_35'
"""


train_txt = 'C:/Users/Adrien/Documents/Datasets/star_spheres/train.txt'
test_txt = 'C:/Users/Adrien/Documents/Datasets/star_spheres/test.txt'
dataset_path = 'C:/Users/Adrien/Documents/Datasets/star_spheres'
signal_path = 'C:/Users/Adrien/Documents/Datasets/star_spheres/patterns_orth_signal'
connectivity_path = 'C:/Users/Adrien/Documents/Datasets/star_spheres/connectivity'
transport_path = 'C:/Users/Adrien/Documents/Datasets/star_spheres/transport'
coord3d_path = ''  # 'C:/Users/Adrien/Documents/Datasets/Faust/coords3d'
labels_path = 'C:/Users/Adrien/Documents/Datasets/star_spheres/patterns_orth_labels'


ds = classificationDataset(radius, nlabels, ndesc, 0, nv, nrings, ndirs,
                           train_txt=train_txt, test_txt=test_txt, dataset_path=dataset_path,
                           signals_path=signal_path, labels_path=labels_path)

# set inputs
train_inputs = [ds.get_train_signal(), ds.get_train_basepoints(), ds.get_train_local_frames(),
                ds.get_train_connectivity(), ds.get_train_transport()]
test_inputs = [ds.get_test_signal(), ds.get_test_basepoints(), ds.get_test_local_frames(),
               ds.get_test_connectivity(), ds.get_test_transport()]

print("rrrrrrrrr")
print(np.shape(ds.get_train_basepoints()))
print(np.shape(ds.get_train_local_frames()))
print(np.shape(ds.get_test_labels()))

# setup architecture
descs, input3d, input3dframes, inputExp, inputFrame, predictions = mymodel(nb_classes=nlabels, n_descs=ndesc,
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
model.fit(x=train_inputs, y=ds.get_train_labels(), batch_size=nbatch, epochs=epochs, verbose=1)


# test/evaluate
# setup architecture
descs_, input3d_, input3dframes_, inputExp_, inputFrame_, predictions_ = mymodel(nb_classes=nlabels, n_descs=ndesc,
                                                          n_batch=nbatch, n_v=nv, n_dirs=ndirs, n_rings=nrings,
                                                          shuffle_vertices=False)
# create the model
test_model = Model(inputs=[descs_, input3d_, input3dframes_, inputExp_, inputFrame_], outputs=predictions_)

test_model.compile(loss=loss, optimizer=optim, metrics=metrics)

test_model.set_weights(model.get_weights())


score = test_model.evaluate(test_inputs, ds.get_test_labels(), batch_size=nbatch, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predict = test_model.predict(test_inputs, batch_size=nbatch, verbose=0)


def get_confusion_matrix_one_hot(truth, model_results):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix_ = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    pred = np.argmax(model_results, axis=1)
    print(np.bincount(pred))
    assert len(pred) == truth.shape[0]
    truth = np.argmax(truth, axis=1)

    """
    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:, actual_class] == 1
        prediction_for_this_class = predictions_[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class == predicted_class)
            confusion_matrix_[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix_) == len(truth)
    assert np.sum(confusion_matrix_) == np.sum(truth)
    """
    confusion_matrix_ = confusion_matrix(truth, pred)
    return confusion_matrix_


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    """
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    """
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print(np.shape(ds.get_test_labels()))
print(np.shape(predict))

# Compute confusion matrix
y_true = np.reshape(ds.get_test_labels(), (ntest, nlabels))
y_pred = np.reshape(predict, (ntest, nlabels))

cnf_matrix = get_confusion_matrix_one_hot(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Confusion matrix, without normalization')
plt.show()

