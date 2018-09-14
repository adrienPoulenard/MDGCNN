import keras
from keras.models import Model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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


def plot_confusion_matrix_(cm, classes,
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
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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

"""
print(np.shape(ds.get_test_labels()))
print(np.shape(predict))

# Compute confusion matrix
y_true = np.reshape(ds.get_test_labels(), (ntest, nlabels))
y_pred = np.reshape(predict, (ntest, nlabels))
"""


def plot_confusion_mat(y_true, y_pred):
    cnf_matrix = get_confusion_matrix_one_hot(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, normalize=False,
                          title='Confusion matrix, without normalization')
    plt.show()


def plot_confusion_mat_(y_true, y_pred, classes, save_path=None):
    cnf_matrix = get_confusion_matrix_one_hot(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix_(cnf_matrix, classes=classes, normalize=False,
                          title='Confusion matrix, without normalization')
    plt.show()
    if save_path is not None:
        np.save(save_path, cnf_matrix)




