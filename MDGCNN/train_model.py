from load_data import classificationDataset
import keras
from keras.models import Model
import numpy as np
import models
from models import SGCNN_3D
import custom_losses
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from load_data import load_fixed_patch_op
from load_data import load_patch_op, load_labels, load_descriptors
from plt_history import plt_history
from confusion_mat import plot_confusion_mat_
from rotation_generator import RandomShapeRotationsGenerator
from models import gcnn_resnet_v1
import os
from load_data import int_to_string


def shape_dataset_segmentation(train_txt,
                               test_txt,
                               patch_op_path,
                               desc_path,
                               input_dim,
                               nclasses,
                               labels_path,
                               radius,
                               nbatch,
                               nv,
                               nrings,
                               ndirs,
                               ratio,
                               nepochs,
                               generator=None,
                               classes=None,
                               save_dir=None,
                               model_name='model'):
    if model_name is 'async':
        sync_mode = 'async'
    else:
        sync_mode = 'radial_sync'

    # create model

    model = gcnn_resnet_v1(n_batch=nbatch,
                           ratio=ratio,
                           n_v=nv,
                           n_rings=nrings,
                           n_dirs=ndirs,
                           fixed_patch_op=False,
                           contributors=None,
                           weights=None,
                           angles=None,
                           parents=None,
                           angular_shifts=None,
                           batch_norm=False,
                           uv=None,
                           input_dim=input_dim,
                           nstacks=1,
                           nresblocks_per_stack=2,
                           nfilters=16,
                           sync_mode=sync_mode,
                           num_classes=nclasses)

    # load patch op
    train_c, train_w, train_t_a, train_p, train_a_s = load_patch_op(shapes_names_txt=train_txt,
                                                                    shapes_nv=nv,
                                                                    radius=radius,
                                                                    nrings=nrings,
                                                                    ndirs=ndirs,
                                                                    ratio=ratio,
                                                                    dataset_path=patch_op_path)

    test_c, test_w, test_t_a, test_p, test_a_s = load_patch_op(shapes_names_txt=test_txt,
                                                               shapes_nv=nv,
                                                               radius=radius,
                                                               nrings=nrings,
                                                               ndirs=ndirs,
                                                               ratio=ratio,
                                                               dataset_path=patch_op_path)

    # load signal

    train_desc = load_descriptors(train_txt, desc_path)
    n_train_samples = train_desc.shape[0]

    test_desc = load_descriptors(test_txt, desc_path)
    n_test_samples = test_desc.shape[0]

    # load labels

    y_train = load_labels(train_txt, labels_path, nclasses)
    y_test = load_labels(test_txt, labels_path, nclasses)

    x_train = [train_desc]
    x_test = [test_desc]

    input_names = ['input_signal']
    for j in range(len(train_c)):
        input_names.append('contributors_' + int_to_string(j))
        input_names.append('weights_' + int_to_string(j))
        input_names.append('transport_' + int_to_string(j))
    for j in range(len(train_p)):
        input_names.append('parents_' + int_to_string(j))
        input_names.append('angular_shifts_' + int_to_string(j))

    for j in range(len(train_c)):
        x_train.append(train_c[j])
        x_train.append(train_w[j])
        x_train.append(train_t_a[j])

        x_test.append(test_c[j])
        x_test.append(test_w[j])
        x_test.append(test_t_a[j])

    for j in range(len(train_p)):
        x_train.append(train_p[j])
        x_train.append(train_a_s[j])

        x_test.append(test_p[j])
        x_test.append(test_a_s[j])

    print('shapes !!!')
    for x_ in x_test:
        print(np.shape(x_))

    x_train = dict(zip(input_names, x_train))
    x_test = dict(zip(input_names, x_test))

    # train model

    opt = 'adam'

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    if generator is None:
        history = model.fit(x_train, y_train,
                            batch_size=nbatch,
                            epochs=nepochs,
                            validation_data=(x_test, y_test),
                            shuffle=True)
    else:
        training_generator = generator(x_train, y_train, nbatch, nv, n_classes=nclasses, shuffle=True)
        test_generator = generator(x_test, y_test, nbatch, nv, n_classes=nclasses, shuffle=True)

        history = model.fit_generator(generator=training_generator,
                                      steps_per_epoch=n_train_samples/nbatch,
                                      epochs=nepochs,
                                      validation_data=test_generator,
                                      validation_steps=1,
                                      use_multiprocessing=False,
                                      workers=1)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1, batch_size=nbatch)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    if save_dir is not None:
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

        # plot_confusion_mat_(y_true=y_test, y_pred=y_pred, classes=classes,
        #                     save_path=os.path.join(save_dir, model_name + '_conf_mat'))
        plt_history(history=history, save_path=os.path.join(save_dir, model_name + '_history'))
    else:
        plt_history(history=history, save_path=None)
