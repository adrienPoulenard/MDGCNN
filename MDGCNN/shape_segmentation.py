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
from models import gcnn_resnet_v1, GcnnUresNetSegmenter, GcnnResNetSegmenter, MlpSegmenter, GcnnUresNet
import os
from load_data import int_to_string
from data_gen import FullLabelizedDataset, ShapeMatchingFullDataset, ShapeMatchingFullDatasetMlp
from data_gen import FullHeterogeneousLabelizedDataset
from save_data import save_matrix
import time


def heterogeneous_dataset_segmentation(num_classes,
                                       num_filters,
                                       train_patch_op_paths,
                                       train_desc_paths,
                                       train_labels_paths,
                                       test_patch_op_paths,
                                       test_desc_paths,
                                       test_labels_paths,
                                       radius,
                                       nrings,
                                       ndirs,
                                       ratio,
                                       nepochs,
                                       sync_mode,
                                       batch_norm=True,
                                       global_3d=False):

    train_generator = FullHeterogeneousLabelizedDataset(num_classes=num_classes,
                                                        desc_paths=train_desc_paths,
                                                        labels_paths=train_labels_paths,
                                                        patch_op_paths=train_patch_op_paths,
                                                        radius=radius,
                                                        nrings=nrings,
                                                        ndirs=ndirs,
                                                        ratio=ratio,
                                                        shuffle=True,
                                                        add_noise=True)

    test_generator = FullHeterogeneousLabelizedDataset(num_classes=num_classes,
                                                       desc_paths=test_desc_paths,
                                                       labels_paths=test_labels_paths,
                                                       patch_op_paths=test_patch_op_paths,
                                                       radius=radius,
                                                       nrings=nrings,
                                                       ndirs=ndirs,
                                                       ratio=ratio,
                                                       shuffle=True,
                                                       add_noise=False)


    """
    model_ = GcnnUresNetSegmenter(n_batch=1, ratios=ratio, n_v=None, n_rings=nrings, n_dirs=ndirs,
                                  num_classes=num_classes,
                                  batch_norm=batch_norm,
                                  input_dim=train_generator.get_input_dim(),
                                  nstacks=2,
                                  nresblocks_per_stack=2,
                                  nfilters=num_filters,
                                  sync_mode=sync_mode,
                                  global_3d=False)
    """



    model_ = GcnnUresNet(n_batch=1, ratios=ratio, n_v=None, n_rings=nrings, n_dirs=ndirs,
                        task='segmentation',
                        output_channels=num_classes,
                        batch_norm=batch_norm,
                        input_dim=train_generator.get_input_dim(),
                        nstacks=2,
                        nresblocks_per_stack=2,
                        nfilters=num_filters,
                        sync_mode=sync_mode,
                        global_3d=False)


    model = model_.get_model()

    # train model

    opt = 'adam'

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/Users/adrien/Documents/Keras/Gcnn/tensorboard',
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.get_nsamples(),
                                  epochs=nepochs,
                                  validation_data=test_generator,
                                  validation_steps=test_generator.get_nsamples(),
                                  use_multiprocessing=False,
                                  workers=1,
                                  callbacks=[])

    return train_generator, test_generator, model, history


def save_correspondences(path, generator, model):
    names = generator.get_shapes_names()

    for i in range(len(names)):
        x = generator.get_input(i)
        y = model.predict(x, batch_size=1, verbose=0, steps=None)
        y_argmax = np.transpose(np.argmax(y, axis=-1), (1, 0))
        print('argmax')
        print(np.shape(y_argmax))
        # y_argmax = np.expand_dims(y_argmax, axis=1)
        # y_max = np.take(y, y_argmax, axis=1)
        y_max = np.transpose(np.amax(y, axis=-1), (1, 0))
        print('max')
        print(np.shape(y_max))
        # y_max = np.expand_dims(y_max, axis=1)
        name = os.path.join(path, names[i] + '_argmax.txt')
        save_matrix(name, y_argmax, dtype=np.int32)

        name = os.path.join(path, names[i] + '_max.txt')
        save_matrix(name, y_max)

def time_dataset(generator, model):
    names = generator.get_shapes_names()
    start = time.time()
    print('start predicting')
    for i in range(len(names)):
        x = generator.get_input(i)
        y = model.predict(x, batch_size=1, verbose=0, steps=None)
    end = time.time()
    print('done predicting')
    print(end - start)

def save_predicted_labels(path, path2, generator, model):
    names = generator.get_shapes_names()
    acc = 0.0
    for i in range(len(names)):
        x = generator.get_input(i)
        y = model.predict(x, batch_size=1, verbose=0, steps=None)
        y_max = np.transpose(np.max(y, axis=-1), (1, 0))
        y = np.transpose(np.argmax(y, axis=-1), (1, 0))
        y_true = generator.get_pred(i)
        print(names[i])
        print(np.shape(x['input_signal']))
        print(np.shape(y))
        print(np.shape(y_true))
        scores = model.evaluate(x=x, y=y_true, batch_size=1, verbose=1, sample_weight=None, steps=None)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        acc += scores[1]
        name = os.path.join(path, names[i] + '.txt')
        name2 = os.path.join(path2, names[i] + '.txt')
        save_matrix(name, y, dtype=np.int32)
        save_matrix(name2, y_max, dtype=np.float32)
    acc /= len(names)
    print('average accuracy:', acc)


def shape_dataset_segmentation(train_shapes,
                               test_shapes,
                               patch_op_path,
                               desc_path,
                               input_dim,
                               nfilters,
                               nclasses,
                               labels_path,
                               radius,
                               nbatch,
                               nv,
                               nrings,
                               ndirs,
                               ratio,
                               nepochs,
                               classes=None,
                               save_dir=None,
                               model_name='model'):

    if model_name is 'async':
        sync_mode = 'async'
    elif model_name is 'radial_sync':
        sync_mode = 'radial_sync'
    else:
        sync_mode = 'mlp'



    # load data
    if sync_mode is 'mlp':
        generator = ShapeMatchingFullDatasetMlp
        train_generator = generator(batch_size=nbatch,
                                    nv=nv,
                                    shapes=train_shapes,
                                    desc_path=desc_path,
                                    shuffle=True)
        test_generator = generator(batch_size=nbatch,
                                    nv=nv,
                                    shapes=test_shapes,
                                    desc_path=desc_path,
                                    shuffle=True)
    else:
        # generator = FullLabelizedDataset
        generator = ShapeMatchingFullDataset


        train_generator = generator(batch_size=nbatch,
                                nv=nv,
                                num_classes=nclasses,
                                shapes=train_shapes,
                                desc_path=desc_path,
                                labels_path=labels_path,
                                patch_op_path=patch_op_path,
                                radius=radius,
                                nrings=nrings,
                                ndirs=ndirs,
                                ratio=ratio,
                                shuffle=True,
                                augment_3d_data=False)

        test_generator = generator(batch_size=nbatch,
                               nv=nv,
                               num_classes=nclasses,
                               shapes=test_shapes,
                               desc_path=desc_path,
                               labels_path=labels_path,
                               patch_op_path=patch_op_path,
                               radius=radius,
                               nrings=nrings,
                               ndirs=ndirs,
                               ratio=ratio,
                               shuffle=True)

    # create model

    """
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
                           input_dim=train_generator.get_input_dim(),
                           nstacks=1,
                           nresblocks_per_stack=2,
                           nfilters=16,
                           sync_mode=sync_mode,
                           num_classes=nclasses)
    """

    if sync_mode is 'mlp':
        model_ = MlpSegmenter(n_batch=nbatch, n_v=nv,
                       num_classes=nclasses,
                       input_dim=train_generator.get_input_dim(),
                       nstacks=1,
                       nresblocks_per_stack=1,
                       nfilters=64)
    else:
        """
        model_ = GcnnUresNetSegmenter(n_batch=nbatch, ratios=ratio, n_v=nv, n_rings=nrings, n_dirs=ndirs,
                       num_classes=nclasses,
                       batch_norm=False,
                       input_dim=train_generator.get_input_dim(),
                       nstacks=2,
                       nresblocks_per_stack=1,
                       nfilters=nfilters,
                       sync_mode=sync_mode)
        """


    """
    model_ = GcnnResNetSegmenter(n_batch=nbatch, ratios=ratio, n_v=nv, n_rings=nrings, n_dirs=ndirs,
                        num_classes=nclasses,
                        batch_norm=False,
                        input_dim=train_generator.get_input_dim(),
                        nstacks=3,
                        nresblocks_per_stack=1,
                        nfilters=16,
                        sync_mode=sync_mode)
    """

    model = model_.get_model()

    # train model

    opt = 'adam'

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    """
    tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/Users/adrien/Documents/Keras/Gcnn/tensorboard',
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True)
    """

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.get_nsamples() / nbatch,
                                  epochs=nepochs,
                                  validation_data=test_generator,
                                  validation_steps=1,
                                  use_multiprocessing=False,
                                  workers=1,
                                  callbacks=[])

    return train_generator, test_generator, model, history





    """
    x_test = test_generator.get_inputs()
    y_test = test_generator.get_labels()

    
    history = model.fit(train_generator.get_inputs(), train_generator.get_labels(),
                        batch_size=nbatch,
                        epochs=nepochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)
    """

    # Score trained model.

    """
    scores = model.evaluate(x_test, y_test, verbose=1, batch_size=nbatch)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    """
    """
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

        # y_pred = model.predict(x_test, batch_size=nbatch, verbose=0)

        # plot_confusion_mat_(y_true=y_test, y_pred=y_pred, classes=classes,
        #                     save_path=os.path.join(save_dir, model_name + '_conf_mat'))
        plt_history(history=history, save_path=os.path.join(save_dir, model_name + '_history'))
    else:
        plt_history(history=history, save_path=None)
    """










