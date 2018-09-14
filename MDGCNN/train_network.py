import keras
from keras.models import Model
import numpy as np
import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from plt_history import plt_history
from confusion_mat import plot_confusion_mat_
import os
from load_data import int_to_string
from save_data import save_matrix
from plt_history import plt_history
from data_gen import LoadHeterogeneousDatasetFromList
from models import GcnnUresNet, GcnnResNetRegressor


def heterogeneous_dataset(task,
                          num_filters,
                          train_list,
                          val_list,
                          test_list,
                          train_preds_path,
                          train_patch_op_path,
                          train_desc_paths,
                          val_preds_path,
                          val_patch_op_path,
                          val_desc_paths,
                          test_preds_path,
                          test_patch_op_path,
                          test_desc_paths,
                          radius,
                          nrings,
                          ndirs,
                          ratio,
                          nepochs,
                          sync_mode,
                          nresblocks_per_stack=2,
                          batch_norm=False,
                          global_3d=False,
                          num_classes=None):

    is_clasifier = False
    if task == 'segmentation' or task == 'classification':
        is_clasifier = True
        if num_classes is None:
            raise ValueError('unspecified number of classes')

    train_generator = LoadHeterogeneousDatasetFromList(names_list=train_list,
                                                       preds_path=train_preds_path,
                                                       patch_ops_path=train_patch_op_path,
                                                       descs_paths=train_desc_paths,
                                                       radius=radius,
                                                       nrings=nrings,
                                                       ndirs=ndirs,
                                                       ratio=ratio,
                                                       shuffle=True,
                                                       add_noise=global_3d,
                                                       num_classes=num_classes)

    val_generator = LoadHeterogeneousDatasetFromList(names_list=val_list,
                                                     preds_path=val_preds_path,
                                                     patch_ops_path=val_patch_op_path,
                                                     descs_paths=val_desc_paths,
                                                     radius=radius,
                                                     nrings=nrings,
                                                     ndirs=ndirs,
                                                     ratio=ratio,
                                                     shuffle=True,
                                                     add_noise=False,
                                                     num_classes=num_classes)

    test_generator = LoadHeterogeneousDatasetFromList(names_list=test_list,
                                                      preds_path=test_preds_path,
                                                      patch_ops_path=test_patch_op_path,
                                                      descs_paths=test_desc_paths,
                                                      radius=radius,
                                                      nrings=nrings,
                                                      ndirs=ndirs,
                                                      ratio=ratio,
                                                      shuffle=True,
                                                      add_noise=False,
                                                      num_classes=num_classes)

    if task == 'regression':
        model_ = GcnnResNetRegressor(n_batch=1, ratios=ratio, n_v=None, n_rings=nrings, n_dirs=ndirs,
                     num_channels=train_generator.get_preds_dim(),
                     batch_norm=False,
                     input_dim=train_generator.get_input_dim(),
                     nstacks=len(ratio),
                     nresblocks_per_stack=2,
                     nfilters=num_filters,
                     sync_mode=sync_mode)
    else:
        model_ = GcnnUresNet(n_batch=1, ratios=ratio, n_v=None, n_rings=nrings, n_dirs=ndirs,
                task=task,
                output_channels=train_generator.get_preds_dim(),
                batch_norm=batch_norm,
                input_dim=train_generator.get_input_dim(),
                nstacks=len(ratio),
                nresblocks_per_stack=nresblocks_per_stack,
                nfilters=num_filters,
                sync_mode=sync_mode,
                global_3d=False)

    model = model_.get_model()

    # train model

    opt = 'adam'
    metrics = []
    loss = 'mse'
    if is_clasifier:
        metrics.append('accuracy')
        loss = 'categorical_crossentropy'

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)

    model.summary()

    """
    tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/Users/adrien/Documents/Keras/Gcnn/tensorboard',
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True)
    """

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.get_nsamples(),
                                  epochs=nepochs,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.get_nsamples(),
                                  use_multiprocessing=False,
                                  workers=1,
                                  callbacks=[])

    """
    test_mse = model.evaluate_generator(generator=test_generator,
                                        steps=test_generator.get_nsamples())

    print('test_mse:')
    print(test_mse)
    """

    return train_generator, val_generator, test_generator, model, history


