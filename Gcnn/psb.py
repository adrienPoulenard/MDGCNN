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
from shape_segmentation import shape_dataset_segmentation, save_correspondences, heterogeneous_dataset_segmentation
from shape_segmentation import save_predicted_labels
from train_network import heterogeneous_dataset
import matplotlib.pyplot as plt


drive = 'E'
# save_dir = drive + ':/Users/adrien/Documents/Keras/Gcnn/results'
sync_mode = 'radial_sync'
path = drive + ':/Users/Adrien/Documents/Datasets/PSB/'

descs_ = ["global_3d"]
descs = []
for i in range(len(descs_)):
    descs.append(path + 'descs/' + descs_[i])

train_generator, val_generator, test_generator, model, history = heterogeneous_dataset(
                                 task='segmentation',
                                 num_filters=16,
                                 train_list=path + 'train.txt',
                                 val_list=path + 'test.txt',
                                 test_list=path + 'test.txt',
                                 train_preds_path=path + 'labels',
                                 train_patch_op_path=path,
                                 train_desc_paths=descs,
                                 val_preds_path=path + 'labels',
                                 val_patch_op_path=path,
                                 val_desc_paths=descs,
                                 test_preds_path=path + 'labels',
                                 test_patch_op_path=path,
                                 test_desc_paths=descs,
                                 radius=[0.100000, 0.200000],
                                 nrings=[2, 2],
                                 ndirs=[8, 8],
                                 ratio=[1.000000, 0.250000],
                                 nepochs=200,
                                 sync_mode=sync_mode,
                                 nresblocks_per_stack=2,
                                 batch_norm=False,
                                 global_3d=True,
                                 num_classes=94)

plt_history(history=history)


predicted_labels_dir = path + 'results/' + descs_[0] + '/sync'
if sync_mode == 'async':
    predicted_labels_dir = path + 'results/' + descs_[0] + '/async'
save_predicted_labels(predicted_labels_dir, test_generator, model)

