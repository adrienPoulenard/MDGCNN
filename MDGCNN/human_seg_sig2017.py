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
from shape_segmentation import heterogeneous_dataset_segmentation, save_predicted_labels, time_dataset
from plt_history import plt_history
from train_network import heterogeneous_dataset

drive = 'C'
# dataset_path = drive + ':/Users/adrien/Documents/Datasets/SIG 17/'
dataset_path = 'C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/'

descs_folders = ['global3d/']
# descs_folders = ['wks/', 'curvature/']




mit_folders = ['bouncing',
               'crane',
               'handstand',
               'jumping',
               'march1',
               'march2',
               'squat1',
               'squat2']

train_descs_paths = []
train_labels_paths = []
train_patch_op_paths = []

desc_paths = []


for i in range(8):
    desc_paths = []
    for j in range(len(descs_folders)):
        desc_paths.append(dataset_path + 'descs/' + descs_folders[j] + 'train/mit/' + mit_folders[i])

    train_descs_paths.append(desc_paths)
    train_labels_paths.append(dataset_path + 'segs/train/mit/' + mit_folders[i])
    train_patch_op_paths.append(dataset_path + 'patch_ops/train/mit/' + mit_folders[i])


desc_paths = []
for j in range(len(descs_folders)):
    desc_paths.append(dataset_path + 'descs/' + descs_folders[j] + 'train/adobe')

train_descs_paths.append(desc_paths)
train_labels_paths.append(dataset_path + 'segs/train/adobe')
train_patch_op_paths.append(dataset_path + 'patch_ops/train/adobe')

desc_paths = []
for j in range(len(descs_folders)):
    desc_paths.append(dataset_path + 'descs/' + descs_folders[j] + 'train/faust')

train_descs_paths.append(desc_paths)
train_labels_paths.append(dataset_path + 'segs/train/faust')
train_patch_op_paths.append(dataset_path + 'patch_ops/train/faust')

desc_paths = []
for j in range(len(descs_folders)):
    desc_paths.append(dataset_path + 'descs/' + descs_folders[j] + 'train/scape')

train_descs_paths.append(desc_paths)
train_labels_paths.append(dataset_path + 'segs/train/scape')
train_patch_op_paths.append(dataset_path + 'patch_ops/train/scape')


test_descs_paths = []
test_labels_paths = []
test_patch_op_paths = []

desc_paths = []
for j in range(len(descs_folders)):
    desc_paths.append(dataset_path + 'descs/' + descs_folders[j] + 'test/shrec')

test_descs_paths.append(desc_paths)
test_labels_paths.append(dataset_path + 'segs/test/shrec')
test_patch_op_paths.append(dataset_path + 'patch_ops/test/shrec')


radius = [0.100000, 0.200000]
ratios = [1.000000, 0.250000]
nrings = [2, 2]
ndirs = [8, 8]

train_gen, test_gen, model, history = heterogeneous_dataset_segmentation(num_classes=8,
                                                                         num_filters=16,
                                                                         train_patch_op_paths=train_patch_op_paths,
                                                                         train_desc_paths=train_descs_paths,
                                                                         train_labels_paths=train_labels_paths,
                                                                         test_patch_op_paths=test_patch_op_paths,
                                                                         test_desc_paths=test_descs_paths,
                                                                         test_labels_paths=test_labels_paths,
                                                                         radius=radius,
                                                                         nrings=nrings,
                                                                         ndirs=ndirs,
                                                                         ratio=ratios,
                                                                         nepochs=1,
                                                                         sync_mode='radial_sync',
                                                                         batch_norm=False,
                                                                         global_3d=True)
print('test_time')
time_dataset(test_gen, model)
print('train_time')
time_dataset(train_gen, model)
predicted_labels_dir = dataset_path + 'preds'
preds_dir = dataset_path + 'raw_preds'
# save_predicted_labels(predicted_labels_dir, preds_dir, test_gen, model)
plt_history(history=history.history)



"""
path = dataset_path
sync_mode = 'async'
descs_ = ['global_3d']
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
                                 radius=[0.080000, 0.160000],
                                 nrings=[2, 2],
                                 ndirs=[8, 8],
                                 ratio=[1.000000, 0.250000],
                                 nepochs=200,
                                 sync_mode=sync_mode,
                                 nresblocks_per_stack=2,
                                 batch_norm=False,
                                 global_3d=True,
                                 num_classes=8)
"""

"""
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
                                 radius=[0.150000],
                                 nrings=[3],
                                 ndirs=[12],
                                 ratio=[1.000000],
                                 nepochs=200,
                                 sync_mode=sync_mode,
                                 nresblocks_per_stack=3,
                                 batch_norm=False,
                                 global_3d=True,
                                 num_classes=8)
"""
"""
preds_dir = path + 'results/preds/' + descs_[0] + '/sync'
predicted_labels_dir = path + 'results/' + descs_[0] + '/sync'

if sync_mode == 'async':
    predicted_labels_dir = path + 'results/' + descs_[0] + '/async'
    preds_dir = path + 'results/preds/' + descs_[0] + '/async'
save_predicted_labels(predicted_labels_dir, preds_dir, test_generator, model)
plt_history(history=history)
"""





