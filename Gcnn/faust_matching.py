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
from train_network import heterogeneous_dataset
import matplotlib.pyplot as plt


drive = 'C'
# save_dir = drive + ':/Users/adrien/Documents/Keras/Gcnn/results'
sync_mode = 'radial_sync'
faust_path = drive + ':/Users/Adrien/Documents/Datasets/Faust_5k/'

"""
train_gen, test_gen, model, history = shape_dataset_segmentation(train_shapes=faust_path + 'train.txt',
                                                                 test_shapes=faust_path + 'test.txt',
                                                                 patch_op_path=faust_path,
                                                                 desc_path=faust_path + 'signals/shot_1_bin_24',
                                                                 input_dim=3,
                                                                 nfilters=32,
                                                                 nclasses=6890,
                                                                 labels_path=faust_path + 'labels/points_id',
                                                                 radius=[0.050000, 0.100000],
                                                                 nbatch=1,
                                                                 nv=[6890, 1724],
                                                                 nrings=[2, 2],
                                                                 ndirs=[8, 8],
                                                                 ratio=[1.000000, 0.250000],
                                                                 nepochs=100,
                                                                 classes=['left', 'right'],
                                                                 save_dir=None,
                                                                 model_name=sync_mode)

# save predictions
if sync_mode is 'radial_sync':
    path = faust_path + 'results/shape_matching_id_preds/sync'
elif sync_mode is 'async':
    path = faust_path + 'results/shape_matching_id_preds/async'
else:
    path = faust_path + 'results/shape_matching_id_preds/mlp'

save_correspondences(path, test_gen, model)

plt_history(history=history)
"""


descs = ['shot_1_bin_6']
descs_str = ''
for i in range(len(descs)):
    descs_str += descs[i] + '_'
    descs[i] = faust_path + 'descs/' + descs[i]


train_generator, val_generator, test_generator, model, history = heterogeneous_dataset(
                                 task='segmentation',
                                 num_filters=16,
                                 train_list=faust_path + 'train.txt',
                                 val_list=faust_path + 'test.txt',
                                 test_list=faust_path + 'test.txt',
                                 train_preds_path=faust_path + 'labels',
                                 train_patch_op_path=faust_path,
                                 train_desc_paths=descs,
                                 val_preds_path=faust_path + 'labels',
                                 val_patch_op_path=faust_path,
                                 val_desc_paths=descs,
                                 test_preds_path=faust_path + 'labels',
                                 test_patch_op_path=faust_path,
                                 test_desc_paths=descs,
                                 radius=[0.100000, 0.200000],
                                 nrings=[2, 2],
                                 ndirs=[8, 8],
                                 ratio=[1.000000, 0.250000],
                                 nepochs=200,
                                 sync_mode=sync_mode,
                                 nresblocks_per_stack=2,
                                 num_classes=5000,
                                 batch_norm=False,
                                 global_3d=False)

# save predictions
if sync_mode is 'radial_sync':
    path = faust_path + 'results/' + descs_str + '/sync'
elif sync_mode is 'async':
    path = faust_path + 'results/' + descs_str + '/async'
else:
    path = faust_path + 'results/' + descs_str + '/mlp'

save_correspondences(path, test_generator, model)

plt_history(history=history)


"""
names = test_generator.get_shapes_names()
for i in range(len(names)):
    x = test_generator.get_input(i)
    y = model.predict(x, batch_size=1, verbose=0, steps=None)[0, :, :]
    name = os.path.join(path, names[i] + '.txt')
    save_matrix(name, y, dtype=np.float32)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
"""





