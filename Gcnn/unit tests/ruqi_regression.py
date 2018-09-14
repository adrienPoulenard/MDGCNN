import numpy as np
from train_network import heterogeneous_dataset
from save_data import save_matrix
from plt_history import plt_history
import matplotlib.pyplot as plt


dataset_path = 'C:/Users/adrien/Documents/Datasets/ruqis_regression_2'
label_preds_path = 'C:/Users/adrien/Documents/Datasets/ruqis_regression_2/split1/'

train_generator, val_generator, test_generator, model, history = heterogeneous_dataset(
                                 task='regression',
                                 num_filters=16,
                                 train_list=label_preds_path + 'train.txt',
                                 val_list=label_preds_path + 'val.txt',
                                 test_list=label_preds_path + 'test.txt',
                                 train_preds_path=label_preds_path + 'train_gt_para.txt',
                                 train_patch_op_path=dataset_path,
                                 train_desc_paths=[dataset_path + '/descs/wks'],
                                 val_preds_path=label_preds_path + 'val_gt_para.txt',
                                 val_patch_op_path=dataset_path,
                                 val_desc_paths=[dataset_path + '/descs/wks'],
                                 test_preds_path=label_preds_path + 'test_gt_para.txt',
                                 test_patch_op_path=dataset_path,
                                 test_desc_paths=[dataset_path + '/descs/wks'],
                                 radius=[0.060000, 0.120000],
                                 nrings=[2, 2],
                                 ndirs=[8, 8],
                                 ratio=[1.000000, 0.250000],
                                 nepochs=200,
                                 nresblocks_per_stack=2,
                                 sync_mode='async',
                                 batch_norm=False,
                                 global_3d=False,
                                 num_classes=None)


preds = model.predict_generator(generator=test_generator, steps=test_generator.get_nsamples())

save_matrix(dataset_path + 'preds.txt', preds, dtype=np.float32)


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

test_score = model.evaluate_generator(generator=test_generator, steps=test_generator.get_nsamples())
print('test_loss')
print(test_score)

