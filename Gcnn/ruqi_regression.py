import numpy as np
from train_network import heterogeneous_dataset_regression
from save_data import save_matrix
from plt_history import plt_history


dataset_path = 'C:/Users/adrien/Documents/Datasets/ruqis_regression/'


train_generator, val_generator, test_generator, model, history = heterogeneous_dataset_regression(num_filters=32,
                                 train_list=dataset_path + 'train/train_id.txt',
                                 val_list=dataset_path + 'val/val_id.txt',
                                 test_list=dataset_path + 'test/test_id.txt',
                                 train_preds_path=dataset_path + 'train/train_preds.txt',
                                 train_patch_op_path=dataset_path + 'train',
                                 train_desc_paths=[dataset_path +'train/descs/wks'],
                                 val_preds_path=dataset_path + 'val/val_preds.txt',
                                 val_patch_op_path=dataset_path + 'val',
                                 val_desc_paths=[dataset_path + 'val/descs/wks'],
                                 test_preds_path=dataset_path + 'test/test_preds.txt',
                                 test_patch_op_path=dataset_path + 'test',
                                 test_desc_paths=[dataset_path +'test/descs/wks'],
                                 radius=[0.050000, 0.100000],
                                 nrings=[2, 2],
                                 ndirs=[8, 8],
                                 ratio=[1.000000, 0.250000],
                                 nepochs=100,
                                 sync_mode='async',
                                 batch_norm=False,
                                 global_3d=False)

preds = model.predict_generator(generator=test_generator, steps=test_generator.get_nsamples())
save_matrix(dataset_path + 'preds.txt', preds, dtype=np.float32)
plt_history(history=history)
