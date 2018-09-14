# import load_data as data
import numpy as np
import scipy as sp
# import tensorflow as tf
import load_data
import save_data
"""
n1 = 10
n2 = 23
n3 = 7

a = np.zeros(n1*n2*n3)

a[n3*(n2*2 + 6) + 3] = 1.0
a = a.reshape((10, 23, 7))
print(a[2, 6, 2])
"""

"""
x = 0.2
print(str(float(x)))
print('%f' % x)
"""
"""
a = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])
print(np.shape(a))
b = np.array([[1, 1, 1], [2, 2, 2]])
print(np.shape(b))
"""

# t = load_data.load_dense_tensor('C:/Users/Adrien/Documents/Datasets/Faust/matlab_descs/tr_reg_000.txt', shape=(6890, 15)).squeeze()
# print(t[1, 1])

"""
print('aaa')
t = load_data.load_dense_tensor('C:/Users/adrien/Documents/Datasets/cifar10_spheres/signals/sphere_3002_ratio=2.000000_test_10000_0.txt',
                                shape=(2000, 3002, 3)).squeeze()
print('bbb')

x1 = t[150, :, :]
x2 = t[250, :, :]
x3 = t[350, :, :]
# print(x.shape())

save_data.save_tensor('C:/Users/adrien/Documents/Datasets/cifar10_spheres/test/test1.txt', x1)
save_data.save_tensor('C:/Users/adrien/Documents/Datasets/cifar10_spheres/test/test2.txt', x2)
save_data.save_tensor('C:/Users/adrien/Documents/Datasets/cifar10_spheres/test/test3.txt', x3)
"""

def variance(l):
    m = 0.
    for i in range(len(l)):
        m += l[i]
    m /= float(len(l))
    v = 0.
    for i in range(len(l)):
        v += (l[i] - m)*(l[i] - m)
    v /= float(len(l))
    return v


from plot_test import history_sync_sig17, history_async_sig17

def get_acc(history, epoch):
    l = []
    for i in range(len(history)):
        l.append(history[i]['val_acc'][epoch])
    return l





"""
print('sync_var')
print(np.sqrt(variance([0.8762, 0.8785, 0.8814, 0.8886, 0.8855])))

l_ = [0.7636, 0.8543, 0.8651, 0.7199, 0.8188, 0.7863, 0.8449, 0.8625, 0.8229, 0.8127, 0.7934]
b_ = [0.7636, 0.8543, 0.8651, 0.7199, 0.8188, 0.7863, 0.8449, 0.8625, 0.8229, 0.8127, 0.7934]
print('async_var')
print(np.sqrt(variance([0.7636, 0.8543, 0.8651, 0.7199, 0.8188])))
"""

epochs = 199
print('epochs')
print(epochs)
print('sync_var')
print(np.sqrt(variance(get_acc(history_sync_sig17, epochs))))

print('async_var')
print(np.sqrt(variance(get_acc(history_async_sig17, epochs))))




