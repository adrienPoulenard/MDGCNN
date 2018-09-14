from run_training import main_training


"""
FLAT TORUS
"""


"""
# flat torus

batch_size = 10
num_classes = 9
epochs = 3

ntrain = 30000
ntest = 5000

train_nv = 28*28
test_nv = 42*42
nrings = 2
ndirs = 8


train_patch_op = "flat_torus_28x28_ndir=8_nrings=2_rad=2x28_30_oriented=1.txt"
test_patch_op = "flat_torus_42x42_ndir=8_nrings=2_rad=2x42_30_oriented=1.txt"

train_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/signals/flat_torus_28x28_training_30000.txt'
train_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/labels/training_labels_30000.txt'
train_path_c = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/connectivity/'
train_path_c += train_patch_op
train_path_t = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/transport/'
train_path_t += train_patch_op

test_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/signals/flat_torus_28x28_test_5000.txt'
test_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/labels/test_labels_5000.txt'
test_path_c = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/connectivity/'
test_path_c += test_patch_op
test_path_t = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/transport/'
test_path_t += test_patch_op

val_signal_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_flat_torus/signals/flat_torus_42x42_test_5000.txt'
"""

"""
SPHERE
"""
# sphere

batch_size = 10
num_classes = 8
epochs = 6

ntrain = 10000
ntest = 2000

train_nv = 3002
test_nv = 1502
nrings = 2
ndirs = 16

train_patch_op = "sphere_3002_ndir=16_nrings=2_rad=pi_12_oriented=1.txt"
test_patch_op = "sphere_1502_ndir=16_nrings=2_rad=pi_12_oriented=1.txt"

train_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/signals/sphere_3002_ratio=4.000000_training_10000.txt'
train_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/labels/training_labels_10000.txt'
train_path_c = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/connectivity/'
train_path_c += train_patch_op
train_path_t = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/transport/'
train_path_t += train_patch_op

test_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/signals/sphere_3002_ratio=4.000000_test_2000.txt'
test_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/labels/test_labels_2000.txt'
test_path_c = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/connectivity/'
test_path_c += test_patch_op
test_path_t = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/transport/'
test_path_t += test_patch_op

val_signal_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_spheres/signals/sphere_1502_ratio=4.000000_test_2000.txt'



"""
SPHERE 2
"""
# sphere
"""
batch_size = 30
num_classes = 2
epochs = 2

ntrain = 11916
ntest = 2064

train_nv = 1502
test_nv = 2002
nrings = 3
ndirs = 16

train_patch_op = "sphere_1502_ndir=16_nrings=3_rad=pi_2_oriented=1.txt"
test_patch_op = "sphere_2002_shuf_ndir=16_nrings=3_rad=pi_2_oriented=1.txt"

train_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/signals/sphere_1502_ratio=2.221441_training_11916.txt'
train_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/labels/training_labels_11916.txt'
train_path_c = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/connectivity/'
train_path_c += train_patch_op
train_path_t = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/transport/'
train_path_t += train_patch_op

test_signals_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/signals/sphere_1502_ratio=2.221441_test_2064.txt'
test_labels_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/labels/test_labels_2064.txt'
test_path_c = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/connectivity/'
test_path_c += test_patch_op
test_path_t = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/transport/'
test_path_t += test_patch_op

val_signal_path = 'C:/Users/Adrien/Documents/Datasets/MNIST_n_spheres/signals/sphere_2002_shuf_ratio=2.221441_test_2064.txt'
"""

main_training(batch_size, num_classes, epochs,
              ntrain, ntest, train_signals_path, train_labels_path, train_path_c, train_path_t,
              test_signals_path, test_labels_path, test_path_c, test_path_t,
              val_signal_path,
              train_nv, test_nv, nrings, ndirs)













