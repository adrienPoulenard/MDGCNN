import keras
from keras import backend as K
import time
import logging
import numpy as np
import pandas as pd


import os
import scipy.sparse as sp
import tensorflow as tf
from os import listdir
from os.path import isfile, join


def load_sparse_matrix(path_file, shape):
    """
    reads a sparse matrix in coo (triplets) format
    where data and ij satisfy the relationship a[ij[0, k], ij[1, k]] = data[k]
    """
    inp = np.loadtxt(path_file, delimiter=' ')
    rows, cols, data = inp.T
    out = sp.csr_matrix((data, (rows, cols)), shape).astype(np.float32)
    return out


def load_dense_matrix_from_triplets(path_file, shape):
    inp = np.loadtxt(path_file, delimiter=' ')
    rows, cols, data = inp.T
    a = np.zeros(shape=shape, dtype=np.float32)
    a[rows.astype(int), cols.astype(int)] = data
    return a


def sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def load_patch_operator(path_file, nv, ndirs, nrings):
    mat = load_sparse_matrix(path_file, (nv*ndirs*nrings, nv*ndirs))
    patch_op = sparse_matrix_to_sparse_tensor(mat)
    return patch_op


def load_patch_operator_as_triplets(path_file, nv, ndirs, nrings):
    mat = load_sparse_matrix(path_file, (nv * ndirs * nrings, nv * ndirs))
    coo = mat.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return (indices, coo.data, coo.shape)


def load_dense_matrix(path_file, d_type=np.float32):
    out = np.genfromtxt(fname=path_file, dtype=d_type, delimiter=' ')
    if np.ndim(out) == 1:
        out = np.expand_dims(out, axis=-1)
    # out = np.loadtxt(path_file, delimiter=' ', dtype=d_type)
    # out = np.fromfile(path_file, dtype=d_type, sep=' ')
    # t = pd.read_csv(path_file, dtype=d_type, header=None)
    # out = t.values
    return out


def load_dense_tensor(file_path, shape, d_type=np.float32):
    # t = np.fromfile(file_path, dtype=d_type, sep=' ')
    print(file_path)
    t = pd.read_csv(file_path, dtype=d_type, header=None)
    t = t.values
    t = np.squeeze(t)
    return np.reshape(t, newshape=shape)


def load_mnist(train_path, test_path, train_labels_path, test_labels_path,
               nv,
               ntrain=60000, ntest=10000):
    train_data = load_dense_matrix_from_triplets(train_path, (ntrain, nv))
    test_data = load_dense_matrix_from_triplets(test_path, (ntest, nv))
    # load_sparse_matrix(test_path, (ntest, nv)).todense()

    train_labels = load_dense_matrix(train_labels_path)
    test_labels = load_dense_matrix(test_labels_path)
    return (train_data, train_labels), (test_data, test_labels)


def load_mnist_test(test_path, test_labels_path, nv, ntest=20000):
    test_data = load_dense_matrix_from_triplets(test_path, (ntest, nv))
    test_labels = load_dense_matrix(test_labels_path)
    return (test_data, test_labels)


def float_to_string(x):
    return '%f' % x


def int_to_string(x):
    return '%ld' % x


def add_ext(name, ext):
    return name + '.' + ext


def patch_op_ff(name, radius, nrings, ndirs):
    res = name + '_rad=' + float_to_string(radius) + '_nrings=' + int_to_string(nrings) + '_ndirs=' + int_to_string(ndirs) + '.txt'
    return res


def pool_op_ff(name, r1, r2, nv, ndirs=None):
    if ndirs is not None:
        res = name + '_ratio_' + float_to_string(r1) + '_to_' + float_to_string(r2) + '_nv=' + int_to_string(nv) + '_ndirs=' + int_to_string(ndirs) + '.txt'
    else:
        res = name + '_ratio_' + float_to_string(r1) + '_to_' + float_to_string(r2) + '_nv=' + int_to_string(nv) + '.txt'

    return res


class classificationDataset(object):
    def __init__(self, radius, nlabels, ndesc, ntarsignals, nv, nrings, ndirs, train_txt, test_txt,
                 dataset_path, signals_path, target_signal_path='', labels_path=''):
        self.radius = radius
        self.nlabels = nlabels
        self.ndesc = ndesc
        self.ntarsignals = ntarsignals
        self.nv = nv
        self.nrings = nrings
        self.ndirs = ndirs
        self.train_txt = train_txt
        self.test_txt = test_txt

        self.signal_path = signals_path
        self.connectivity_path = os.path.join(dataset_path, 'connectivity')
        self.transport_path = os.path.join(dataset_path, 'transport')
        self.local_frames_path = os.path.join(dataset_path, 'local_frames')
        self.basepoints_path = os.path.join(dataset_path, 'basepoints')
        self.labels_path = labels_path
        self.target_signal_path = target_signal_path
        self.train_signal = []
        self.test_signal = []
        self.train_basepoints = []
        self.test_basepoints = []
        self.train_labels = []
        self.test_labels = []
        self.train_connectivity = []
        self.test_connectivity = []
        self.train_transport = []
        self.test_transport = []
        self.train_local_frames = []
        self.test_local_frames = []
        self.train_target_signal = []
        self.test_target_signal = []

        with open(self.train_txt, 'r') as f:
            self.train_fnames = [line.rstrip() for line in f]
        with open(self.test_txt, 'r') as f:
            self.test_fnames = [line.rstrip() for line in f]

        if self.signal_path:
            print("Loading train signal")
            tic = time.time()
            train_signal = []
            for name in self.train_fnames:
                train_signal.append(load_dense_tensor(os.path.join(self.signal_path, add_ext(name, 'txt')),
                                                      (self.nv, self.ndesc)))
            print("elapsed time %f" % (time.time() - tic))
            self.train_signal = np.stack(train_signal, axis=0)
            print("Loading test signal")
            tic = time.time()
            test_signal = []
            for name in self.test_fnames:
                test_signal.append(load_dense_tensor(os.path.join(self.signal_path, add_ext(name, 'txt')),
                                                     (self.nv, self.ndesc)))
            print("elapsed time %f" % (time.time() - tic))
            self.test_signal = np.stack(test_signal, axis=0)

        if self.target_signal_path:
            print("Loading train target signal")
            tic = time.time()
            train_target_signal = []
            for name in self.train_fnames:
                train_target_signal.append(load_dense_tensor(os.path.join(self.target_signal_path, add_ext(name, 'txt')),
                                                             (self.nv, self.ntarsignals)))
            print("elapsed time %f" % (time.time() - tic))
            self.train_target_signal = np.stack(train_target_signal, axis=0)
            print("Loading test target signal")
            tic = time.time()
            test_target_signal = []
            for name in self.test_fnames:
                test_target_signal.append(load_dense_tensor(os.path.join(self.target_signal_path, add_ext(name, 'txt')),
                                                            (self.nv, self.ntarsignals)))
            print("elapsed time %f" % (time.time() - tic))
            self.test_target_signal = np.stack(test_target_signal, axis=0)

        print("Loading train connectivity")
        tic = time.time()
        train_connectivity = []
        for name in self.train_fnames:
            path_ = os.path.join(self.connectivity_path, patch_op_ff(name, self.radius, self.nrings, self.ndirs))
            train_connectivity.append(
                load_dense_tensor(path_, shape=(self.nv, self.nrings, self.ndirs), d_type=np.int32))
        self.train_connectivity = np.stack(train_connectivity, axis=0)
            # del train_connectivity[:]
        print("elapsed time %f" % (time.time() - tic))

        print("Loading test connectivity")
        tic = time.time()
        test_connectivity = []
        for name in self.test_fnames:
            path_ = os.path.join(self.connectivity_path, patch_op_ff(name, self.radius, self.nrings, self.ndirs))
            test_connectivity.append(
                load_dense_tensor(path_, shape=(self.nv, self.nrings, self.ndirs), d_type=np.int32))
        self.test_connectivity = np.stack(test_connectivity, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        print("Loading train transport")
        tic = time.time()
        train_transport = []
        for name in self.train_fnames:
            path_ = os.path.join(self.transport_path, patch_op_ff(name, self.radius, self.nrings, self.ndirs))
            train_transport.append(
                load_dense_tensor(path_, (self.nv, self.nrings, self.ndirs), d_type=np.int32))
        self.train_transport = np.stack(train_transport, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        print("Loading test transport")
        tic = time.time()
        test_transport = []
        for name in self.test_fnames:
            path_ = os.path.join(self.transport_path, patch_op_ff(name, self.radius, self.nrings, self.ndirs))
            test_transport.append(
                load_dense_tensor(path_, shape=(self.nv, self.nrings, self.ndirs), d_type=np.int32))
        self.test_transport = np.stack(test_transport, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        print("Loading train basepoints")
        tic = time.time()
        train_basepoints = []
        for name in self.train_fnames:
            path_ = os.path.join(self.basepoints_path, add_ext(name, 'txt'))
            train_basepoints.append(load_dense_tensor(path_, shape=(self.nv, 3)))
        self.train_basepoints = np.stack(train_basepoints, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        print("Loading test basepoints")
        tic = time.time()
        test_basepoints = []
        for name in self.test_fnames:
            path_ = os.path.join(self.basepoints_path, add_ext(name, 'txt'))
            test_basepoints.append(load_dense_tensor(path_, shape=(self.nv, 3)))
        self.test_basepoints = np.stack(test_basepoints, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        print("Loading train local 3d frames")
        tic = time.time()
        train_3d_frames = []
        for name in self.train_fnames:
            path_ = os.path.join(self.local_frames_path, add_ext(name, 'txt'))
            y = load_dense_tensor(path_, shape=(self.nv, 3, 3))
            train_3d_frames.append(y)
        self.train_local_frames = np.stack(train_3d_frames, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        print("Loading test local 3d frames")
        tic = time.time()
        test_3d_frames = []
        for name in self.test_fnames:
            path_ = os.path.join(self.local_frames_path, add_ext(name, 'txt'))
            test_3d_frames.append(load_dense_tensor(path_, shape=(self.nv, 3, 3)))
        self.test_local_frames = np.stack(test_3d_frames, axis=0)
        print("elapsed time %f" % (time.time() - tic))

        if self.labels_path:
            print("Loading train labels")
            tic = time.time()
            train_labels = []
            for name in self.train_fnames:
                y = load_dense_matrix(os.path.join(self.labels_path, add_ext(name, 'txt')), d_type=np.int32).squeeze()
                y = keras.utils.to_categorical(y, self.nlabels)
                # y = np.reshape(y, (self.nv*self.nlabels))
                # y = y.flatten()
                train_labels.append(y)
            self.train_labels = np.stack(train_labels, axis=0)
            print("elapsed time %f" % (time.time() - tic))

            print("Loading test labels")
            tic = time.time()
            test_labels = []
            for name in self.test_fnames:
                y = load_dense_matrix(os.path.join(self.labels_path, add_ext(name, 'txt')), d_type=np.int32).squeeze()
                y = keras.utils.to_categorical(y, self.nlabels)
                # y = np.reshape(y, (self.nv*self.nlabels))
                # y = y.flatten()
                test_labels.append(y)
            self.test_labels = np.stack(test_labels, axis=0)
            print("elapsed time %f" % (time.time() - tic))

    def get_train_signal(self):
        return self.train_signal

    def get_test_signal(self):
        return self.test_signal

    def get_train_target_signal(self):
        return self.train_target_signal

    def get_test_target_signal(self):
        return self.test_target_signal

    def get_train_connectivity(self):
        return self.train_connectivity

    def get_test_connectivity(self):
        return self.test_connectivity

    def get_train_transport(self):
        return self.train_transport

    def get_test_transport(self):
        return self.test_transport

    def get_train_basepoints(self):
        return self.train_basepoints

    def get_test_basepoints(self):
        return self.test_basepoints

    def get_train_local_frames(self):
        return self.train_local_frames

    def get_test_local_frames(self):
        return self.test_local_frames

    def get_train_labels(self):
        return self.train_labels

    def get_test_labels(self):
        return self.test_labels


def load_fixed_patch_op(dataset_path, name, radius, nv, nrings, ndirs, ratio):
    contributors = []
    weights = []
    angles = []
    print('loading patch operators')
    for i in range(len(radius)):
        name_ = name + '_ratio=' + float_to_string(ratio[i]) + '_nv=' + int_to_string(nv[i])
        c_path = os.path.join(dataset_path + '/bin_contributors',
                              patch_op_ff(name_, radius[i], nrings[i], ndirs[i]))
        contributors.append(load_dense_tensor(c_path, shape=(nv[i], nrings[i], ndirs[i], 3), d_type=np.int32))

        w_path = os.path.join(dataset_path + '/contributors_weights',
                              patch_op_ff(name_, radius[i], nrings[i], ndirs[i]))
        weights.append(load_dense_tensor(w_path, (nv[i], nrings[i], ndirs[i], 3), d_type=np.float32))

        a_path = os.path.join(dataset_path + '/transported_angles',
                              patch_op_ff(name_, radius[i], nrings[i], ndirs[i]))
        angles.append(load_dense_tensor(a_path, (nv[i], nrings[i], ndirs[i], 3), d_type=np.float32))
    print('loading pooling operators')

    angular_shifts = []
    parent_vertices = []

    for i in range(len(ratio) - 1):
        a_path = os.path.join(dataset_path + '/angular_shifts',
                              pool_op_ff(name, ratio[i], ratio[i + 1], nv[i + 1]))
        angular_shifts.append(load_dense_matrix(a_path, d_type=np.float32))
        p_path = os.path.join(dataset_path + '/parent_vertices',
                              pool_op_ff(name, ratio[i], ratio[i + 1], nv[i + 1]))
        parent_vertices.append(load_dense_matrix(p_path, d_type=np.int32))

    return contributors, weights, angles, parent_vertices, angular_shifts

# def parse_patch_op(path, name, radius, nrings, ndirs, ratio):


def parse_patch_op(file_name):
    spl = file_name.split('_ratio=')
    name = spl[0]
    spl = spl[1].split('_nv=')
    ratio = float(spl[0])
    spl = spl[1].split('_rad=')
    nv = int(spl[0])
    spl = spl[1].split('_nrings=')
    rad = float(spl[0])
    spl = spl[1].split('_ndirs=')
    nrings = int(spl[0])
    spl = spl[1].split('.txt')
    ndirs = int(spl[0])
    return name, ratio, nv, rad, nrings, ndirs


def is_patch_op_file(file):
    return True


def patch_op_name(file_name):
    spl = file_name.split('_ratio=')
    return spl[0]


def patch_op_key(name, ratio, rad, nrings, ndirs):
    key = name + '_' + float_to_string(ratio) + '_' + float_to_string(rad)
    key += '_' + int_to_string(nrings) + '_' + int_to_string(ndirs)
    return key


def list_patch_op_dir(path):
    path = os.path.join(path, 'bin_contributors')
    res = dict([])
    for file in os.listdir(path):
        if is_patch_op_file(file):
            name, ratio, nv, rad, nrings, ndirs = parse_patch_op(file)
            key = patch_op_key(name, ratio, rad, nrings, ndirs)
            res[key] = nv
    return res


def load_single_patch_op(dataset_path, name, radius, nv, nrings, ndirs, ratio):

    name_ = name + '_ratio=' + float_to_string(ratio) + '_nv=' + int_to_string(nv)
    c_path = os.path.join(dataset_path + '/bin_contributors',
                          patch_op_ff(name_, radius, nrings, ndirs))
    contributors = load_dense_tensor(c_path, shape=(nv, nrings, ndirs, 3), d_type=np.int32)

    w_path = os.path.join(dataset_path + '/contributors_weights',
                          patch_op_ff(name_, radius, nrings, ndirs))
    weights = load_dense_tensor(w_path, (nv, nrings, ndirs, 3), d_type=np.float32)

    a_path = os.path.join(dataset_path + '/transported_angles',
                          patch_op_ff(name_, radius, nrings, ndirs))
    angles = load_dense_tensor(a_path, (nv, nrings, ndirs, 3), d_type=np.float32)

    return contributors, weights, angles


def load_single_pool_op(dataset_path, name, old_nv, new_nv, ratio1, ratio2):
    if ratio1 < ratio2:
        ratio1, ratio2 = ratio2, ratio1
        new_nv, old_nv = old_nv, new_nv
    a_path = os.path.join(dataset_path + '/angular_shifts',
                          pool_op_ff(name, ratio1, ratio2, new_nv))
    angular_shifts = load_dense_matrix(a_path, d_type=np.float32).flatten()
    p_path = os.path.join(dataset_path + '/parent_vertices',
                          pool_op_ff(name, ratio1, ratio2, new_nv))
    parent_vertices = load_dense_matrix(p_path, d_type=np.int32).flatten()

    return parent_vertices, angular_shifts


def load_patch_op(shapes_names_txt,
                  shapes_nv,
                  radius,
                  nrings,
                  ndirs,
                  ratio,
                  dataset_path):

    if type(shapes_names_txt) is str:
        with open(shapes_names_txt, 'r') as f:
            fnames = [line.rstrip() for line in f]
    elif type(shapes_names_txt) is list:
        fnames = shapes_names_txt
    else:
        raise ValueError('wrong input type')


    dynamic_nv = False
    if type(shapes_nv) is str:
        dynamic_nv = True
        nv = load_dense_matrix(shapes_nv, d_type=np.int32)
    else:
        nv = shapes_nv

    contributors = []
    weights = []
    angles = []

    parents = []
    angular_shifts = []

    for j in range(len(radius)):

        c_ = []
        w_ = []
        a_ = []

        for i in range(len(fnames)):
            if dynamic_nv:
                c, w, a = load_single_patch_op(dataset_path, fnames[i], radius[j], nv[i, j], nrings[j], ndirs[j], ratio[j])
            else:
                c, w, a = load_single_patch_op(dataset_path, fnames[i], radius[j], nv[j], nrings[j], ndirs[j], ratio[j])

            c_.append(c)
            w_.append(w)
            a_.append(a)

        contributors.append(np.stack(c_, axis=0))
        weights.append(np.stack(w_, axis=0))
        angles.append(np.stack(a_, axis=0))

    for j in range(len(radius)-1):
        p_ = []
        a_s_ = []
        for i in range(len(fnames)):
            p, a_s = load_single_pool_op(dataset_path, fnames[i], nv[j], nv[j+1], ratio[j], ratio[j+1])
            p_.append(p)
            a_s_.append(a_s)

        parents.append(np.stack(p_, axis=0))
        angular_shifts.append(np.stack(a_s_, axis=0))

    return contributors, weights, angles, parents, angular_shifts


def load_names(shapes_names_txt):
    if type(shapes_names_txt) is str:
        with open(shapes_names_txt, 'r') as f:
            fnames = [line.rstrip() for line in f]
    elif type(shapes_names_txt) is list:
        fnames = shapes_names_txt
    else:
        raise ValueError('wrong input type')
    return fnames


def load_labels(shapes_names_txt, labels_path, nclases, to_categorical=False):

    fnames = load_names(shapes_names_txt)
    labels = []
    for name in fnames:
        y = load_dense_matrix(os.path.join(labels_path, name + '.txt'), d_type=np.int32)
        if to_categorical:
            y = keras.utils.to_categorical(y, nclases)

        labels.append(y)

    labels = np.stack(labels, axis=0)

    return labels


def load_descriptors(shapes_names_txt, descriptors_path):

    fnames = load_names(shapes_names_txt)
    descriptors = []
    for name in fnames:
        x = load_dense_matrix(os.path.join(descriptors_path, name + '.txt'), d_type=np.float32)
        descriptors.append(x)

    descriptors = np.stack(descriptors, axis=0)

    return descriptors


