
import numpy as np
import keras
import os
from load_data import int_to_string, load_patch_op, load_labels, load_descriptors, load_names
from load_data import load_single_patch_op, load_single_pool_op
from load_data import load_dense_matrix, load_dense_tensor
from load_data import list_patch_op_dir, patch_op_key
from rotation_generator import generate


def variance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    y = np.subtract(x, mean)
    y = np.multiply(y, y)
    return np.sum(np.mean(y, axis=1), axis=-1)


def gaussian_scaling_perturbation(x):
    shape = np.shape(x)
    nv = shape[1]
    i = np.random.randint(low=0, high=nv)
    p = np.expand_dims(x[:, i, :], axis=1)
    y = np.subtract(x, p)
    a, b = np.random.rand(2)
    var = variance(x)
    sigma = np.divide(1., (0.5*(a + 0.1)*var))
    scale = 0.4*b
    y_norm = np.sum(np.multiply(y, y), axis=-1, keepdims=True)
    mult = np.multiply(scale, np.exp(-np.multiply(sigma, y_norm)))
    mult = np.add(mult, np.ones(shape=shape, dtype=np.float32))
    y = np.multiply(mult, y)
    return np.add(y, p)


class FullHeterogeneousLabelizedDataset(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 num_classes,
                 desc_paths,
                 labels_paths,
                 patch_op_paths,
                 radius,
                 nrings,
                 ndirs,
                 ratio,
                 shuffle=True,
                 add_noise=False):

        self.add_noise = add_noise
        self.batch_size = 1

        nb_directories = len(labels_paths)
        self.names = []
        self.labels = []
        self.inputs = []
        nb_patch_op = len(radius)
        self.keys = ['input_signal']
        for j in range(nb_patch_op):
            self.keys.append('contributors_' + int_to_string(j))
            self.keys.append('weights_' + int_to_string(j))
            self.keys.append('transport_' + int_to_string(j))
        for j in range(nb_patch_op-1):
            self.keys.append('parents_' + int_to_string(j))
            self.keys.append('angular_shifts_' + int_to_string(j))

        for i in range(nb_directories):
            for file in os.listdir(labels_paths[i]):
                x = []
                if file.endswith('.txt'):
                    name = file.split('.')[0]
                    print(name)
                    self.names.append(name)
                    labels = load_dense_matrix(os.path.join(labels_paths[i], file), d_type=np.int32)
                    self.labels.append(np.expand_dims(labels, axis=0))
                    # nv.append(np.shape(labels_)[-1])
                    descs = []
                    for j in range(len(desc_paths[i])):
                        desc = load_dense_matrix(os.path.join(desc_paths[i][j], file), d_type=np.float32)
                        descs.append(np.expand_dims(desc, axis=0))

                    x.append(np.concatenate(descs, axis=-1))
                    patch_op_nv = list_patch_op_dir(patch_op_paths[i])

                    # load patch op
                    nv = []
                    for j in range(len(radius)):
                        nv.append(patch_op_nv[patch_op_key(name, ratio[j], radius[j], nrings[j], ndirs[j])])

                    for j in range(len(radius)):
                        contributors, weights, transport = load_single_patch_op(dataset_path=patch_op_paths[i],
                                                                                name=name,
                                                                                radius=radius[j],
                                                                                nv=nv[j],
                                                                                nrings=nrings[j],
                                                                                ndirs=ndirs[j],
                                                                                ratio=ratio[j])
                        x.append(np.expand_dims(contributors, axis=0))
                        x.append(np.expand_dims(weights, axis=0))
                        x.append(np.expand_dims(transport, axis=0))

                    for j in range(len(radius)-1):
                        parent_vertices, angular_shifts = load_single_pool_op(dataset_path=patch_op_paths[i],
                                                                              name=name,
                                                                              old_nv=nv[j],
                                                                              new_nv=nv[j+1],
                                                                              ratio1=ratio[j],
                                                                              ratio2=ratio[j+1])
                        x.append(np.expand_dims(parent_vertices, axis=0))
                        x.append(np.expand_dims(angular_shifts, axis=0))
                    self.inputs.append(dict(zip(self.keys, x)))

        self.nsamples = len(self.inputs)
        self.input_dim = self.inputs[0]['input_signal'].shape[-1]
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idx = self.indexes[index]
        # Generate data
        X, y = self.__data_generation(idx)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.inputs[indexes].copy()
        if self.add_noise:
            # X['input_signal'] = gaussian_scaling_perturbation(X['input_signal'])
            rot = generate(3)
            a = np.random.rand(1)
            rot = np.multiply((1. + 0.15*(2*a-1.0)), rot)
            X['input_signal'] = np.einsum('ijk,lk->ijl', X['input_signal'], rot)
        y = keras.utils.to_categorical(self.labels[indexes])
        return X, y

    def get_input(self, i):
        return self.inputs[i]

    def get_label(self, i):
        return keras.utils.to_categorical(self.labels[i])

    def get_inputs(self):
        return self.inputs

    def get_labels(self):
        return self.labels

    def get_nsamples(self):
        return self.nsamples

    def get_input_dim(self):
        return self.input_dim

    def get_shapes_names(self):
        return self.names


class LoadHeterogeneousDatasetFromList(keras.utils.Sequence):

    def __init__(self,
                 names_list,
                 preds_path,
                 patch_ops_path,
                 descs_paths,
                 radius,
                 nrings,
                 ndirs,
                 ratio,
                 shuffle=True,
                 add_noise=False,
                 num_classes=2):

        self.add_noise = add_noise
        self.batch_size = 1

        if num_classes is None:
            dtype = np.float32
            self.is_classifier = False
        else:
            dtype = np.int32
            self.is_classifier = True


        self.names = []
        self.preds = []
        self.inputs = []
        nb_patch_op = len(radius)
        self.keys = ['input_signal']
        for j in range(nb_patch_op):
            self.keys.append('contributors_' + int_to_string(j))
            self.keys.append('weights_' + int_to_string(j))
            self.keys.append('transport_' + int_to_string(j))
        for j in range(nb_patch_op - 1):
            self.keys.append('parents_' + int_to_string(j))
            self.keys.append('angular_shifts_' + int_to_string(j))

        self.names = load_names(names_list)
        self.nsamples = len(self.names)
        ext = preds_path[-4:]

        if ext == '.txt':
            preds = load_dense_matrix(preds_path, d_type=dtype)
            for i in range(self.nsamples):
                self.preds.append(np.expand_dims(preds[i, :], axis=0))
        else:
            for i in range(self.nsamples):
                labels = load_dense_matrix(os.path.join(preds_path, self.names[i] + '.txt'), d_type=dtype)
                self.preds.append(np.expand_dims(labels, axis=0))

        for i in range(self.nsamples):
            x = []
            descs = []
            for j in range(len(descs_paths)):
                desc = load_dense_matrix(os.path.join(descs_paths[j], self.names[i] + '.txt'), d_type=np.float32)
                descs.append(np.expand_dims(desc, axis=0))
            x.append(np.concatenate(descs, axis=-1))
            patch_op_nv = list_patch_op_dir(patch_ops_path)
            # load patch op
            nv = []
            for j in range(len(radius)):
                nv.append(patch_op_nv[patch_op_key(self.names[i], ratio[j], radius[j], nrings[j], ndirs[j])])

            for j in range(len(radius)):
                contributors, weights, transport = load_single_patch_op(dataset_path=patch_ops_path,
                                                                        name=self.names[i],
                                                                        radius=radius[j],
                                                                        nv=nv[j],
                                                                        nrings=nrings[j],
                                                                        ndirs=ndirs[j],
                                                                        ratio=ratio[j])
                x.append(np.expand_dims(contributors, axis=0))
                x.append(np.expand_dims(weights, axis=0))
                x.append(np.expand_dims(transport, axis=0))

            for j in range(len(radius) - 1):
                parent_vertices, angular_shifts = load_single_pool_op(dataset_path=patch_ops_path,
                                                                      name=self.names[i],
                                                                      old_nv=nv[j],
                                                                      new_nv=nv[j + 1],
                                                                      ratio1=ratio[j],
                                                                      ratio2=ratio[j + 1])
                x.append(np.expand_dims(parent_vertices, axis=0))
                x.append(np.expand_dims(angular_shifts, axis=0))
            self.inputs.append(dict(zip(self.keys, x)))



        # self.nsamples = len(self.inputs)
        self.input_dim = self.inputs[0]['input_signal'].shape[-1]
        if self.is_classifier:
            self.preds_dim = num_classes
        else:
            self.preds_dim = self.preds[0].shape[-1]

        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idx = self.indexes[index]
        # Generate data
        X, y = self.__data_generation(idx)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.inputs[indexes].copy()
        if self.add_noise:
            # X['input_signal'] = gaussian_scaling_perturbation(X['input_signal'])
            rot = generate(3)
            a = np.random.rand(1)
            rot = np.multiply((1. + 0.15*(2*a-1.0)), rot)
            X_3d = X['input_signal'][:, :, 0:3]
            X_3d = np.einsum('ijk,lk->ijl', X_3d, rot)
            if X['input_signal'].shape[-1] == 3:
                X['input_signal'] = X_3d
            else:
                X['input_signal'] = np.concatenate([X_3d, X['input_signal'][:, :, 3:]], axis=-1)

        if self.is_classifier:
            y = keras.utils.to_categorical(self.preds[indexes], self.num_classes)
        else:
            y = self.preds[indexes]
        return X, y

    def get_input(self, i):
        return self.inputs[i]

    def get_pred(self, i):
        if self.is_classifier:
            return keras.utils.to_categorical(self.preds[i], self.num_classes)
        else:
            return self.preds[i]

    def get_preds_dim(self):
        return self.preds_dim

    def get_inputs(self):
        return self.inputs

    def get_preds(self):
        return self.preds

    def get_nsamples(self):
        return self.nsamples

    def get_input_dim(self):
        return self.input_dim

    def get_shapes_names(self):
        return self.names

    def is_classifier(self):
        return self.is_classifier


class FullLabelizedDataset(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, nv,
                 num_classes,
                 shapes,
                 desc_path,
                 labels_path,
                 patch_op_path,
                 radius,
                 nrings,
                 ndirs,
                 ratio,
                 shuffle=True):
        'Initialization'
        self.nv = nv
        if nv is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        # save shapes names
        self.names = load_names(shapes)

        # load patch op
        contributors, weights, transport, parents, angular_shifts = load_patch_op(shapes_names_txt=shapes,
                                                                                  shapes_nv=nv,
                                                                                  radius=radius,
                                                                                  nrings=nrings,
                                                                                  ndirs=ndirs,
                                                                                  ratio=ratio,
                                                                                  dataset_path=patch_op_path)

        # load signal
        descriptors = load_descriptors(shapes, desc_path)
        self.nsamples = descriptors.shape[0]
        self.input_dim = descriptors.shape[-1]

        # load labels
        self.labels = load_labels(shapes, labels_path, num_classes, to_categorical=False)

        x = [descriptors]
        self.keys = ['input_signal']
        for j in range(len(contributors)):
            self.keys.append('contributors_' + int_to_string(j))
            self.keys.append('weights_' + int_to_string(j))
            self.keys.append('transport_' + int_to_string(j))
        for j in range(len(parents)):
            self.keys.append('parents_' + int_to_string(j))
            self.keys.append('angular_shifts_' + int_to_string(j))

        for j in range(len(contributors)):
            x.append(contributors[j])
            x.append(weights[j])
            x.append(transport[j])

        for j in range(len(parents)):
            x.append(parents[j])
            x.append(angular_shifts[j])

        self.inputs = dict(zip(self.keys, x))
        # self.nsamples = np.ma.size(self.inputs['input_signal'], axis=0)
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = {k: np.take(v, indexes, axis=0) for k, v in self.inputs.items()}
        y = keras.utils.to_categorical(np.take(self.labels, indexes, axis=0), self.num_classes)
        return X, y

    def get_input(self, i):
        return {k: np.take(v, [i], axis=0) for k, v in self.inputs.items()}

    def get_inputs(self):
        return self.inputs

    def get_labels(self):
        return self.labels

    def get_nsamples(self):
        return self.nsamples

    def get_input_dim(self):
        return self.input_dim

    def get_shapes_names(self):
        return self.names


class ShapeMatchingFullDataset(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, nv,
                 num_classes,
                 shapes,
                 desc_path,
                 labels_path,
                 patch_op_path,
                 radius,
                 nrings,
                 ndirs,
                 ratio,
                 shuffle=True,
                 augment_3d_data=False):
        'Initialization'
        self.augment_3d_data = augment_3d_data
        self.nv = nv
        if nv is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        # load patch op
        contributors, weights, transport, parents, angular_shifts = load_patch_op(shapes_names_txt=shapes,
                                                                                  shapes_nv=nv,
                                                                                  radius=radius,
                                                                                  nrings=nrings,
                                                                                  ndirs=ndirs,
                                                                                  ratio=ratio,
                                                                                  dataset_path=patch_op_path)
        # save shapes names
        self.names = load_names(shapes)

        # load signal
        descriptors = load_descriptors(shapes, desc_path)
        self.nsamples = descriptors.shape[0]
        self.input_dim = descriptors.shape[-1]

        # labels batch
        labels = np.arange(nv[0], dtype=np.int32)
        labels = np.expand_dims(labels, axis=0)
        labels = np.repeat(labels, repeats=batch_size, axis=0)
        self.labels_batch = keras.utils.to_categorical(labels, nv[0])

        x = [descriptors]
        self.keys = ['input_signal']
        for j in range(len(contributors)):
            self.keys.append('contributors_' + int_to_string(j))
            self.keys.append('weights_' + int_to_string(j))
            self.keys.append('transport_' + int_to_string(j))
        for j in range(len(parents)):
            self.keys.append('parents_' + int_to_string(j))
            self.keys.append('angular_shifts_' + int_to_string(j))

        for j in range(len(contributors)):
            x.append(contributors[j])
            x.append(weights[j])
            x.append(transport[j])

        for j in range(len(parents)):
            x.append(parents[j])
            x.append(angular_shifts[j])

        self.inputs = dict(zip(self.keys, x))
        # self.nsamples = np.ma.size(self.inputs['input_signal'], axis=0)

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = {k: np.take(v, indexes, axis=0) for k, v in self.inputs.items()}.copy()
        if self.augment_3d_data:
            rot = generate(3)
            a = np.random.rand(1)
            rot = np.multiply((1. + 0.15 * (2 * a - 1.0)), rot)
            X['input_signal'] = np.einsum('ijk,lk->ijl', X['input_signal'], rot)

        # y = np.take(self.labels, indexes, axis=0)
        y = self.labels_batch
        return X, y

    def get_input(self, i):
        return {k: np.take(v, [i], axis=0) for k, v in self.inputs.items()}

    def get_inputs(self):
        return self.inputs

    def get_nsamples(self):
        return self.nsamples

    def get_input_dim(self):
        return self.input_dim

    def get_shapes_names(self):
        return self.names


class ShapeMatchingFullDatasetMlp(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, nv,
                 shapes,
                 desc_path,
                 shuffle=True):
        'Initialization'
        self.nv = nv
        if nv is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size


        # save shapes names
        self.names = load_names(shapes)

        # load signal
        descriptors = load_descriptors(shapes, desc_path)
        self.nsamples = descriptors.shape[0]
        self.input_dim = descriptors.shape[-1]

        # labels batch
        labels = np.arange(nv[0], dtype=np.int32)
        labels = np.expand_dims(labels, axis=0)
        labels = np.repeat(labels, repeats=batch_size, axis=0)
        self.labels_batch = keras.utils.to_categorical(labels, nv[0])

        x = [descriptors]
        self.keys = ['input_signal']

        self.inputs = dict(zip(self.keys, x))
        # self.nsamples = np.ma.size(self.inputs['input_signal'], axis=0)

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = {k: np.take(v, indexes, axis=0) for k, v in self.inputs.items()}
        # y = np.take(self.labels, indexes, axis=0)
        y = self.labels_batch
        return X, y

    def get_input(self, i):
        return {k: np.take(v, [i], axis=0) for k, v in self.inputs.items()}

    def get_inputs(self):
        return self.inputs

    def get_nsamples(self):
        return self.nsamples

    def get_input_dim(self):
        return self.input_dim

    def get_shapes_names(self):
        return self.names


"""
class RandomNoiseGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, labels, batch_size, nv,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.nv = nv
        self.batch_size = batch_size
        self.labels = labels
        self.inputs = inputs
        self.keys = inputs.keys()
        self.nsamples = np.ma.size(self.inputs['input_signal'], axis=0)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization


        # print('gfyugufu')
        X = dict(zip(self.keys, map(lambda x: np.take(x, indexes, axis=0), self.inputs.values())))
        y = np.take(self.labels, indexes, axis=0)
        # print(np.shape(X['input_signal']))

        # rotate X[0]
        # M = generate(3)
        rotations = np.zeros((self.batch_size, 3, 3))
        for i in range(self.batch_size):
            rotations[i, :, :] = generate(3)

        X['input_signal'] -= np.mean(X['input_signal'], axis=1, keepdims=True)
        # print(np.shape(X['input_signal']))
        # X['input_signal'] = np.dot(X['input_signal'], np.transpose(M))
        X['input_signal'] = np.einsum('ijl,ikl->ijk', X['input_signal'], rotations)
        # print(np.shape(X['input_signal']))






        # y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y
"""