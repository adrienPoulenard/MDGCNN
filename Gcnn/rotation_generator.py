"""Random rotation matrix generators."""

import numpy as np
import keras


def generate(dim):
    """Generate a random rotation matrix.
    Args:
        dim (int): The dimension of the matrix.
    Returns:
        np.matrix: A rotation matrix.
    Raises:
        ValueError: If `dim` is not 2 or 3.
    """
    if dim == 2:
        return generate_2d()
    elif dim == 3:
        return generate_3d()
    else:
        raise ValueError('Dimension {} is not supported. Use 2 or 3 instead.'
                         .format(dim))


def generate_2d():
    """Generate a 2D random rotation matrix.
    Returns:
        np.matrix: A 2D rotation matrix.
    """
    x = np.random.random()
    M = np.matrix([[np.cos(2 * np.pi * x), -np.sin(2 * np.pi * x)],
                   [np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]])
    return M


def generate_3d():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M


class RandomShapeRotationsGenerator(keras.utils.Sequence):
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

