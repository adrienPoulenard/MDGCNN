import keras
from keras import backend as K
import time
import logging
import numpy as np

import os
import scipy.sparse as sp
import tensorflow as tf
from os import listdir
from os.path import isfile, join


def save_tensor(path, x):
    np.savetxt(path, x.flatten(), delimiter=' ', fmt='%f')


def save_matrix(path, x, dtype=np.float32):
    if dtype is np.float32:
        np.savetxt(path, x, delimiter=' ', fmt='%f')
    if dtype is np.int32:
        np.savetxt(path, x.astype(np.int32), delimiter=' ', fmt='%i')
