from keras.regularizers import Regularizer
from keras import backend as K


def angular_diff_reg(x, a, norm='l2'):
    x_ = x[:, 1, :, :]
    x_ = K.expand_dims(x_, axis=1)
    # x_ = K.repeat_elements(x_, rep=K.shape(x)[1], axis=1)

    y = K.concatenate([x[:, 1:, :, :], x_], axis=1)
    z = x - y
    if norm is 'l1':
        return K.sum(a * K.abs(z))
    else:
        return K.sum(a * K.square(z))


class L1L2_circ(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0., l1_d=0., l2_d=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.l1_d = K.cast_to_floatx(l1_d)
        self.l2_d = K.cast_to_floatx(l2_d)

    def __call__(self, x):
        regularization = 0.
        if K.ndim(x) == 2:
            if self.l1:
                regularization += K.sum(self.l1 * K.abs(x))
            if self.l2:
                regularization += K.sum(self.l2 * K.square(x))
        else:
            if self.l1:
                regularization += K.sum(self.l1 * K.abs(x))
            if self.l2:
                regularization += K.sum(self.l2 * K.square(x))
            if self.l1_d:
                regularization += angular_diff_reg(x, self.l1_d, norm='l1')
            if self.l2_d:
                regularization += angular_diff_reg(x, self.l2_d, norm='l2')

        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'l1_d': float(self.l1_d),
                'l2_d': float(self.l2_d)}
