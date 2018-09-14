from keras import backend as K
import tensorflow as tf
import numpy as np
_EPSILON = 1e-7


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.
    # Returns
        A float.
    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON


def categorical_cross_entropy_bis_(y_true, y_pred):
    # reshape entries

    print('shape !!!')
    print(y_pred.get_shape())
    print(K.shape(y_true))
    nbatch = y_pred.get_shape()[0]
    nv = y_pred.get_shape()[1]
    nclasses = y_pred.get_shape()[2]
    print(nbatch)
    print(nv)
    print(nclasses)

    y_true = K.reshape(y_true, (nbatch * nv, nclasses))
    y_pred = K.reshape(y_pred, (nbatch * nv, nclasses))
    print('new shapes !!!')
    print(y_pred.get_shape())
    print(y_true.get_shape())
    return K.categorical_crossentropy(y_true, y_pred)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def categorical_cross_entropy_bis(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        l1norm = tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keep_dims=True)
        print('output shape')
        print(output.get_shape())
        output /= l1norm
        print('output normalized shape')
        print(output.get_shape())
        print('l1norm shape')
        print(l1norm.get_shape())
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        res = - tf.reduce_sum(target * tf.log(output), axis=None)

        print('res shape')
        print(res.get_shape())
        return res
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


def categorical_accuracy_bis(y_true, y_pred):
    # reshape entries
    # nbatch = y_pred.get_shape()[0]
    # nv = y_pred.get_shape()[1]
    # nclasses = y_pred.get_shape()[2]
    # y_true = K.reshape(y_true, (nbatch * nv, nclasses))
    # y_pred = K.reshape(y_pred, (nbatch * nv, nclasses))
    res = K.cast(K.equal(K.argmax(y_true, axis=-1),
                         K.argmax(y_pred, axis=-1)),
                 K.floatx())
    print('metrics res shape')
    print(res.get_shape())
    return res

