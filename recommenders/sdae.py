
"""
Custom pieces for better handling sparsity
"""

import tensorflow as tf


@tf.function
def get_sparsity_weights(y_true):
    """
    Elementwise weights for balancing loss w.r.t sparsity
    :param y_true: Label tensor
    :return:
    """
    sparse_frac = tf.math.zero_fraction(y_true)

    if not isinstance(y_true, tf.SparseTensor):
        # convert to sparse to find the non-zero indices
        sparsed = tf.sparse.from_dense(y_true)
    else:
        sparsed = y_true

    # create a dense weight tensor by first creating a copy of the sparsified
    # labels with values replaced by sparse_frac, then converting back to dense
    #  and setting the default vals to (1-sparse). when element-wise multiplied
    # with the loss tensor, this will weigh each loss term by the complement
    # of its prevalence, balancing the contribution of zero and
    # non-zero elements in the batch-level loss
    non_zero_weights = tf.sparse.SparseTensor(
            indices=sparsed.indices,
            values=tf.constant(sparse_frac, shape=(len(sparsed.indices))),
            dense_shape=sparsed.dense_shape
    )

    weight_tensor = tf.sparse.to_dense(
            non_zero_weights,
            default_value=tf.constant(1.0)-tf.constant(sparse_frac)
    )

    return weight_tensor


@tf.function
def sparse_mean_absolute_error(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    batch_sparsity_weights = get_sparsity_weights(y_true)
    loss = tf.math.abs(y_true - y_pred)

    return tf.reduce_mean(tf.math.multiply(loss, batch_sparsity_weights))


@tf.function
def sparse_mean_squared_error(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    batch_sparsity_weights = get_sparsity_weights(y_true)
    loss = tf.math.squared_difference(y_true, y_pred)

    return tf.reduce_mean(tf.math.multiply(loss, batch_sparsity_weights))
