# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 10:14 下午
# @Author  : Du Jing
# @FileName: base
# ---- Description ----
#

import tensorflow as tf
import numpy as np

__all__ = [
    'feed_forward',
    'residual_connection',
    'position_encoding',
    'gelu',
    'label_smoothing',
    'noam_scheme',
]


def feed_forward(inputs, units: list, dropout, training: bool):
    inner = tf.layers.dense(inputs, units[0])
    activated_inner = gelu(inner)
    outputs = tf.layers.dense(activated_inner, units[1])
    if training is not None and training is True:
        outputs = tf.layers.dropout(outputs, dropout, training=training)
    return outputs


def residual_connection(x, fx, training):
    outputs = x + fx
    outputs = tf.layers.batch_normalization(outputs, training=training)
    return outputs


def position_encoding(inputs, maxlen, masking=True, scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (batch_size, seq_len, nfeature)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    nfeature = inputs.get_shape().as_list()[-1]  # static
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - (i % 2)) / nfeature) for i in range(nfeature)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, nfeature)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def gelu(input_tensor):
    '''Gaussian Error Linear Unit.'''
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def label_smoothing(inputs, epsilon=0.1):
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)