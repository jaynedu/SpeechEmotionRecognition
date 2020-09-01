# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 10:25 下午
# @Author  : Du Jing
# @FileName: attention
# ---- Description ----
#

# -*- coding: utf-8 -*-
# @Date: 2020/7/23 12:41
# @Author: Du Jing
# @FileName: attention
# ---- Description ----
#

import tensorflow as tf

__all__ = [
    'scaled_dot_product_attention',
    'linear_attention_kernel',
    'linear_attention_multiple_softmax',
    'linear_attention_taylor',
    'multi_head_attention',
    'time_attention',
    'feature_attention',
]

def scaled_dot_product_attention(Q, K, V, dropout, training, scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, K, transpose_b=True)  # dot product
        outputs /= (d_k ** 0.5)  # scale
        outputs = tf.nn.softmax(outputs)  # softmax
        # draw
        score = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention score", tf.expand_dims(score[:1], -1))

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)
        outputs = tf.matmul(outputs, V)  # weighted sum (batch_size, seqlen, d_v)
        return outputs


def scaled_dot_product_attention_handmade(Q, K, V, dropout, training, scope="scaled_dot_product_attention"):
    '''
    手写实现缩放点积注意力，不用tf里的softmax
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, K, transpose_b=True)  # dot product
        outputs /= (d_k ** 0.5)  # scale

        z = outputs - tf.reduce_max(outputs, axis=2, keepdims=True)
        z = tf.exp(z)
        outputs = z / tf.reduce_sum(z, axis=2, keepdims=True)

        # draw
        score = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention score", tf.expand_dims(score[:1], -1))

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        outputs = tf.matmul(outputs, V)  # weighted sum (batch_size, seqlen, d_v)
        return outputs


def linear_attention_taylor(Q, K, V, dropout, training, scope="linear_attention_tayler"):
    eps = 1e-7
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        n_k, d_k = Q.get_shape().as_list()[1:]
        _Q = tf.nn.l2_normalize(Q)
        _K = tf.nn.l2_normalize(K)

        outputs = (V + tf.matmul(_Q, tf.matmul(_K, V, transpose_a=True)) / d_k ** 0.5) / \
                  (tf.reduce_sum(tf.cast(n_k, tf.float32) + tf.matmul(_Q, _K, transpose_b=True) / d_k**0.5, axis=2, keepdims=True) + eps)

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        return outputs


def linear_attention_kernel(Q, K, V, dropout, training, scope="linear_attention_kernel"):
    '''
    Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
    https://arxiv.org/abs/2006.16236
    通过核函数定义内积
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _Q = tf.nn.elu(Q) + 1
        _K = tf.nn.elu(K) + 1
        outputs = tf.matmul(_Q, tf.matmul(_K, V, transpose_a=True))/tf.reduce_sum(tf.matmul(_Q, _K, transpose_b=True), axis=2, keepdims=True)

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        return outputs


def linear_attention_multiple_softmax(Q, K, V, dropout, training, scope="linear_attention_multiple_softmax"):
    '''
    Efficient Attention: Attention with Linear Complexities
    https://arxiv.org/abs/1812.01243
    分别对Q、K进行softmax，可视为对不同维的特征加权且效果等价
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        '''手写实现'''
        # _Q = tf.exp(Q)
        # _K = tf.exp(K)
        # Q_ = _Q / tf.reduce_sum(_Q, axis=2, keepdims=True)
        # K_ = _K / tf.reduce_sum(_K, axis=1, keepdims=True)
        # outputs = tf.matmul(Q_, tf.matmul(K_, V, transpose_a=True))

        '''函数实现'''
        _Q = tf.nn.softmax(Q)
        _K = tf.nn.softmax(K)
        outputs = tf.matmul(tf.matmul(_Q, _K, transpose_b=True), V)

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        return outputs


def multi_head_attention(keys, queries, values, head_num, head_size, dropout, training, scope="multi_head_attention",
                         type="softmax"):
    '''
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    '''
    assert type in ["softmax", "taylor", "kernel", "multi_softmax"], print("check the attention unit type!")
    hidden_size = head_num * head_size  # d_model = hidden_size
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, hidden_size, use_bias=True)
        K = tf.layers.dense(keys, hidden_size, use_bias=True)
        V = tf.layers.dense(values, hidden_size, use_bias=True)

        # Split and concat
        _Q = tf.concat(tf.split(Q, head_num, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        _K = tf.concat(tf.split(K, head_num, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        _V = tf.concat(tf.split(V, head_num, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        if type == "softmax":
            score = scaled_dot_product_attention(_Q, _K, _V, dropout=dropout, training=training)
        if type == "taylor":
            score = linear_attention_taylor(_Q, _K, _V, dropout=dropout, training=training)
        if type == "kernel":
            score = linear_attention_kernel(_Q, _K, _V, dropout, training)
        if type == "multi_softmax":
            score = linear_attention_multiple_softmax(_Q, _K, _V, dropout, training)
        outputs = tf.concat(tf.split(score, head_num, axis=0), axis=2)

        return outputs


def feature_attention(output, scope='FeatureAttention'):
    shape_output = output.get_shape().as_list()
    assert len(shape_output) == 3

    with tf.variable_scope(scope):
        w = tf.get_variable('weight', (shape_output[-1], shape_output[-1]), tf.float32,
                            tf.truncated_normal_initializer(stddev=0.1))
        u = tf.get_variable('u', (shape_output[-1], shape_output[-1]), tf.float32,
                            tf.truncated_normal_initializer(stddev=0.1))

    alpha = tf.einsum("aij,jk->aik", tf.tanh(tf.einsum("aij,jk->aik", output, w)), u)
    score = tf.nn.softmax(alpha, dim=-1)
    return tf.reduce_sum(score * output, 1)  # 不能用mean


def time_attention(laststate, output, scope='TimeAttention'):
    shape_laststate = laststate.get_shape().as_list()
    shape_output = output.get_shape().as_list()
    assert len(shape_laststate) == 2 and len(shape_output) == 3

    with tf.variable_scope(scope):
        weights = tf.get_variable('weight', (shape_output[-1], shape_laststate[-1]), tf.float32,
                                  tf.truncated_normal_initializer(stddev=0.1))

    alpha = tf.einsum("aij,jk->aik", output, weights)
    v = tf.matmul(tf.expand_dims(laststate, 1), alpha, transpose_b=True)
    score = tf.nn.softmax(v, dim=-1)
    return tf.squeeze(tf.matmul(score, output), [1])
