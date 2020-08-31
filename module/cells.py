# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 10:24 下午
# @Author  : Du Jing
# @FileName: cells
# ---- Description ----
#

import tensorflow as tf

__all__ = [
    'AdvancedLSTMCell',
    'AttentionConvLSTMCell',
    'AttentionLSTMCell',
    'BasicConvLSTMCell'
]

initializer = tf.truncated_normal_initializer(stddev=0.1)


class AdvancedLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 use_peepholes=True,
                 use_bias=True,
                 normalize=True,
                 name="AdvancedLSTMCell",
                 reuse=None):
        super(AdvancedLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._normalize = normalize
        self._use_peepholes = use_peepholes
        self._use_bias = use_bias
        self._name = name

        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        # inputs = [batch, timestep, features]
        # TODO:
        #  当前是普通的LSTM实现
        #  要改成对时间步进行attention的

        c, h = state
        x = tf.concat([inputs, h], -1)

        input_size = x.get_shape().as_list()[-1]
        kernel = tf.get_variable('kernel', [input_size, 4 * self._num_units], tf.float32, initializer)
        mul = tf.matmul(x, kernel)
        if self._use_bias:
            mul += tf.get_variable('bias', mul.shape[-1], tf.float32, tf.constant_initializer(0.1))

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=mul, num_or_size_splits=4, axis=1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f + self._forget_bias)
        c = f * c + i * tf.tanh(j)
        o = tf.sigmoid(o)
        h = o * tf.tanh(c)

        state = tf.contrib.rnn.LSTMStateTuple(c, h)
        return h, state


class AttentionConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 kernel_size=3,
                 forget_bias=1.0,
                 use_peepholes=True,
                 use_bias=True,
                 normalize=True,
                 name="AttentionConvLSTMCell",
                 reuse=None):
        super(AttentionConvLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._kernel_size = kernel_size
        self._forget_bias = forget_bias
        self._normalize = normalize
        self._use_peepholes = use_peepholes
        self._use_bias = use_bias
        self._name = name

        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        # inputs = [batch, timestep, features]

        c, h = state
        x = tf.concat([inputs, h], -1)
        expand = tf.expand_dims(x, 0)

        input_size = x.get_shape().as_list()[-1]
        kernel = tf.get_variable('kernel', [self._kernel_size] + [input_size, 2 * self._num_units], tf.float32,
                                 initializer)

        conv = tf.nn.conv1d(expand, kernel, 1, 'SAME', data_format='NWC')
        conv = tf.squeeze(conv, [0])

        if self._use_bias:
            conv += tf.get_variable('bias', conv.shape[-1], tf.float32, tf.constant_initializer(0.1))

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        j, o = tf.split(value=conv, num_or_size_splits=2, axis=1)

        with tf.variable_scope('attention_gate'):
            _w = tf.get_variable('w_attention', (self._num_units, self._num_units), tf.float32)
            _v = tf.get_variable('v_attention', (self._num_units, self._num_units), tf.float32)
            alpha = tf.matmul(c, _w)
            alpha = tf.matmul(tf.tanh(alpha), _v)
            c = tf.sigmoid(alpha + self._forget_bias) * c + (1 - tf.sigmoid(alpha)) * tf.tanh(j)

        o = tf.sigmoid(o)
        h = o * tf.tanh(c)

        state = tf.contrib.rnn.LSTMStateTuple(c, h)
        return h, state


from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.ops import partitioned_variables
class AttentionLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None):

        super(AttentionLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            tf.logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        self._num_units = num_units
        self._use_peepholes = False
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.python.math_ops.tanh

        if num_proj:
            self._state_size = (
                tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units
        self._linear1 = None
        self._linear2 = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):

        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = tf.python.math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = tf.python.array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = tf.python.array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        if self._linear1 is None:
            scope = tf.get_variable_scope()
            with tf.variable_scope(
                    scope, initializer=self._initializer) as unit_scope:
                if self._num_unit_shards is not None:
                    unit_scope.set_partitioner(
                        partitioned_variables.fixed_size_partitioner(
                            self._num_unit_shards))
                self._linear1 = _Linear([inputs, m_prev], 2 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = self._linear1([inputs, m_prev])
        j, o = tf.python.array_ops.split(
            value=lstm_matrix, num_or_size_splits=2, axis=1)

        with tf.variable_scope("attention_gate"):
            _W = tf.get_variable("W_atten", (self._num_units, self._num_units), tf.float32)
            _V = tf.get_variable("V_atten", (self._num_units, self._num_units), tf.float32)
            fattn = tf.matmul(tf.tanh(tf.matmul(c_prev, _W)), _V)
            c = sigmoid(fattn + self._forget_bias) * c_prev + (1 - sigmoid(fattn)) * self._activation(j)

        m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            if self._linear2 is None:
                scope = tf.get_variable_scope()
                with tf.variable_scope(scope, initializer=self._initializer):
                    with tf.variable_scope("projection") as proj_scope:
                        if self._num_proj_shards is not None:
                            proj_scope.set_partitioner(
                                partitioned_variables.fixed_size_partitioner(
                                    self._num_proj_shards))
                        self._linear2 = _Linear(m, self._num_proj, False)
            m = self._linear2(m)

            if self._proj_clip is not None:
                m = tf.python.clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)

        new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, m) if self._state_is_tuple else
                     tf.python.array_ops.concat([c, m], 1))
        return m, new_state


class BasicConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 kernel_size=3,
                 forget_bias=1.0,
                 use_peepholes=True,
                 use_bias=True,
                 normalize=True,
                 name="BasicConvLSTMCell",
                 reuse=None):
        super(BasicConvLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._kernel_size = kernel_size
        self._forget_bias = forget_bias
        self._normalize = normalize
        self._use_peepholes = use_peepholes
        self._use_bias = use_bias
        self._name = name

        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        # inputs = [batch, timestep, features]

        c, h = state
        x = tf.concat([inputs, h], -1)
        expand = tf.expand_dims(x, 0)

        input_size = x.get_shape().as_list()[-1]
        kernel = tf.get_variable('kernel', [self._kernel_size] + [input_size, 4 * self._num_units], tf.float32,
                                 initializer)

        conv = tf.nn.conv1d(expand, kernel, 1, 'SAME', data_format='NWC')
        conv = tf.squeeze(conv, [0])
        if self._use_bias:
            conv += tf.get_variable('bias', conv.shape[-1], tf.float32, tf.constant_initializer(0.1))

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=conv, num_or_size_splits=4, axis=1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f + self._forget_bias)
        c = f * c + i * tf.tanh(j)
        o = tf.sigmoid(o)
        h = o * tf.tanh(c)

        state = tf.contrib.rnn.LSTMStateTuple(c, h)
        return h, state
