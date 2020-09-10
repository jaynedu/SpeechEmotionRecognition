# -*- coding: utf-8 -*-
# @Date    : 2020/9/9 14:45
# @Author  : Du Jing
# @FileName: graph
# ---- Description ----

import sys
import tensorflow as tf
from fixedChunksModel import args
import module


class Graph:
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, [None] + args.raw_feature_shape, name='x_input')
        self.y_true = tf.placeholder(tf.int32, [None, ], name='y_true')
        self.lstm_output_keep_prob = tf.placeholder(tf.float32)
        self.type = None

    def build(self, training):
        nc, cs, nf = args.raw_feature_shape
        # input = [batch, n_chunk, chunk_size, n_feature] -> [batch, time_step, n_feature]
        output = tf.reshape(self.x_input, [-1, cs, nf])
        # output = self.x_input
        with tf.name_scope("shared_lstm"):
            dropout_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(unit, use_peepholes=True, reuse=tf.AUTO_REUSE),
                output_keep_prob=self.lstm_output_keep_prob
            ) for unit in args.lstm_units]
            lstm = tf.nn.rnn_cell.MultiRNNCell(dropout_cells)
            output, laststate = tf.nn.dynamic_rnn(lstm, output, dtype=tf.float32)

        output = tf.reshape(laststate[-1][-1], [-1, nc, args.lstm_units[-1]])
        print(output)
        output = tf.layers.batch_normalization(output, training=training)

        with tf.name_scope("sentence_representation"):
            if self.type == "NonAtten":
                output = tf.keras.layers.AveragePooling1D(1, 1)(output)
            elif self.type == "GatedVec":
                g = tf.layers.dense(output, 130, 'sigmoid')
                output = tf.reduce_sum(tf.matmul(g, output, transpose_b=True), axis=-1)
            elif self.type == "AttenVec":
                output, state = tf.keras.layers.SimpleRNN(130, return_state=True, dropout=args.dropout if training else 0)
                output = module.attention.time_attention(state, output)
            else:
                print("Model Type is %s! Available type: NonAtten, GatedVec, AttenVec." % self.type)
                self.type = input("Input type:")
                self.build(training)

        with tf.name_scope("output"):
            output = tf.layers.flatten(output)
            for unit in args.output_dense_units:
                output = tf.layers.dense(output, unit)

            self.logits = output
            self.logits_softmax = tf.nn.softmax(self.logits, name='logits_softmax')

        if training:
            module.utils.params_usage(tf.trainable_variables())

        y = tf.one_hot(self.y_true, args.n_label)
        self.y_smooth = module.base.label_smoothing(y)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_smooth, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.global_step = tf.train.get_or_create_global_step()
        self.lr = module.base.noam_scheme(args.eta, global_step=self.global_step, warmup_steps=args.warmup)
        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([train_op, update_ops])
        self.y_pred = tf.argmax(self.logits_softmax, axis=1, name="y_pred")
        pred_prob = tf.equal(tf.cast(self.y_pred, tf.int32), self.y_true)
        self.accuracy = tf.reduce_mean(tf.cast(pred_prob, tf.float32), name="accuracy")

        tf.compat.v1.summary.scalar('accuracy', self.accuracy)
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('learning rate', self.lr)
        self.merged_summary_op = tf.compat.v1.summary.merge_all()


    def __call__(self, type, training=True):
        self.type = type
        self.build(training)


model = Graph()

if __name__ == '__main__':
    model(None)


