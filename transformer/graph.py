# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:34 下午
# @Author  : Du Jing
# @FileName: graph
# ---- Description ----
#


import tensorflow as tf
import module
from transformer import args


class Graph(object):
    def __init__(self):
        self.feature_dimension = args.nfeature
        self.sequence_length = args.seq_length

        self.x_input = tf.placeholder(tf.float32, [None, args.seq_length, args.nfeature], name='x_input')
        self.y_true = tf.placeholder(tf.int32, [None, ], name='y_true')
        self.seqLen = tf.placeholder(tf.int32, name='seq_len')  # 用于存储每个样本中timestep的数目
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.training = tf.placeholder(tf.bool, name='training')

        self.build()

    def build(self):
        position_encoding_outputs = module.base.position_encoding(self.x_input, args.position_size)
        if args.position_encoding_type == 'add':
            outputs = position_encoding_outputs + self.x_input
        if args.position_encoding_type == "concat":
            outputs = tf.concat([self.x_input, position_encoding_outputs], axis=2)

        for i in range(6):
            sublayer1 = module.attention.multi_head_attention(outputs, outputs, outputs, args.head_num, args.head_size,
                                                     self.dropout, self.training, type=args.attention_unit_type)
            outputs = module.base.residual_connection(outputs, sublayer1, self.training)
            sublayer2 = module.base.feed_forward(outputs, args.feed_forward_size, self.dropout, self.training)
            outputs = module.base.residual_connection(outputs, sublayer2, self.training)

        outputs = tf.layers.dense(outputs, 1, use_bias=True, name='last_output')
        outputs = tf.squeeze(outputs, -1)  # (batch_size, seqlen)
        outputs = tf.layers.dense(outputs, args.nlabel, name='output_logit')

        self.logits = outputs
        self.logits_softmax = tf.nn.softmax(outputs, name='output_logit_softmax')

        if self.training is not None:
            module.utils.params_usage(tf.trainable_variables())

        y = tf.one_hot(self.y_true, args.nlabel)
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


model = Graph()
