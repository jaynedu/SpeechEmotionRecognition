# -*- coding: utf-8 -*-
# @Date    : 2020/9/9 14:45
# @Author  : Du Jing
# @FileName: graph
# ---- Description ----

import tensorflow as tf
from fixedChunksModel import args


class NonAtten:
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
        pass