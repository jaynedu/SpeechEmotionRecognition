# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:47 下午
# @Author  : Du Jing
# @FileName: test
# ---- Description ----
#

import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import utils
from transformer import args
from transformer.graph import model

# 加载数据集
test_data_path = utils.audio.AudioUtils.generate_testset(args.test_path, args.nfeature, args.seq_length, args.test_size)
test_data = np.load(test_data_path)
testData, testLabel, testx, testy = test_data['inputs'], test_data['labels'], test_data['seqlen'], test_data['dim'],
feed_dict_test = {model.x_input: testData,
                  model.y_true: testLabel,
                  model.seqLen: testx,
                  model.dropout: 0,
                  model.training: False}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if tf.train.checkpoint_exists(args.model_save_dir):
        model_file = tf.train.latest_checkpoint(args.model_save_dir)
        saver.restore(sess, model_file)
        y_pred_test, test_acc, test_loss = sess.run([model.y_pred, model.accuracy, model.loss],
                                                    feed_dict=feed_dict_test)
        print(classification_report(y_true=testLabel, y_pred=y_pred_test))
        matrix = confusion_matrix(y_true=testLabel, y_pred=y_pred_test)
        print(matrix)
        if args.plot_matrix:
            utils.draw.plot_confusion_matrix(matrix, args.classes, args.figure_title, args.figure_save_path)
    else:
        sys.stderr.write('Checkpoint Not Found!')
        sys.exit(1)