# -*- coding: utf-8 -*-
# @Date    : 2020/8/30 4:27 下午
# @Author  : Du Jing
# @FileName: data
# ---- Description ----
#

import os
import contextlib
import json
import tensorflow as tf
import numpy as np
import utils

class Data(object):
    def __init__(self, jsonPath):
        self.jsonPath = jsonPath

    def load_json(self, selectedLabels):
        paths = []
        labels = []
        with contextlib.closing(open(self.jsonPath, 'r')) as rf:
            content = json.load(rf)['list']
            for item in content.values():
                if isinstance(selectedLabels, list):
                    if item['label'] in selectedLabels:
                        paths.append(item['path'])
                        labels.append(item['label'])
                elif selectedLabels is None:
                    paths.append(item['path'])
                    labels.append(item['label'])
        return paths, labels

    @staticmethod
    def generate_testset(self, tfrecordPath, nFeature, seqLength, totalSize):
        test_iterator = utils.tfrecord.readTFrecord(tfrecordPath, nFeature, seqLength, 1, totalSize, False)
        x, y, ndim, nframe = test_iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(test_iterator.initializer)
            inputs, labels, dim, seqlen = sess.run([x, y, ndim, nframe])
            save_path = os.path.splitext(tfrecordPath)[0] + '.npz'
            np.savez(save_path, inputs=inputs, labels=labels, dim=dim, seqlen=seqlen)

        return save_path