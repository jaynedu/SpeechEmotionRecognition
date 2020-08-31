# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 9:16 下午
# @Author  : Du Jing
# @FileName: utils
# ---- Description ----
#

import os
import sys
import tensorflow as tf

__all__ = [
    'check_dir',
    'clear',
    'saveTFrecord',
    'readTFrecord',
    'createWriter',
    'disposeWriter'
]

def clear(*args):
    for arg in args:
        try:
            os.remove(arg)
        except FileNotFoundError:
            print("文件 [%s] 不存在!" % arg)


def check_dir(dir):
    if not os.path.exists(dir):
        parent = os.path.split(dir)[0]
        check_dir(parent)
        os.mkdir(dir)


def bytes_feature(value):
    """生成字符串型的属性"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    """生成整数型的属性"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    """生成实数型的属性"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def createWriter(path):
    return tf.io.TFRecordWriter(path)


def disposeWriter(writer: tf.io.TFRecordWriter):
    writer.close()


def saveTFrecord(feature, label, writer):
    nframe, ndim = feature.shape
    context_dict = {
        'label': int64_feature(label),
        'ndim': int64_feature(ndim),
        'nframe': int64_feature(nframe),
    }
    feature_list = {
        'feature': tf.train.FeatureList(feature=[bytes_feature(f.tobytes()) for f in feature])
    }
    tf_example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_dict),
        feature_lists=tf.train.FeatureLists(feature_list=feature_list)
    )
    tf_serial = tf_example.SerializeToString()
    writer.write(tf_serial)


def _parseTFrecord(example_series):
    context_features, sequence_features = tf.io.parse_single_sequence_example(
        serialized=example_series,
        context_features={
            "label": tf.io.FixedLenFeature([], tf.int64),
            "ndim": tf.io.FixedLenFeature([], tf.int64),
            "nframe": tf.io.FixedLenFeature([], tf.int64),
        },
        sequence_features={
            'feature': tf.io.FixedLenSequenceFeature([], tf.string)
        }
    )
    label = tf.cast(context_features["label"], tf.int32)
    ndim = tf.cast(context_features["ndim"], tf.int32, 'ndim')
    nframe = tf.cast(context_features["nframe"], tf.int32, 'nframe')
    feature = tf.decode_raw(sequence_features['feature'], out_type=tf.float64)
    feature = tf.cast(feature, tf.float32)
    return feature, label, ndim, nframe


def readTFrecord(file, nfeature, seq_length=None, epoch=None, batch_size=None, isTrain=True):
    if not isTrain and epoch > 1:
        sys.stderr.write('Testing Mode! (epoch should be -1)')
        sys.exit(1)
    if isinstance(file, list):
        example_series = tf.data.TFRecordDataset(file)
    else:
        example_series = tf.data.TFRecordDataset([file])
    epoch_series = example_series.map(_parseTFrecord).repeat(epoch)
    epoch_series = epoch_series.shuffle(batch_size * 5)
    padded_batch_series = epoch_series.padded_batch(batch_size, ([seq_length, nfeature], [], [], []))
    iterator = tf.compat.v1.data.make_initializable_iterator(padded_batch_series)
    return iterator