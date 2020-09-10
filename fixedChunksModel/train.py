# -*- coding: utf-8 -*-
# @Date    : 2020/9/9 14:47
# @Author  : Du Jing
# @FileName: train
# ---- Description ----

import os
import sys
from tqdm import tqdm
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from fixedChunksModel.graph import model
from fixedChunksModel import args
import utils


# 环境设置
np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.device('/gpu:0')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需加内存

utils.base.check_dir(args.model_save_dir)
model_save_path = os.path.join(args.model_save_dir, args.model_version + '.ckpt')
print("[训练集]: %s\n[验证集]: %s\n[模型保存路径]: %s\n" % (args.train_path, args.val_path, model_save_path))


class Train(utils.TrainBase):
    def __init__(self):
        super().__init__()

    def parse_tfrecord(self, example_series):
        _context_features = {
            "label": tf.io.FixedLenFeature([], tf.int64),
            "shape_0": tf.io.FixedLenFeature([], tf.int64),
            "shape_1": tf.io.FixedLenFeature([], tf.int64),
            "shape_2": tf.io.FixedLenFeature([], tf.int64),
        }
        context_features, sequence_features = tf.io.parse_single_sequence_example(
            serialized=example_series,
            context_features=_context_features,
            sequence_features={
                'feature': tf.io.FixedLenSequenceFeature([], tf.string)
            }
        )
        label = tf.cast(context_features["label"], tf.int32)
        n_chunk = tf.cast(context_features["shape_0"], tf.int32, 'n_chunk')
        chunk_size = tf.cast(context_features["shape_1"], tf.int32, 'chunk_size')
        n_feature = tf.cast(context_features["shape_2"], tf.int32, 'n_feature')
        feature = tf.decode_raw(sequence_features['feature'], out_type=tf.float64)
        feature = tf.cast(feature, tf.float32)
        feature = tf.reshape(feature, args.raw_feature_shape)
        return feature, label

    def read_tfrecord(self, file, epoch=None, batch_size=None, isTrain=True):
        if not isTrain and epoch > 1:
            sys.stderr.write('Testing Mode! (epoch should be -1)')
            sys.exit(1)
        if isinstance(file, list):
            example_series = tf.data.TFRecordDataset(file)
        else:
            example_series = tf.data.TFRecordDataset([file])
        epoch_series = example_series.map(self.parse_tfrecord).repeat(epoch)
        epoch_series = epoch_series.shuffle(batch_size * 5)
        batch_series = epoch_series.batch(batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(batch_series)
        return iterator

    def train(self):
        with tf.get_default_graph().as_default() as graph:
            # 加载数据
            train_iterator = self.read_tfrecord(args.train_path, args.epoch, args.train_batch, True)
            train_x, train_y = train_iterator.get_next()
            val_iterator = self.read_tfrecord(args.val_path, -1, args.val_batch, False)
            val_x, val_y = val_iterator.get_next()

            with tf.compat.v1.Session(config=config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())
                sess.run(train_iterator.initializer)
                sess.run(val_iterator.initializer)
                # global_step = sess.run(model.global_step)
                # Saver = tf.train.Saver()
                #
                # # Tensorboard可视化
                # train_summary_writer = tf.summary.FileWriter(args.train_tensorboard_path, graph=tf.get_default_graph())
                # val_summary_writer = tf.summary.FileWriter(args.val_tensorboard_path, graph=tf.get_default_graph())

                try:
                    tbar = tqdm(range(args.epoch * args.train_size // args.train_batch + 2))
                    for _i in tbar:
                        traindataBatch, trainlabelBatch = sess.run([train_x, train_y])
                        print(traindataBatch.shape)
                        # feed_dict_train = {model.x_input: traindataBatch,
                        #                    model.y_true: trainlabelBatch,
                        #                    model.lstm_output_keep_prob: 1-args.dropout,
                        #                    model.training: True}
                        # _, global_step, lrate = sess.run([model.train_op, model.global_step, model.lr],
                        #                                  feed_dict=feed_dict_train)
                        # if global_step % 100 == 0:
                        #     y_pred_train, train_acc, train_loss, train_summary = sess.run(
                        #         [model.y_pred, model.accuracy, model.loss, model.merged_summary_op],
                        #         feed_dict=feed_dict_train)
                        #     train_summary_writer.add_summary(train_summary, global_step=global_step)
                        #     tbar.set_description("step: %d" % global_step)
                        #     tbar.set_postfix_str("lr: %.10f, acc: %s, loss: %s" % (lrate, train_acc, train_loss))

                        # <editor-fold desc="Validate">
                        # if global_step % 20 == 0 and global_step != 0:
                        #     tqdm.write("\n============================== val ==============================")
                        #     valData, valLabel, valx, valy = sess.run([val_x, val_y, val_nframe, val_ndim])
                        #     feed_dict_val = {model.x_input: valData,
                        #                      model.y_true: valLabel,
                        #                      model.seqLen: valx,
                        #                      model.dropout: 0,
                        #                      model.training: False}
                        #     y_pred_val, val_acc, val_loss, val_summary = sess.run(
                        #         [model.y_pred, model.accuracy, model.loss, model.merged_summary_op],
                        #         feed_dict=feed_dict_val)
                        #     val_summary_writer.add_summary(val_summary, global_step=global_step)
                        #     print("\n", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                        #           "[Validation] [step]: %d    [loss]: %s    [acc]: %s    " % (
                        #               global_step, val_loss, val_acc))
                        #     print(classification_report(y_true=valLabel, y_pred=y_pred_val))
                        #     print(confusion_matrix(y_true=valLabel, y_pred=y_pred_val))
                        # </editor-fold>


                except tf.errors.OutOfRangeError as e:
                    print("结束！")
                finally:
                    # Saver.save(sess, save_path=model_save_path, global_step=global_step)
                    print(1)

if __name__ == '__main__':
    train_op = Train()
    train_op.train()
