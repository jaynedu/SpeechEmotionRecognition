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
from fixedChunksModel import args
import utils
import module


class Model(utils.ModelBase):
    def __init__(self):
        super().__init__()
        # 环境设置
        np.set_printoptions(threshold=np.inf)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.device('/gpu:0')
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True  # 按需加内存

        utils.base.check_dir(args.model_save_dir)
        self.model_save_path = os.path.join(args.model_save_dir, args.model_version + '.ckpt')
        print("[训练集]: %s\n[验证集]: %s\n[模型保存路径]: %s\n" % (args.train_path, args.val_path, self.model_save_path))

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

    def setup(self, model_type):
        with tf.get_default_graph().as_default() as graph:

            # load data
            train_iterator = self.read_tfrecord(args.train_path, args.epoch, args.train_batch, True)
            train_x, train_y = train_iterator.get_next()
            val_iterator = self.read_tfrecord(args.val_path, -1, args.val_batch, False)
            val_x, val_y = val_iterator.get_next()

            # define parameters
            self.x_input = tf.placeholder(tf.float32, [None] + args.raw_feature_shape, name='x_input')
            self.y_true = tf.placeholder(tf.int32, [None, ], name='y_true')
            self.lstm_output_keep_prob = tf.placeholder(tf.float32)
            self.type = model_type
            training = tf.placeholder(tf.bool, name='training')

            # build network
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
            output = tf.layers.batch_normalization(output, training=training)
            print(output)

            with tf.name_scope("sentence_representation"):
                if self.type == "NonAtten":
                    output = tf.keras.layers.AveragePooling1D(9, 1)(output)
                elif self.type == "GatedVec":
                    weights = tf.get_variable(shape=(9, 130), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1), name="weights")
                    bias = tf.get_variable(shape=(9, 1), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1), name="bias")
                    g = tf.sigmoid(tf.matmul(weights, output, transpose_b=True) + bias)
                    output = tf.reduce_sum(tf.matmul(g, output), axis=1)
                elif self.type == "AttenVec":
                    output = tf.reshape(output, [-1, 1, args.lstm_units[-1]])
                    output= tf.keras.layers.SimpleRNN(130, dropout=args.dropout)(output)
                    output = tf.reshape(output, [-1, 9, 130])
                    # output = module.attention.time_attention(state, output)
                    output = module.attention.feature_attention(output)
                    print(output)
                else:
                    print("Model Type is %s! Available type: NonAtten, GatedVec, AttenVec." % self.type)
                    sys.exit(1)

                print(output)

            with tf.name_scope("output"):
                output = tf.layers.flatten(output)
                for unit in args.output_dense_units:
                    output = tf.layers.dense(output, unit)

                self.logits = output
                self.logits_softmax = tf.nn.softmax(self.logits, name='logits_softmax')

            module.utils.params_usage(tf.trainable_variables())

            # sys.exit(0)

            # define train operations
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

            # begin train
            with tf.compat.v1.Session(config=self.config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())
                sess.run(train_iterator.initializer)
                sess.run(val_iterator.initializer)
                global_step = sess.run(self.global_step)
                Saver = tf.train.Saver()

                # Tensorboard可视化
                train_summary_writer = tf.summary.FileWriter(args.train_tensorboard_path, graph=tf.get_default_graph())
                val_summary_writer = tf.summary.FileWriter(args.val_tensorboard_path, graph=tf.get_default_graph())

                try:
                    tbar = tqdm(range(args.epoch * args.train_size // args.train_batch + 2))
                    for _i in tbar:
                        traindataBatch, trainlabelBatch = sess.run([train_x, train_y])
                        feed_dict_train = {self.x_input: traindataBatch,
                                           self.y_true: trainlabelBatch,
                                           self.lstm_output_keep_prob: 1-args.dropout,
                                           training: True}
                        _, global_step, lrate = sess.run([self.train_op, self.global_step, self.lr],
                                                         feed_dict=feed_dict_train)
                        if global_step % 100 == 0:
                            y_pred_train, train_acc, train_loss, train_summary = sess.run(
                                [self.y_pred, self.accuracy, self.loss, self.merged_summary_op],
                                feed_dict=feed_dict_train)
                            train_summary_writer.add_summary(train_summary, global_step=global_step)
                            tbar.set_description("step: %d" % global_step)
                            tbar.set_postfix_str("lr: %.10f, acc: %s, loss: %s" % (lrate, train_acc, train_loss))

                        if global_step % 20 == 0 and global_step != 0:
                            tqdm.write("\n============================== val ==============================")
                            valData, valLabel= sess.run([val_x, val_y])
                            feed_dict_val = {self.x_input: valData,
                                             self.y_true: valLabel,
                                             self.lstm_output_keep_prob: 1,
                                             training: False}
                            y_pred_val, val_acc, val_loss, val_summary = sess.run(
                                [self.y_pred, self.accuracy, self.loss, self.merged_summary_op],
                                feed_dict=feed_dict_val)
                            val_summary_writer.add_summary(val_summary, global_step=global_step)
                            print("\n", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                                  "[Validation] [step]: %d    [loss]: %s    [acc]: %s    " % (
                                      global_step, val_loss, val_acc))
                            print(classification_report(y_true=valLabel, y_pred=y_pred_val))
                            print(confusion_matrix(y_true=valLabel, y_pred=y_pred_val))

                except tf.errors.OutOfRangeError as e:
                    print("结束！")
                finally:
                    Saver.save(sess, save_path=self.model_save_path, global_step=global_step)

if __name__ == '__main__':
    model = Model()
    model.setup("AttenVec")
