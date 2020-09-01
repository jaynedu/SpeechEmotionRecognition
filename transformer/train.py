# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:44 下午
# @Author  : Du Jing
# @FileName: train
# ---- Description ----
#

import os
import tensorflow as tf
from tqdm import tqdm
import utils
from transformer import args
from transformer.graph import model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.device('/gpu:0')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

utils.base.check_dir(args.model_save_dir)
model_save_path = os.path.join(args.model_save_dir, args.model_version + '.ckpt')
print("[训练集]: %s\n[验证集]: %s\n[模型保存路径]: %s\n" % (args.train_path, args.val_path, model_save_path))

with tf.get_default_graph().as_default() as graph:
    # 加载数据
    train_iterator = utils.tfrecord.readTFrecord(args.train_path, model.feature_dimension, model.sequence_length, args.epoch,
                                       args.train_batch,
                                       True)
    train_x, train_y, train_ndim, train_nframe = train_iterator.get_next()

    val_iterator = utils.tfrecord.readTFrecord(args.val_path, model.feature_dimension, model.sequence_length, -1, args.val_batch,
                                     False)
    val_x, val_y, val_ndim, val_nframe = val_iterator.get_next()

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        global_step = sess.run(model.global_step)
        Saver = tf.train.Saver()

        # Tensorboard可视化
        train_summary_writer = tf.summary.FileWriter(args.train_tensorboard_path, graph=tf.get_default_graph())
        val_summary_writer = tf.summary.FileWriter(args.val_tensorboard_path, graph=tf.get_default_graph())

        try:
            tbar = tqdm(range(args.epoch * args.train_size // args.train_batch + 2))
            for _i in tbar:
                traindataBatch, trainlabelBatch, trainx, trainy = sess.run(
                    [train_x, train_y, train_nframe, train_ndim])
                feed_dict_train = {model.x_input: traindataBatch,
                                   model.y_true: trainlabelBatch,
                                   model.seqLen: trainx,
                                   model.dropout: args.dropout,
                                   model.training: True}
                _, global_step, lrate = sess.run([model.train_op, model.global_step, model.lr],
                                                 feed_dict=feed_dict_train)
                if global_step % 100 == 0:
                    y_pred_train, train_acc, train_loss, train_summary = sess.run(
                        [model.y_pred, model.accuracy, model.loss, model.merged_summary_op],
                        feed_dict=feed_dict_train)
                    train_summary_writer.add_summary(train_summary, global_step=global_step)
                    tbar.set_description("step: %d" % global_step)
                    tbar.set_postfix_str("lr: %.10f, acc: %s, loss: %s" % (lrate, train_acc, train_loss))

                # <editor-fold desc="Validate">
                # if global_step % 10 == 0 and global_step != 0:
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
            Saver.save(sess, save_path=model_save_path, global_step=global_step)