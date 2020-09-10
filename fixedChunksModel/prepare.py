# -*- coding: utf-8 -*-
# @Date    : 2020/9/3 15:36
# @Author  : Du Jing
# @FileName: prepare
# ---- Description ----

import os
import numpy as np
import utils
from fixedChunksModel import args

np.set_printoptions(threshold=np.inf)

# 读取数据库文件
dataset = utils.audio.Data(args.input_json_file, args.selected_labels)
result = dataset.load_json(paths=True, labels=True, sex=True, duration=True)

# 确定chunk数量
# n * step + (winlen - step) = total time => (n-1) * step + winlen = total time
maxDuration = max(result.duration)
nChunk = int(maxDuration / args.chunk_length) + 1
steps = [(time-args.chunk_length) / (nChunk-1) for time in result.duration]
nFramePerChunk = int((args.chunk_length-args.window_length) / args.window_step + 1) + 1
nFramePerStep = [int((step - args.window_length) / args.window_step + 1) + 1 for step in steps]
print("chunk number: %s" % nChunk)
print("chunk steps: %s" % steps)
print("nframe per step: %s" % nFramePerStep)
print("nframe per chunk: %d" % nFramePerChunk)
print("----------------------------------------------------------------------------------\n")

# 提取特征
result.features = []
output_dir = os.path.join(args.output_feature_dir, 'lld')
utils.base.check_dir(output_dir)
for i, path in enumerate(result.paths):
    tool = utils.audio.OpenSMILE(path)
    audio_name = os.path.split(path)[-1].split('.')[0]
    output_path = os.path.join(output_dir, audio_name + '_lld.csv')
    if os.path.exists(output_path):
        frame_feature = tool.read_lld_feature(output_path)
    else:
        frame_feature = tool.get_IS13_lld(output_path)
    print("%d\t - output path: %s" % (i, output_path))
    print("\t - label: %s" % (result.labels[i]))
    print("\t - frame feature shape: %s" % (list(frame_feature.shape)))
    chunk_feature = utils.audio.AudioUtils.get_chunk_feature(frame_feature, nChunk, nFramePerChunk, nFramePerStep[i])
    print("\t - chunk feature shape: %s" % (list(chunk_feature.shape)))
    result.features.append(chunk_feature)
print("total feature length: %s" % len(result.features))
print("----------------------------------------------------------------------------------\n")

# 划分数据集
(x_train, y_train), (x_test, y_test), (x_val, y_val) = utils.audio.AudioUtils.split_features_labels(result.features, result.labels)
print("train_length: %s" % len(x_train))
print("test_length: %s" % len(x_test))
print("val_length: %s" % len(x_val))
utils.tfrecord.generate_tfrecord(((x_train, y_train), (x_test, y_test), (x_val, y_val)), os.path.join(args.output_feature_dir, 'emodb_fixed_chunk'))
print("----------------------------------------------------------------------------------\n")