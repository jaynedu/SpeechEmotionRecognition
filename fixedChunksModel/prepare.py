# -*- coding: utf-8 -*-
# @Date    : 2020/9/3 15:36
# @Author  : Du Jing
# @FileName: prepare
# ---- Description ----

import os
import numpy as np
import utils

np.set_printoptions(threshold=np.inf)

# 读取数据库文件信息
jsonPath = r'E:\Datasets\emodb.json'  # 原始数据库json文件
savePath = r'E:\Datasets\emodb'
utils.base.check_dir(savePath)
selectedLabels = [0, 1, 2, 3]
dataset = utils.audio.Data(jsonPath, selectedLabels)
result = dataset.load_json(paths=True, labels=True, sex=True, duration=True)

# 确定chunk数量
maxDuration = max(result.duration)
windowLength = 1
nChunk = int(maxDuration/windowLength) + 5
steps = [(time-windowLength)/(nChunk-1) for time in result.duration]
nFramePerChunk = int(windowLength/0.020)
nFramePerStep = [int(step/0.020) for step in steps]
print("chunk number: %s" % nChunk)
print("chunk steps: %s" % steps)
print("nframe per step: %s" % nFramePerStep)
print("nframe per chunk: %d" % nFramePerChunk)
print("----------------------------------------------------------------------------------\n")

# 提取特征
result.features = []
output_dir = os.path.join(savePath, 'lld')
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
utils.tfrecord.generate_tfrecord(((x_train, y_train), (x_test, y_test), (x_val, y_val)), os.path.join(savePath, 'emodb_fixed_chunk'))
print("----------------------------------------------------------------------------------\n")