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
for i, path in enumerate(result.paths[:2]):
    tool = utils.audio.OpenSMILE(path)
    audio_name = os.path.split(path)[-1].split('.')[0]+'_lld.csv'
    output_path = os.path.join(tool.default_feature_file_save_path, audio_name)
    frame_feature = tool.get_IS13_lld(output_path)
    print("%d\t - output path: %s" % (i, output_path))
    print("\t - frame feature shape: %s" % (list(frame_feature.shape)))
    chunk_feature = utils.audio.AudioUtils.get_chunk_feature(frame_feature, nChunk, nFramePerChunk, nFramePerStep[i])
    print("\t - chunk feature shape: %s" % (list(chunk_feature.shape)))
    result.features.append(chunk_feature)
print("----------------------------------------------------------------------------------\n")

# 划分数据集

