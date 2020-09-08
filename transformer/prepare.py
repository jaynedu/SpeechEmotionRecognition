# -*- coding: utf-8 -*-
# @Date    : 2020/9/3 15:11
# @Author  : Du Jing
# @FileName: prepare
# ---- Description ----

import utils

# parameters
jsonPath = r'E:\Datasets\emodb.json'  # 原始数据库json文件
savePath = r'E:\Datasets\emodb_768'
selectedLabels = [0, 1, 2, 3]
seqLength = 300

dataset = utils.audio.Data(jsonPath, selectedLabels)
print(dataset.load_json())
# extractor = utils.audio.AudioFeatureExtractor()
#
# print(' 开始 '.center(30, '='))
#
# print("加载数据库JSON文件...")
# paths, labels = dataset.load_json(selectedLabels)
#
# print("正在提取特征...")
# paths, labels, features = extractor.extract_from_lib(paths, labels, "logfbank", seqLength)
#
# print("正在划分数据集...")
# splits = utils.audio.AudioUtils.split_features_labels(features, labels)
#
# print("正在生成数据集...")
# utils.tfrecord.generate_tfrecord(splits, savePath)
#
# print(' 结束 '.center(30, '='))
