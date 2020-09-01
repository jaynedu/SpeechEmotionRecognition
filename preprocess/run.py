# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 10:01 下午
# @Author  : Du Jing
# @FileName: run
# ---- Description ----
#

from .data import Data
from .extraction import AudioFeatureExtraction

# parameters
jsonPath = 'dataset/emodb.json'  # 原始数据库json文件
savePath = r'E:\Datasets\emodb_768'
selectedLabels = [0, 1, 2, 3]

dataset = Data(jsonPath)
extractor = AudioFeatureExtraction()
extractor.frame_length = 512
extractor.frame_step = 256
feature_func = extractor.stFeatures  # 特征函数
seqLength = 768
use_vad = False

print(' 开始 '.center(30, '='))

print("加载数据库JSON文件...")
paths, labels = dataset.load_json(selectedLabels)

print("正在提取特征...")
paths, labels, features = extractor.extract(paths, labels, feature_func, seqLength, use_vad)

print("正在划分数据集...")
splits = extractor.split(features, labels)

print("正在生成数据集...")
extractor.tfrecord_genrator(splits, savePath)

print(' 结束 '.center(30, '='))
