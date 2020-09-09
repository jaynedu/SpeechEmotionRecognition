# -*- coding: utf-8 -*-
# @Date    : 2020/9/9 11:15
# @Author  : Du Jing
# @FileName: args
# ---- Description ----

import utils
import os
import time

datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

# General settings
model_name = os.path.split(os.getcwd())[-1]
dataset = 'emodb'
feature_name = "IS13_ComParE_LLD"

# Preprocess settings
input_json_file = os.path.join(r'E:\Datasets', dataset+'.json')
output_feature_dir = os.path.join(r'E:\Datasets', dataset)
selected_labels = [0, 1, 2, 3]
chunk_length = 1

# Model version settings
model_type = "NonAtten"
model_version = '$'.join([dataset, feature_name, model_type])
model_save_dir = os.path.join(r'E:\Models', model_name, model_version)

# Dataset settings
train_tensorboard_path = os.path.join('logs', datetime + '_' + model_version, 'train')
val_tensorboard_path = os.path.join('logs', datetime + '_' + model_version, 'val')
train_path = r'E:\Datasets\iemocap_512_3155.train'
val_path = r'E:\Datasets\iemocap_512_395.val'
test_path = r'E:\Datasets\iemocap_512_394.test'
train_size = eval(os.path.splitext(train_path)[0].split('_')[-1])
val_size = eval(os.path.splitext(val_path)[0].split('_')[-1])
test_size = eval(os.path.splitext(test_path)[0].split('_')[-1])

# Parameter Settings
n_label = 4
raw_feature_shape = [13, 50, 130]
dropout = 0.1
train_batch = 16
val_batch = val_size if val_size <= 100 else 100
validate = False
epoch = 100
eta = 0.001
warmup = 2000.
# classes = ['anger', 'happy', 'neutral', 'sad']  # urdu / casia / des
classes = ['anger', 'sad', 'happy', 'neutral']  # iemocap
# classes = ['anger', 'happy', 'sad', 'neutral']  # emodb
# classes = ["agressiv", "neutral", "cheerful", "tired"]  # abc




