# -*- coding: utf-8 -*-
# @Date    : 2020/9/9 14:47
# @Author  : Du Jing
# @FileName: train
# ---- Description ----

import os
import tensorflow as tf
import numpy as np


# 环境设置
np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.device('/gpu:0')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需加内存