# -*- coding: utf-8 -*-
# @Date    : 2020/9/10 18:21
# @Author  : Du Jing
# @FileName: common
# ---- Description ----


class ModelBase:
    def __init__(self):
        pass

    def parse_tfrecord(self, example_series):
        pass

    def read_tfrecord(self, file, epoch=None, batch_size=None, isTrain=True):
        pass

    def setup(self, *args, **kwargs):
        pass
