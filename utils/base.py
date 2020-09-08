# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:54 下午
# @Author  : Du Jing
# @FileName: base
# ---- Description ----
#

import os

__all__ = [
    'ConfigDict',
    'clear',
    'check_dir'
]


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def clear(*args):
    for arg in args:
        try:
            os.remove(arg)
        except FileNotFoundError:
            print("文件 [%s] 不存在!" % arg)


def check_dir(path):
    if not os.path.exists(path):
        parent = os.path.split(path)[0]
        check_dir(parent)
        os.mkdir(path)


