# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:54 下午
# @Author  : Du Jing
# @FileName: base
# ---- Description ----
#

import os

__all__ = [
    'clear',
    'check_dir'
]


def clear(*args):
    for arg in args:
        try:
            os.remove(arg)
        except FileNotFoundError:
            print("文件 [%s] 不存在!" % arg)


def check_dir(dir):
    if not os.path.exists(dir):
        parent = os.path.split(dir)[0]
        check_dir(parent)
        os.mkdir(dir)


