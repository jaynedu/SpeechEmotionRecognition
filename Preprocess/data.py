# -*- coding: utf-8 -*-
# @Date    : 2020/8/30 4:27 下午
# @Author  : Du Jing
# @FileName: data
# ---- Description ----
#

import os
import contextlib
import json

class Data(object):
    def __init__(self, jsonPath):
        self.jsonPath = jsonPath

    def load_json(self, selectedLabels):
        paths = []
        labels = []
        with contextlib.closing(open(self.jsonPath, 'r')) as rf:
            content = json.load(rf)['list']
            for item in content.values():
                if isinstance(selectedLabels, list):
                    if item['label'] in selectedLabels:
                        paths.append(item['path'])
                        labels.append(item['label'])
                elif selectedLabels is None:
                    paths.append(item['path'])
                    labels.append(item['label'])
        return paths, labels
