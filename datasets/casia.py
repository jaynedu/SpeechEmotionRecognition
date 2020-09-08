# -*- coding: utf-8 -*-
# @Date    : 2020/9/8 14:53
# @Author  : Du Jing
# @FileName: casia
# ---- Description ----

import os
import sys
import glob
import wave
import contextlib
import json

name = 'casia'
classes = {'anger': 0, 'happy': 2, 'neutral': 1, 'sad': 3, 'fear': 4, 'surprise': 5}
path = r'E:\Corpus\CASIA'
targetFile = os.path.join(r'E:\Datasets', name+'.json')
target = {
    'Name': name,
    'Categories': classes,
    'Path': path,
    'Items': {}
}

for file in glob.glob(os.path.join(path, '*', '*.wav')):
    with contextlib.closing(wave.open(file, 'r')) as rf:
        fs, nframe = rf.getparams()[2:4]
    emotion = file.split('/')[-2] if sys.platform == 'darwin' else file.split('\\')[-2]
    label = classes[emotion]
    filename = os.path.basename(file).split('.')[0]
    target['Items'][filename] = {
        'path': file,
        'emotion': emotion,
        'label': label,
        'sampleRate': fs,
        'frames': nframe,
        'sex': "male" if filename.split('_')[1] in ["2", "4"] else "female"
    }

with contextlib.closing(open(targetFile, 'w')) as wf:
    obj = json.dumps(target, ensure_ascii=False)
    wf.write(obj)
