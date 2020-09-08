# -*- coding: utf-8 -*-
# @Date    : 2020/9/8 15:09
# @Author  : Du Jing
# @FileName: urdu
# ---- Description ----

import os
import sys
import glob
import wave
import contextlib
import json

name = 'urdu'
classes = {'Angry': 0, 'Happy': 2, 'Neutral': 1, 'Sad': 3,}
path = r'E:\Corpus\URDU'
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
        'sex': "male" if filename.split('_')[0][1] == 'M' else "female"
    }

with contextlib.closing(open(targetFile, 'w')) as wf:
    obj = json.dumps(target, ensure_ascii=False)
    wf.write(obj)