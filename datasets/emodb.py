# -*- coding: utf-8 -*-
# @Date    : 2020/9/8 15:07
# @Author  : Du Jing
# @FileName: emodb
# ---- Description ----

import os
import sys
import glob
import wave
import contextlib
import json

name = 'emodb'
classes = {'anger': 0, 'happiness': 2, 'sadness': 3, 'neutral': 1, 'boredom': 4, 'disgust': 5, 'fear': 6,}
path = r'E:\Corpus\EmoDB'
targetFile = os.path.join(r'E:\Datasets', name+'.json')
target = {
    'Name': name,
    'Categories': classes,
    'Path': path,
    'Items': {}
}

for file in glob.glob(os.path.join(path, '*.wav')):
    with contextlib.closing(wave.open(file, 'r')) as rf:
        fs, nframe = rf.getparams()[2:4]

    filename = os.path.basename(file)
    # 性别
    male = ['03', '10', '11', '12', '15']
    female = ['08', '09', '13', '14', '16']
    sex = filename[:2]
    # 情感类别
    emotionDict = {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }
    emotion = emotionDict[filename[5]]

    if sex in male:
        sextual='male'
    elif sex in female:
        sextual='female'
    else:
        sextual='unknown'

    label = classes[emotion]
    filename = os.path.basename(file).split('.')[0]
    target['Items'][filename] = {
        'path': file,
        'emotion': emotion,
        'label': label,
        'sampleRate': fs,
        'frames': nframe,
        'sex': sextual,
    }

with contextlib.closing(open(targetFile, 'w')) as wf:
    obj = json.dumps(target, ensure_ascii=False)
    wf.write(obj)