# -*- coding: utf-8 -*-
# @Date    : 2020/9/8 15:03
# @Author  : Du Jing
# @FileName: iemocap
# ---- Description ----

import os
import sys
import glob
import wave
import contextlib
import json
import tqdm

name = 'iemocap'
classes = {
        'anger': 0,
        'sad': 3,
        'fru': 9,  # sad
        'happy': 2,
        'neutral': 1,
        'dis': 6,
        'exc': 5,  # sur
        'fea': 4,
        'oth': 7,  # unknown
        'sur': 5,
        'xxx': 8,  # unknown
    }
path = r'E:\Corpus\IEMOCAP'
targetFile = os.path.join(r'E:\Datasets', name+'.json')
target = {
    'Name': name,
    'Categories': classes,
    'Path': path,
    'Items': {}
}

for file in tqdm.tqdm(glob.glob(os.path.join(path, 'discrete', '*', '*.wav'))):
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
        'sex': "male" if filename.split('_')[0].endswith('M') else "female"
    }

with contextlib.closing(open(targetFile, 'w')) as wf:
    obj = json.dumps(target, ensure_ascii=False)
    wf.write(obj)