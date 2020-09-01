# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 9:22 下午
# @Author  : Du Jing
# @FileName: extraction
# ---- Description ----
#

import os
import random
import librosa
import numpy as np
import python_speech_features
import tqdm
from pyAudioAnalysis import ShortTermFeatures
from sklearn.model_selection import train_test_split
import utils


class AudioFeatureExtraction:
    def __init__(self):
        self.frame_length = 400
        self.frame_step = 160

    @staticmethod
    def read_audio(input, use_vad=True):
        output = input
        if use_vad:
            output = os.path.join(r'E:\temp', os.path.basename(input))
            utils.Vad().get_audio_with_vad(input, output)

        y, sr = librosa.load(output, sr=16000)

        if use_vad:
            utils.base.clear(output)  # 清除临时文件

        signal = python_speech_features.sigproc.preemphasis(y, 0.97)
        return signal, sr

    def stFeatures(self, signal, sr):
        features, feature_names = ShortTermFeatures.feature_extraction(signal, sr, self.frame_length, self.frame_step)
        features = np.transpose(features)
        return features  # [nFrame (variable), nFeature (fixed)]

    def logfbank(self, signal, sr):
        winlen = float(self.frame_length / sr)
        winstep = float(self.frame_step / sr)
        features = python_speech_features.logfbank(signal, sr, winlen, winstep, nfilt=64)
        return features

    def extract(self, pathList, labelList, feature_func, seqLength=None, use_vad=True):
        paths = []
        labels = []
        features = []
        tbar = tqdm.tqdm(zip(pathList, labelList))
        for path, label in tbar:
            tbar.set_description("FILE: %s" % path)
            try:
                signal, sr = self.read_audio(path, use_vad)
                feature = feature_func(signal, sr)
                tbar.set_postfix_str("label: %d, shape: %s" % (label, feature.shape))

                # 如果指定了seq_length，需要对特征进行分割或补零
                if seqLength is not None:
                    length = feature.shape[0]
                    if length <= seqLength:
                        feature = np.pad(feature, ((0, seqLength - length), (0, 0)), 'constant', constant_values=0)
                    else:
                        times = (length - seqLength) // 100 + 1
                        for i in range(times):
                            begin = 100 * i
                            end = begin + seqLength
                            feature = feature[begin: end]

                            features.append(feature)
                            labels.append(label)
                            paths.append(path)

                        # 跳出当前循环
                        continue

                features.append(feature)
                labels.append(label)
                paths.append(path)
            except Exception as error:
                if use_vad:
                    print('[WARNING] - pop: [%s] [%d]' % (path, label))
                print(error)
        return paths, labels, features

    @staticmethod
    def split(xs, ys: list):
        count = {}
        for i in set(ys):
            count[i] = ys.count(i)
        maxCount = min(count.values())
        print('数据集中单个情感数量: %d' % maxCount)
        labelCount = {}
        labelCount = labelCount.fromkeys(count.keys(), 0)
        collect = list(zip(xs, ys))
        random.shuffle(collect)
        xs[:], ys[:] = zip(*collect)
        _xs, _ys = [], []
        for i in range(len(xs)):
            if labelCount[ys[i]] < maxCount:
                _xs.append(xs[i])
                _ys.append(ys[i])
                labelCount[ys[i]] += 1
        x_train, x_test, y_train, y_test = train_test_split(_xs, _ys, test_size=0.2, stratify=_ys)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)
        return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    @staticmethod
    def tfrecord_genrator(splits, path):
        '''
        :param splits: [[feature, label], [feature, label], [feature, label], ...]
        :param path: save_path
        :return:
        '''
        suffix = ['.train', '.test', '.val']
        for i, split in enumerate(splits):
            x, y = split
            writer = utils.tfrecord.createWriter(path + '_' + str(len(x)) + suffix[i])
            for feature, label in zip(x, y):
                utils.tfrecord.saveTFrecord(feature, label, writer)
            utils.tfrecord.disposeWriter(writer)
