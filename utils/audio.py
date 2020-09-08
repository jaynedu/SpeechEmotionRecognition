# -*- coding: utf-8 -*-
# @Date    : 2020/9/3 15:13
# @Author  : Du Jing
# @FileName: audio
# ---- Description ----

import os
import sys
import contextlib
import json
import tensorflow as tf
import numpy as np
import subprocess
import random
import wave
import csv
import librosa
import python_speech_features
import tqdm
from pyAudioAnalysis import ShortTermFeatures
from sklearn.model_selection import train_test_split
import utils


__all__ = [
    'Data',
    'AudioUtils',
    'AudioFeatureExtractor',
    'OpenSMILE'
]


class Data:
    def __init__(self, jsonPath, selectedLabels):
        self.jsonPath = jsonPath
        self.selectedLabels = selectedLabels

        if selectedLabels is None:
            pass
        elif isinstance(selectedLabels, list):
            if len(selectedLabels) == 0:
                self.selectedLabels = None
            else:
                pass
        else:
            print("Please check out SELECTEDLABELS! - Available values: None, [], or [int, ...]. ")
            sys.exit(1)

    def load_json(self,
                  labels=True,
                  paths=True,
                  sex=False,
                  emotions=False,
                  duration=False,
                  nFrames=False,
                  sampleRate=False):
        Labels = []
        Paths = []
        Sexs = []
        Emotions = []
        NFrames = []
        Duration = []
        with contextlib.closing(open(self.jsonPath, 'r')) as rf:
            content = json.load(rf)['Items']

            for value in content.values():
                if isinstance(self.selectedLabels, list):
                    if value['label'] in self.selectedLabels:
                        Paths.append(value['path'])
                        Labels.append(value['label'])
                        if emotions:
                            Emotions.append(value['emotion'])
                        if sex:
                            Sexs.append(value['sex'])
                        if nFrames:
                            NFrames.append(value['frames'])
                        if sampleRate:
                            SampleRate = value['sampleRate']
                        if duration:
                            Duration.append(float(value['frames']/value['sampleRate']))

                elif self.selectedLabels is None:
                    Paths.append(value['path'])
                    Labels.append(value['label'])
                    if emotions:
                        Emotions.append(value['emotion'])
                    if sex:
                        Sexs.append(value['sex'])
                    if nFrames:
                        NFrames.append(value['frames'])
                    if sampleRate:
                        SampleRate = value['sampleRate']
                    if duration:
                        Duration.append(float(value['frames'] / value['sampleRate']))

        res = utils.base.ConfigDict()
        if paths:
            res.paths = Paths
        if labels:
            res.labels = Labels
        if emotions:
            res.emotions = Emotions
        if sex:
            res.sex = Sexs
        if duration:
            res.duration = Duration
        if nFrames:
            res.nFrames = NFrames
        if sampleRate:
            res.sampleRate = SampleRate
        res.size = len(Paths)
        return res


class AudioUtils:
    '''
    音频处理工具
    '''

    @staticmethod
    def read(audio):
        y, sr = librosa.load(audio, sr=16000)
        signal = python_speech_features.sigproc.preemphasis(y, 0.97)
        return signal, sr

    @staticmethod
    def get_chunk_feature(frame_feature, n_chunk, nframe_per_chunk, nframe_per_step):
        frame_length, feature_dimension = frame_feature.shape
        features = np.zeros((n_chunk, nframe_per_chunk, feature_dimension))
        for i in range(n_chunk):
            start = i * nframe_per_step
            end = i * nframe_per_step + nframe_per_chunk
            chunk_feature = np.zeros((nframe_per_chunk, feature_dimension))
            for j, frame in enumerate(frame_feature[start: end]):
                chunk_feature[j, :] = frame
            features[i, :, :] = chunk_feature
        return features

    @staticmethod
    def split_features_labels(xs, ys: list):
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
    def generate_testset(self, tfrecordPath, nFeature, seqLength, totalSize):
        test_iterator = utils.tfrecord.read_tfrecord(tfrecordPath, nFeature, seqLength, 1, totalSize, False)
        x, y, ndim, nframe = test_iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(test_iterator.initializer)
            inputs, labels, dim, seqlen = sess.run([x, y, ndim, nframe])
            save_path = os.path.splitext(tfrecordPath)[0] + '.npz'
            np.savez(save_path, inputs=inputs, labels=labels, dim=dim, seqlen=seqlen)

        return save_path


class OpenSMILE:
    default_feature_file_save_path = os.getcwd()  # 默认特征集文件保存路径
    def __init__(self, input_file):
        self.openSmile_exe = 'D:\\openSMILE\\opensmile-2.3.0\\bin\\Win32\\SMILExtract_Release.exe'
        self.input_file = input_file

        # 2009-InterSpeech Emotion Challenge特征集，共384个特征
        self.IS09 = 'D:\\openSMILE\\opensmile-2.3.0\\config\\IS09_emotion.conf'
        self.IS13 = 'D:\\openSMILE\\opensmile-2.3.0\\config\\IS13_ComParE.conf'
        # 2016-ComParE特征集，共6373个特征
        self.ComParE_2016 = 'D:\\openSMILE\\opensmile-2.3.0\\config\\ComParE_2016.conf'
        # 2016-eGeMAPS特征集，共88个特征
        self.eGeMAPSv01a = 'D:\\openSMILE\\opensmile-2.3.0\\config\\gemaps\\eGeMAPSv01a.conf'

    def extract_hld_features(self, config_file, output_file):
        cmd = self.openSmile_exe + " -noconsoleoutput -C " + config_file + " -I " + self.input_file + " -O " + output_file + " -instname " + os.path.split(self.input_file)[1].split('.')[0]
        res = subprocess.call(cmd)

    def extract_lld_features(self, config_file, output_file):
        cmd = self.openSmile_exe + " -noconsoleoutput -C " + config_file + " -I " + self.input_file + " -lldarffoutput " + output_file + " -instname " + os.path.split(self.input_file)[1].split('.')[0]
        res = subprocess.call(cmd)

    @staticmethod
    def read_hld_feature(feature_file):
        with open(feature_file, 'r') as file:
            last_line = file.readlines()[-1]
        features = np.array(last_line.split(',')[1: -1], dtype="float64")
        return features
    
    @staticmethod
    def read_lld_feature(feature_file):
        with open(feature_file, 'r') as file:
            features = []
            while file.readline():
                line = file.readline()
                if line.startswith('\''):
                    features.append(line.split(',')[2: -1])
        return np.array(features, dtype="float64")

    def get_ComParE_hsv(self, output_file):
        self.extract_hld_features(self.ComParE_2016, output_file)
        features = self.read_hld_feature(output_file)
        return features
    
    def get_ComParE_lld(self, output_file):
        self.extract_lld_features(self.ComParE_2016, output_file)
        features = self.read_lld_feature(output_file)
        return features

    def get_IS09_hsv(self, output_file):
        self.extract_hld_features(self.IS09, output_file)
        features = self.read_hld_feature(output_file)
        return features
    
    def get_IS09_lld(self, output_file):
        self.extract_lld_features(self.IS09, output_file)
        features = self.read_lld_feature(output_file)
        return features

    def get_IS13_hsv(self, output_file):
        self.extract_hld_features(self.IS13, output_file)
        features = self.read_hld_feature(output_file)
        return features
    
    def get_IS13_lld(self, output_file):
        self.extract_lld_features(self.IS13, output_file)
        features = self.read_lld_feature(output_file)
        return features

    def get_eGeMAPS_hsv(self, output_file):
        self.extract_hld_features(self.eGeMAPSv01a, output_file)
        features = self.read_hld_feature(output_file)
        return features
    
    def get_eGeMAPS_lld(self, output_file):
        self.extract_lld_features(self.eGeMAPSv01a, output_file)
        features = self.read_lld_feature(output_file)
        return features


class AudioFeatureExtractor:
    '''
    音频特征提取
    '''

    def __init__(self):
        self.frame_length = 400
        self.frame_step = 160
        self._openSMILE_EXE = r"D:\openSMILE\opensmile-2.3.0\bin\Win32\SMILExtract_Release.exe"

    def _stFeatures(self, signal, sr):
        features, feature_names = ShortTermFeatures.feature_extraction(signal, sr, self.frame_length, self.frame_step)
        features = np.transpose(features)
        return features  # [nFrame (variable), nFeature (fixed)]

    def _logFBank(self, signal, sr):
        window_length = float(self.frame_length / sr)
        window_step = float(self.frame_step / sr)
        features = python_speech_features.logfbank(signal, sr, window_length, window_step, nfilt=64)
        return features

    def extract_from_lib(self, paths, labels, featureName=None, seqLength=None):
        if featureName is None:
            print("** Feature Available: ['stFeatures', 'logfbank'] **")
            f_name = input("Input Feature Name: ")
            while f_name not in ['stFeatures', 'logfbank']:
                print("** Available: ['stFeatures', 'logfbank'] **")
                f_name = input("Input Feature Name: ")
        else:
            f_name = featureName

        result_labels = []
        result_features = []
        tqdmTool = tqdm.tqdm(zip(paths, labels))
        for path, label in tqdmTool:
            tqdmTool.set_description("Current File: %s" % path)
            try:
                # read audio
                signal, sr = AudioUtils.read(path)

                # extract features
                if f_name == "stFeatures":
                    feature = self._stFeatures(signal, sr)
                if f_name == "logfbank":
                    feature = self._logFBank(signal, sr)
                tqdmTool.set_postfix_str(" Current Label: %d, Shape: %s. " % (label, feature.shape))

                # split frames or padding zeros
                if seqLength is not None:
                    current_length = feature.shape[0]
                    if current_length <= seqLength:
                        feature = np.pad(feature, ((0, seqLength - current_length), (0, 0)), 'constant',
                                         constant_values=0)
                    else:
                        times = (current_length - seqLength) // 100 + 1  # step for 100 frames
                        for i in range(times):
                            start_frame = 100 * i
                            stop_frame = start_frame + seqLength
                            feature = feature[start_frame: stop_frame]
                            result_features.append(feature)
                            result_labels.append(label)
                        continue
                result_features.append(feature)
                result_labels.append(label)

            except Exception as error:
                print(error)

        return result_features, result_labels

    def extract_from_openSMILE(self, paths, labels, configFile, outputDir):
        result_labels = []
        result_features = []
        for i, path, label in zip(range(len(paths)), paths, labels):
            # extract features
            audio_name = os.path.split(path)[-1].split('.')[0]
            config_name = os.path.split(configFile)[-1].split('.')[0].replace('_', '-')
            output = os.path.join(outputDir, audio_name + "_" + config_name + '.csv')
            cmd = self._openSMILE_EXE + " -C " + configFile + " -I " + path + " -O " + output
            subprocess.call(cmd)

            # read features
            with open(output, 'r') as file:
                content = csv.reader(file)
                for row in content:
                    compare_features = row
            compare_features_floats = [float(item) for item in compare_features[1:-1]]
            return compare_features_floats

        return result_features, result_labels
