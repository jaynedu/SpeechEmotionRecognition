# -*- coding: utf-8 -*-
# @Date    : 2020/8/31 9:19 下午
# @Author  : Du Jing
# @FileName: vad
# ---- Description ----
#

import collections
import contextlib
import wave

import webrtcvad

__all__ = ['Vad']


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Vad:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)  # most strict vad
        self.frameLen = 30  # frame_duration_ms

    def read_audio(self, path):
        """
        :param path: 完整音频文件路径
        :return:
        """
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            nframes = wf.getnframes()
            pcm_data = wf.readframes(nframes)
            time = nframes / sample_rate
            return pcm_data, sample_rate, time

    def write_audio(self, path, frames, sr):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(frames)

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        i = 0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n
            i += 1

    def vad_collector(self, sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        start = []
        end = []
        for i, frame in enumerate(frames):
            if not triggered:
                ring_buffer.append(frame)
                num_voiced = len([f for f in ring_buffer if vad.is_speech(f.bytes, sample_rate)])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    start.append(ring_buffer[0].timestamp)
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append(frame)
                num_unvoiced = len([f for f in ring_buffer if not vad.is_speech(f.bytes, sample_rate)])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    end.append(frame.timestamp + frame.duration)
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames]), start[-1], end[-1]
                    ring_buffer.clear()
                    voiced_frames = []

    def get_audio_with_vad(self, input, output):
        audio, sr, time = self.read_audio(input)
        frames = self.frame_generator(self.frameLen, audio, sr)
        frames = list(frames)
        segments = self.vad_collector(sr, self.frameLen, 300, self.vad, frames)
        wav = bytes()
        for i, seg in enumerate(segments):
            voicedFrames, start, end = seg
            wav += voicedFrames
        if len(wav) > 0:
            self.write_audio(output, wav, sr)