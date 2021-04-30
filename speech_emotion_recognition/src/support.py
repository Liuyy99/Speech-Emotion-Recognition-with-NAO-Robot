#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module provide a utility functions used cross the project.
"""

import wave

import numpy as np
import python_speech_features as ps


def read_file(filename):
    speech_file = wave.open(filename, 'r')
    params = speech_file.getparams()
    n_channels, samp_width, framerate, wav_length = params[:4]
    str_data = speech_file.readframes(wav_length)
    wave_data = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    speech_file.close()
    return wave_data, time, framerate


def generate_label(emotion):
    if emotion == 'ang':
        label = 0
    elif emotion == 'sad':
        label = 1
    elif emotion == 'hap':
        label = 2
    elif emotion == 'neu':
        label = 3
    else:
        label = 4

    return label


def extract_3d_mel(data, rate, filter_num):
    mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    time = mel_spec.shape[0]
    return mel_spec, delta1, delta2, time


def add_padding(mel_spec, delta1, delta2):
    part = np.pad(mel_spec, ((0, 300 - mel_spec.shape[0]), (0, 0)), 'constant',
                  constant_values=0)
    delta11 = np.pad(delta1, ((0, 300 - delta1.shape[0]), (0, 0)), 'constant',
                     constant_values=0)
    delta21 = np.pad(delta2, ((0, 300 - delta2.shape[0]), (0, 0)), 'constant',
                     constant_values=0)
    return part, delta11, delta21
