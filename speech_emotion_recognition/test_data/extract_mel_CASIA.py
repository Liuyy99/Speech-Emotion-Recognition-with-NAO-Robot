#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Extract and normalize features from CASIA data"""

import os
import cPickle
import logging

import numpy as np

from support import read_file, extract_3d_mel, add_padding, generate_label

eps = 1e-5


def load_data():
    f = open('./zscore_casia_40.pkl', 'rb')
    mean1, std1, mean2, std2, mean3, std3 = cPickle.load(f)
    return mean1, std1, mean2, std2, mean3, std3


def read_CASIA():
    test_num = 865
    tnum = 800
    filter_num = 40
    pernums_test = np.arange(tnum)
    rootdir = './CASIA_formatted'

    mean1, std1, mean2, std2, mean3, std3 = load_data()

    test_label = np.empty((tnum, 1), dtype=np.int8)
    Test_label = np.empty((test_num, 1), dtype=np.int8)
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)

    tnum = 0
    test_num = 0
    test_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}

    all_emotions = ["Angry", "Happy", "Sad", "Neutral"]

    for emotion_name in os.listdir(rootdir):
        if emotion_name not in all_emotions:
            continue

        emotion = ''
        if emotion_name[0] == 'A':  # angry
            emotion = 'ang'
        if emotion_name[0] == 'H':  # happy
            emotion = 'hap'
        if emotion_name[0] == 'S':  # sad
            emotion = 'sad'
        if emotion_name[0] == 'N':  # neu
            emotion = 'neu'

        sub_dir = os.path.join(rootdir, emotion_name)
        for record in os.listdir(sub_dir):
            if record.startswith('.'):
                continue

            record_path = os.path.join(sub_dir, record)

            data, time, rate = read_file(record_path)
            mel_spec, delta1, delta2, time = extract_3d_mel(data, rate, filter_num)

            em = generate_label(emotion)
            test_label[tnum] = em
            if time <= 300:
                pernums_test[tnum] = 1
                static1, delta11, delta21 = add_padding(mel_spec, delta1, delta2)
                test_data[test_num, :, :, 0] = (static1 - mean1) / (std1 + eps)
                test_data[test_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                test_data[test_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)
                test_emt[emotion] = test_emt[emotion] + 1
                Test_label[test_num] = em
                test_num = test_num + 1
                tnum = tnum + 1
            else:
                pernums_test[tnum] = 2
                tnum = tnum + 1
                for i in range(2):
                    if (i == 0):
                        begin = 0
                        end = begin + 300
                    else:
                        begin = time - 300
                        end = time

                    static1 = mel_spec[begin:end, :]
                    delta11 = delta1[begin:end, :]
                    delta21 = delta2[begin:end, :]
                    test_data[test_num, :, :, 0] = (static1 - mean1) / (std1 + eps)
                    test_data[test_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                    test_data[test_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)
                    test_emt[emotion] = test_emt[emotion] + 1
                    Test_label[test_num] = em
                    test_num = test_num + 1

    print("tnum: ", tnum)
    print("test num: ", test_num)
    print test_emt
    output = './CASIA.pkl'
    f = open(output, 'wb')
    cPickle.dump((test_data, test_label, Test_label, pernums_test), f)
    f.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    read_CASIA()
