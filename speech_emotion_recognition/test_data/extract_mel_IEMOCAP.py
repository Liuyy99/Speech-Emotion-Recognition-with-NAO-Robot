#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Extract and normalize features from IEMOCAP dataset Session 2 (the whole corpus is very large)"""

import os
import cPickle
import logging
import glob

import numpy as np

from support import read_file, extract_3d_mel, add_padding, generate_label

eps = 1e-5


def load_data():
    f = open('./zscore_iemocap_40.pkl', 'rb')
    mean1, std1, mean2, std2, mean3, std3 = cPickle.load(f)
    return mean1, std1, mean2, std2, mean3, std3


def read_IEMOCAP():
    tnum = 393
    test_num = 652
    filter_num = 40
    pernums_test = np.arange(tnum)
    rootdir = './IEMOCAP_full_release'

    mean1, std1, mean2, std2, mean3, std3 = load_data()

    test_label = np.empty((tnum, 1), dtype=np.int8)
    Test_label = np.empty((test_num, 1), dtype=np.int8)
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)

    tnum = 0
    test_num = 0
    test_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}

    for speaker in os.listdir(rootdir):
        if speaker[0] == 'S':
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                if sess[7] == 'i':
                    emotdir = emoevl + '/' + sess + '.txt'
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if line[0] == '[':
                                t = line.split()
                                emot_map[t[3]] = t[4]

                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        # wavname = filename[-23:-4]
                        wavname = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavname]
                        if emotion in ['hap', 'ang', 'neu', 'sad']:
                            data, time, rate = read_file(filename)
                            mel_spec, delta1, delta2, time = extract_3d_mel(data, rate, filter_num)

                            if speaker in ['Session2']:
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
                                        if i == 0:
                                            begin = 0
                                            end = begin + 300
                                        else:
                                            end = time
                                            begin = time - 300
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
    output = './IEMOCAP.pkl'
    f = open(output, 'wb')
    cPickle.dump((test_data, test_label, Test_label, pernums_test), f)
    f.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    read_IEMOCAP()
