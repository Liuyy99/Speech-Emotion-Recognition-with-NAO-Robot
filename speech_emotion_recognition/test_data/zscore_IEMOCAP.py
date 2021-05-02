# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Get neutral mean and standard deviation of IEMOCAP dataset Session 2 (the whole corpus is very large)"""
import os
import cPickle
import logging
import glob

import numpy as np

from support import read_file, extract_3d_mel, add_padding

filter_num = 40


def read_IEMOCAP():
    neutral_num = 364
    rootdir = './IEMOCAP_full_release'
    testdata1 = np.empty((neutral_num * 300, filter_num), dtype=np.float32)
    testdata2 = np.empty((neutral_num * 300, filter_num), dtype=np.float32)
    testdata3 = np.empty((neutral_num * 300, filter_num), dtype=np.float32)
    neutral_num = 0

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
                        wavname = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavname]
                        if emotion == 'neu':
                            data, time, rate = read_file(filename)
                            mel_spec, delta1, delta2, time = extract_3d_mel(data, rate, filter_num)

                            if speaker in ['Session2']:
                                if time <= 300:
                                    static1, delta11, delta21 = add_padding(mel_spec, delta1, delta2)
                                    testdata1[neutral_num * 300:(neutral_num + 1) * 300] = static1
                                    testdata2[neutral_num * 300:(neutral_num + 1) * 300] = delta11
                                    testdata3[neutral_num * 300:(neutral_num + 1) * 300] = delta21

                                    neutral_num = neutral_num + 1
                                else:
                                    for i in range(2):
                                        if i == 0:
                                            begin = 0
                                            end = begin + 300
                                        else:
                                            begin = time - 300
                                            end = time

                                        static1 = mel_spec[begin:end, :]
                                        delta11 = delta1[begin:end, :]
                                        delta21 = delta2[begin:end, :]
                                        testdata1[neutral_num * 300:(neutral_num + 1) * 300] = static1
                                        testdata2[neutral_num * 300:(neutral_num + 1) * 300] = delta11
                                        testdata3[neutral_num * 300:(neutral_num + 1) * 300] = delta21
                                        neutral_num = neutral_num + 1

    print "neutral_num", neutral_num
    mean1 = np.mean(testdata1, axis=0)
    std1 = np.std(testdata1, axis=0)
    mean2 = np.mean(testdata2, axis=0)
    std2 = np.std(testdata2, axis=0)
    mean3 = np.mean(testdata3, axis=0)
    std3 = np.std(testdata3, axis=0)
    output = './zscore_iemocap_' + str(filter_num) + '.pkl'
    f = open(output, 'wb')
    cPickle.dump((mean1, std1, mean2, std2, mean3, std3), f)
    f.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    read_IEMOCAP()
