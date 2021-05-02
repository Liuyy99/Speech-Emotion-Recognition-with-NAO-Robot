# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Get neutral mean and standard deviation of CASIA dataset"""
import os
import cPickle
import logging

import numpy as np

from support import read_file, extract_3d_mel, add_padding

filter_num = 40


def read_CASIA():
    neutral_num = 201
    rootdir = './CASIA_formatted'
    testdata1 = np.empty((neutral_num * 300, filter_num), dtype=np.float32)
    testdata2 = np.empty((neutral_num * 300, filter_num), dtype=np.float32)
    testdata3 = np.empty((neutral_num * 300, filter_num), dtype=np.float32)
    neutral_num = 0

    for emotion_name in os.listdir(rootdir):
        if emotion_name != "Neutral":  # ignore emotions other than "neutral"
            continue

        sub_dir = os.path.join(rootdir, emotion_name)
        for record in os.listdir(sub_dir):
            if record.startswith('.'):
                continue

            record_path = os.path.join(sub_dir, record)

            data, time, rate = read_file(record_path)
            mel_spec, delta1, delta2, time = extract_3d_mel(data, rate, filter_num)

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
    output = './zscore_casia_' + str(filter_num) + '.pkl'
    f = open(output, 'wb')
    cPickle.dump((mean1, std1, mean2, std2, mean3, std3), f)
    f.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    read_CASIA()
