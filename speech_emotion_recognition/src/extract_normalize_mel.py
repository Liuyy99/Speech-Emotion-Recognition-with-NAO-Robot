#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Usage: extract_normalize_mel.py

Description: This script extracts 3-d mel-spectrogram from speech signal. Each channel (static, delta, or delta-delta
) is normalized by the mean and standard deviation of that channel of neutral utterances in training set.
"""

import os
import cPickle
import logging

import numpy as np

from split_dataset import split_dataset
from support import read_file, extract_3d_mel, add_padding, generate_label

eps = 1e-5
filter_num = 40
# 1336 train utterance --> 2158 train 2s segments
angnum = 463  # 0
sadnum = 515  # 1
hapnum = 478  # 2
neunum = 702  # 3


def load_mean_std():
    """Load neutral means and standard deviations for z-score normalization"""
    f = open('./zscore_4_datasets_40.pkl', 'rb')
    dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3 = cPickle.load(f)
    return dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3


def add_to_list(data, label, num, emo_num, dataset_index, emotion, static, delta1, delta2):
    """Add data to training, validation or test data list"""
    dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3 = load_mean_std()
    data[num, :, :, 0] = (static - dataset_mean1[dataset_index])/(dataset_std1[dataset_index] + eps)
    data[num, :, :, 1] = (delta1 - dataset_mean2[dataset_index])/(dataset_std2[dataset_index] + eps)
    data[num, :, :, 2] = (delta2 - dataset_mean3[dataset_index])/(dataset_std3[dataset_index] + eps)

    em = generate_label(emotion)
    label[num] = em
    emo_num[emotion] = emo_num[emotion] + 1


def balance_training_data(train_num, train_label, train_data, pernum):
    """Randomly select equal number of data for each emotion"""
    hap_index = np.arange(hapnum)
    neu_index = np.arange(neunum)
    sad_index = np.arange(sadnum)
    ang_index = np.arange(angnum)

    a0 = 0
    s1 = 0
    h2 = 0
    n3 = 0

    for i in range(train_num):
        if train_label[i] == 0:
            ang_index[a0] = i
            a0 = a0 + 1
        elif train_label[i] == 1:
            sad_index[s1] = i
            s1 = s1 + 1
        elif train_label[i] == 2:
            hap_index[h2] = i
            h2 = h2 + 1
        elif train_label[i] == 3:
            neu_index[n3] = i
            n3 = n3 + 1

    np.random.shuffle(neu_index)
    np.random.shuffle(hap_index)
    np.random.shuffle(sad_index)
    np.random.shuffle(ang_index)

    hap_data = train_data[hap_index[0:pernum]].copy()
    hap_label = train_label[hap_index[0:pernum]].copy()
    ang_data = train_data[ang_index[0:pernum]].copy()
    ang_label = train_label[ang_index[0:pernum]].copy()
    sad_data = train_data[sad_index[0:pernum]].copy()
    sad_label = train_label[sad_index[0:pernum]].copy()
    neu_data = train_data[neu_index[0:pernum]].copy()
    neu_label = train_label[neu_index[0:pernum]].copy()
    train_num = 4 * pernum

    Train_label = np.empty((train_num, 1), dtype=np.int8)
    Train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)
    Train_data[0:pernum] = hap_data
    Train_label[0:pernum] = hap_label
    Train_data[pernum:2 * pernum] = sad_data
    Train_label[pernum:2 * pernum] = sad_label
    Train_data[2 * pernum:3 * pernum] = neu_data
    Train_label[2 * pernum:3 * pernum] = neu_label
    Train_data[3 * pernum:4 * pernum] = ang_data
    Train_label[3 * pernum:4 * pernum] = ang_label

    arr = np.arange(train_num)
    np.random.shuffle(arr)
    Train_data = Train_data[arr[0:]]
    Train_label = Train_label[arr[0:]]
    return Train_data, Train_label


def extract_features(record_path, pernums_set, u_num, set_num, set_data, set_label, set_emo, dataset_index, emotion):
    """Extract 3-d mel-spectrogram, format it to 300 (frames) * 40 (filter) segments."""
    data, time, rate = read_file(record_path)
    mel_spec, delta1, delta2, time = extract_3d_mel(data, rate, filter_num)

    if time <= 300:
        pernums_set[u_num] = 1
        u_num = u_num + 1

        static1, delta11, delta21 = add_padding(mel_spec, delta1, delta2)

        add_to_list(set_data, set_label, set_num, set_emo, dataset_index, emotion, static1, delta11,
                    delta21)
        set_num = set_num + 1

    else:
        pernums_set[u_num] = 2
        u_num = u_num + 1

        static1 = mel_spec[0:300, :]
        delta11 = delta1[0:300, :]
        delta21 = delta2[0:300, :]
        static2 = mel_spec[time - 300:time, :]
        delta12 = delta1[time - 300:time, :]
        delta22 = delta2[time - 300:time, :]

        add_to_list(set_data, set_label, set_num, set_emo, dataset_index, emotion, static1, delta11,
                    delta21)
        set_num = set_num + 1
        add_to_list(set_data, set_label, set_num, set_emo, dataset_index, emotion, static2, delta12,
                    delta22)
        set_num = set_num + 1

    return set_num, u_num


def read_combined_dataset():
    train_num = 2158  # the number of training 2s segments
    trnum = 1336  # the number of training utterances
    valid_num = 527  # the number of validating 2s segments
    vnum = 329  # the number of validating utterances
    test_num = 510  # the number of testing 2s segments
    tnum = 324  # the number of testing utterances
    pernums_train = np.arange(trnum)  # remember each training utterance contain how many segments
    pernums_valid = np.arange(vnum)  # remember each validating utterance contain how many segments
    pernums_test = np.arange(tnum)  # remember each testing utterance contain how many segments

    # EmoDB, Urdu, RAVDESS, SAVEE
    rootdir = '../combined_4_dataset'

    pernum = np.min([hapnum, angnum, sadnum, neunum])
    train_label = np.empty((train_num, 1), dtype=np.int8)
    test_label = np.empty((tnum, 1), dtype=np.int8)
    valid_label = np.empty((vnum, 1), dtype=np.int8)
    Test_label = np.empty((test_num, 1), dtype=np.int8)
    Valid_label = np.empty((valid_num, 1), dtype=np.int8)
    train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)
    valid_data = np.empty((valid_num, 300, filter_num, 3), dtype=np.float32)

    trnum = 0
    vnum = 0
    tnum = 0
    train_num = 0
    valid_num = 0
    test_num = 0
    train_emo = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    test_emo = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    valid_emo = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}

    dataset_train_nums = {"DG": 0, "DU": 0, "DR": 0, "DS": 0}
    dataset_valid_nums = {"DG": 0, "DU": 0, "DR": 0, "DS": 0}
    dataset_test_nums = {"DG": 0, "DU": 0, "DR": 0, "DS": 0}

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
            dataset_index = record[0:2]

            is_training_set, is_validation_set, is_test_set = split_dataset(record)

            if is_training_set:
                # training set
                dataset_train_nums[dataset_index] = dataset_train_nums[dataset_index] + 1
                train_num, trnum = extract_features(record_path, pernums_train, trnum, train_num, train_data,
                                                    train_label, train_emo, dataset_index, emotion)

            else:
                em = generate_label(emotion)
                if is_test_set:
                    # test_set
                    dataset_test_nums[dataset_index] = dataset_test_nums[dataset_index] + 1
                    test_label[tnum] = em
                    test_num, tnum = extract_features(record_path, pernums_test, tnum, test_num, test_data, Test_label,
                                                      test_emo, dataset_index, emotion)

                elif is_validation_set:
                    # valid_set
                    dataset_valid_nums[dataset_index] = dataset_valid_nums[dataset_index] + 1
                    em = generate_label(emotion)
                    valid_label[vnum] = em
                    valid_num, vnum = extract_features(record_path, pernums_valid, vnum, valid_num, valid_data,
                                                       Valid_label, valid_emo, dataset_index, emotion)

    print("train_num: ", train_num)
    print("valid_num: ", valid_num)
    print("test_num: ", test_num)
    print("vnum: ", vnum)
    print("tnum: ", tnum)
    print("dataset train nums: ", dataset_train_nums)
    print("dataset validate nums: ", dataset_valid_nums)
    print("dataset test nums: ", dataset_test_nums)

    Train_data, Train_label = balance_training_data(train_num, train_label, train_data, pernum)
    print train_label.shape
    print train_emo
    print test_emo
    print valid_emo
    output = './mel_4_datasets_40.pkl'
    f = open(output, 'wb')
    cPickle.dump((Train_data, Train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label,
                  pernums_test, pernums_valid), f)
    f.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    load_mean_std()
    read_combined_dataset()
