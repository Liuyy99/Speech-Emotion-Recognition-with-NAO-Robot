#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Usage: zscore_mean_std.py

Description: This script calculates the mean and standard deviation of "neutral" speech's 3-d mel-spectrogram (in
vector representation) for z-score normalization. The means and standard deviations of all training corpora (4 in
total) are calculated and stored.
"""

import os
import cPickle
import logging

import numpy as np

from split_dataset import split_dataset
from support import read_file, extract_3d_mel, add_padding

filter_num = 40


def add_to_list(train_nums, dataset_index, combined_train_data, static, delta1, delta2):
    train_num = train_nums[dataset_index]
    combined_train_data[dataset_index]["mel"][train_num * 300:(train_num + 1) * 300] = static
    combined_train_data[dataset_index]["d_mel"][train_num * 300:(train_num + 1) * 300] = delta1
    combined_train_data[dataset_index]["dd_mel"][train_num * 300:(train_num + 1) * 300] = delta2
    train_nums[dataset_index] = train_nums[dataset_index] + 1


def get_mean_std(combined_train_data, dataset_indices):
    dataset_mean1 = {}
    dataset_std1 = {}
    dataset_mean2 = {}
    dataset_std2 = {}
    dataset_mean3 = {}
    dataset_std3 = {}

    for index in dataset_indices:
        mean1 = np.mean(combined_train_data[index]["mel"], axis=0)
        std1 = np.std(combined_train_data[index]["mel"], axis=0)
        mean2 = np.mean(combined_train_data[index]["d_mel"], axis=0)
        std2 = np.std(combined_train_data[index]["d_mel"], axis=0)
        mean3 = np.mean(combined_train_data[index]["dd_mel"], axis=0)
        std3 = np.std(combined_train_data[index]["dd_mel"], axis=0)

        dataset_mean1[index] = mean1
        dataset_std1[index] = std1
        dataset_mean2[index] = mean2
        dataset_std2[index] = std2
        dataset_mean3[index] = mean3
        dataset_std3[index] = std3

    output = './zscore_4_datasets_' + str(filter_num) + '.pkl'
    f = open(output, 'wb')
    cPickle.dump((dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3), f)
    f.close()


def extract_features(record_path, record, train_nums, dataset_index, combined_train_data):
    data, time, rate = read_file(record_path)
    mel_spec, delta1, delta2, time = extract_3d_mel(data, rate, filter_num)

    is_training_set, is_validation_set, is_test_set = split_dataset(record)

    if is_training_set:  # neutral data in the training set
        if time <= 300:  # the pre-set length of 3d-mel is 300
            static1, delta11, delta21 = add_padding(mel_spec, delta1, delta2)

            add_to_list(train_nums, dataset_index, combined_train_data, static1, delta11, delta21)
        else:
            static1 = mel_spec[0:300, :]
            delta11 = delta1[0:300, :]
            delta21 = delta2[0:300, :]
            static2 = mel_spec[time - 300:time, :]
            delta12 = delta1[time - 300:time, :]
            delta22 = delta2[time - 300:time, :]

            add_to_list(train_nums, dataset_index, combined_train_data, static1, delta11, delta21)
            add_to_list(train_nums, dataset_index, combined_train_data, static2, delta12, delta22)


def read_combined_dataset():
    dataset_indices = ["DG", "DU", "DR", "DS"]

    # number of neutral 3s segments
    neutral_train_num_DG = 138  # EmoDB
    neutral_train_num_DU = 72  # Urdu
    neutral_train_num_DR = 382  # RAVDESS
    neutral_train_num_DS = 110  # SAVEE

    # number of neutral utterances (1 recording is 1 utterance)
    # utterance_nums = {"DG": 104, "DU": 72, "DR": 192, "DS": 60}

    train_nums = {
        "DG": neutral_train_num_DG,
        "DU": neutral_train_num_DU,
        "DR": neutral_train_num_DR,
        "DS": neutral_train_num_DS
    }

    # EmoDB, Urdu, RAVDESS, SAVEE
    rootdir = '../combined_4_dataset'

    combined_train_data = {}

    for index in dataset_indices:
        traindata1 = np.empty((train_nums[index] * 300, filter_num), dtype=np.float32)
        traindata2 = np.empty((train_nums[index] * 300, filter_num), dtype=np.float32)
        traindata3 = np.empty((train_nums[index] * 300, filter_num), dtype=np.float32)

        train_data_3d = {"mel": traindata1, "d_mel": traindata2, "dd_mel": traindata3}

        combined_train_data[index] = train_data_3d
        train_nums[index] = 0

    for emotion_name in os.listdir(rootdir):
        if emotion_name != "Neutral":  # ignore emotions other than "neutral"
            continue

        # calculate the mean and std of 3-d mel-spectrogram of neutral speech in each corpus
        sub_dir = os.path.join(rootdir, emotion_name)
        for record in os.listdir(sub_dir):
            if record.startswith('.'):  # ignore non-speech files
                continue

            record_path = os.path.join(sub_dir, record)
            dataset_index = record[0:2]
            extract_features(record_path, record, train_nums, dataset_index, combined_train_data)

    print "number of neutral segments in training set:", train_nums

    get_mean_std(combined_train_data, dataset_indices)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    read_combined_dataset()
