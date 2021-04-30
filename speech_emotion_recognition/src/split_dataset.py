#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module provide function splitting the 4-corpora combined dataset into training, validation and
testing set. The validation and testing set are speaker-independent (having no common speakers
with training set).
"""


def split_dataset(record):
    is_training_set = False
    is_validation_set = False
    is_test_set = False

    dataset_index = record[0:2]
    if dataset_index == 'DG':  # Emo-DB
        speaker_number = int(record[3]) * 10 + int(record[4])
        if speaker_number == 3 or speaker_number == 10:
            is_validation_set = True
        elif speaker_number == 12 or speaker_number == 13:
            is_test_set = True
        else:
            is_training_set = True
    elif dataset_index == 'DU':  # Urdu
        if record[3] == "V":
            is_validation_set = True
        elif record[3] == "T":
            is_test_set = True
        else:
            is_training_set = True
    elif dataset_index == 'DR':  # RAVDESS
        actor_number = int(record[21]) * 10 + int(record[22])
        if actor_number == 17 or actor_number == 18 or actor_number == 19 or actor_number == 20:
            is_validation_set = True
        elif actor_number == 21 or actor_number == 22 or actor_number == 23 or actor_number == 24:
            is_test_set = True
        else:
            is_training_set = True
    elif dataset_index == 'DS':  # SAVEE
        actor_name = record[3] + "" + record[4]
        if actor_name == "KL":
            is_test_set = True
        elif actor_name == "JK":
            is_validation_set = True
        else:
            is_training_set = True

    return is_training_set, is_validation_set, is_test_set
