# Normalized by the mean and std of Training Set Neutral
import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import cPickle
import logging

eps = 1e-5


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def generate_label(emotion, classnum):
    label = -1
    if (emotion == 'ang'):
        label = 0
    elif (emotion == 'sad'):
        label = 1
    elif (emotion == 'hap'):
        label = 2
    elif (emotion == 'neu'):
        label = 3
    else:
        label = 4
    return label


def load_data():
    f = open('./zscore_4_datasets_combined_SI40.pkl', 'rb')
    dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3 = cPickle.load(f)
    return dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3


def split_dataset(record):
    is_training_set = False
    is_validation_set = False
    is_test_set = False

    dataset_index = record[0:2]
    if (dataset_index == 'DG'):
        # Emodb
        speaker_number = int(record[3]) * 10 + int(record[4])
        
        # print("Emodb", speaker_number)
        if (speaker_number == 3 or speaker_number == 10):
            is_validation_set = True
        elif (speaker_number == 12 or speaker_number == 13):
            is_test_set = True
        else:
            is_training_set = True
    elif (dataset_index == 'DU'):
        # Urdu
        # print("Urdu", record[3])
        if record[3] == "V":
            is_validation_set = True
        elif record[3] == "T":
            is_test_set = True
        else:
            is_training_set = True
    elif (dataset_index == 'DR'):
        # RAVDESS
        actor_number = int(record[21]) * 10 + int(record[22])
        # print("RAVDESS", actor_number)
        if (actor_number == 17 or actor_number == 18 or actor_number == 19 or actor_number == 20):
            is_validation_set = True
        elif (actor_number == 21 or actor_number == 22 or actor_number == 23 or actor_number == 24):
            is_test_set = True
        else:
            is_training_set = True
    elif (dataset_index == 'DS'):
        # SAVEE
        actor_name = record[3] + "" + record[4]
        # if record[6] == 'a' or record[6] == 'u':
        #     record_number = int(record[7]) * 10 + int(record[8])
        # else:
        #     record_number = int(record[6]) * 10 + int(record[7])
        if (actor_name == "KL"):
            is_test_set = True
        elif (actor_name == "JK"):
            is_validation_set = True
        else:
            is_training_set = True

    return is_training_set, is_validation_set, is_test_set


def read_combined_dataset():
    eps = 1e-5
    datasets_index = ["DG", "DU", "DR", "DS"]

    tnum = 324  # the number of test utterance
    vnum = 329  # the number of validate utterance
    test_num = 510 # the number of test 2s segments
    valid_num = 527 # the number of validate 2s segments

    trnum = 1336 # the number of train utterance
    train_num = 2158 # the number of train 2s segments
    filter_num = 40
    pernums_test = np.arange(tnum) # remember each test utterance contain how many segments
    pernums_valid = np.arange(vnum) # remember each valid utterance contain how many segments
    
    # EmoDB, Urdu, RAVDESS, SAVEE
    rootdir = '/home/yiyang/Combined_4_dataset_SI'

    dataset_mean1, dataset_std1, dataset_mean2, dataset_std2, dataset_mean3, dataset_std3 = load_data()

    # 1336 train utterance --> 2158 train 2s segments
    angnum = 463  # 0
    sadnum = 515  # 1
    hapnum = 478  # 2
    neunum = 702  # 3

    pernum = np.min([hapnum, angnum, sadnum, neunum])
    # valid_num = divmod((train_num),10)[0]
    train_label = np.empty((train_num, 1), dtype=np.int8)
    test_label = np.empty((tnum, 1), dtype=np.int8)
    valid_label = np.empty((vnum, 1), dtype=np.int8)
    Test_label = np.empty((test_num, 1), dtype=np.int8)
    Valid_label = np.empty((valid_num, 1), dtype=np.int8)
    train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)
    valid_data = np.empty((valid_num, 300, filter_num, 3), dtype=np.float32)

    tnum = 0
    vnum = 0
    train_num = 0
    test_num = 0
    valid_num = 0
    train_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    test_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    valid_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    
    dataset_train_nums = {"DG": 0, "DU": 0, "DR": 0, "DS": 0}
    dataset_valid_nums = {"DG": 0, "DU": 0, "DR": 0, "DS": 0}
    dataset_test_nums = {"DG": 0, "DU": 0, "DR": 0, "DS": 0}

    all_emotions = ["Angry", "Happy", "Sad", "Neutral"]

    for emotion_name in os.listdir(rootdir):
        emotion = ''
        if (emotion_name[0] == 'A'): # angry
            emotion = 'ang'
        if (emotion_name[0] == 'H'): # happy
            emotion = 'hap'
        if (emotion_name[0] == 'S'): # sad
            emotion = 'sad'
        if (emotion_name[0] == 'N'): # neu
            emotion = 'neu'

        if emotion_name not in all_emotions:
            continue

        sub_dir = os.path.join(rootdir, emotion_name)
        for record in os.listdir(sub_dir):
            if record.startswith('.'):
                continue
            
            record_path = os.path.join(sub_dir, record)
            dataset_index = record[0:2]

            data, time, rate = read_file(record_path)
            mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
            delta1 = ps.delta(mel_spec, 2)
            delta2 = ps.delta(delta1, 2)
            time = mel_spec.shape[0]

            is_training_set, is_validation_set, is_test_set = split_dataset(record)

            if (is_training_set):
                # training set
                dataset_train_nums[dataset_index] = dataset_train_nums[dataset_index] + 1
                if (time <= 300):
                    part = mel_spec
                    delta11 = delta1
                    delta21 = delta2
                    part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant', constant_values=0)
                    delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant', constant_values=0)
                    delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant', constant_values=0)

                    train_data[train_num, :, :, 0] = (part - dataset_mean1[dataset_index]) / (
                                dataset_std1[dataset_index] + eps)
                    train_data[train_num, :, :, 1] = (delta11 - dataset_mean2[dataset_index]) / (
                                dataset_std2[dataset_index] + eps)
                    train_data[train_num, :, :, 2] = (delta21 - dataset_mean3[dataset_index]) / (
                                dataset_std3[dataset_index] + eps)

                    em = generate_label(emotion, 6)
                    train_label[train_num] = em
                    train_emt[emotion] = train_emt[emotion] + 1
                    train_num = train_num + 1

                else:
                    # if (emotion in ['ang', 'neu', 'sad', 'hap']):
                    for i in range(2):
                        if (i == 0):
                            begin = 0
                            end = begin + 300
                        else:
                            begin = time - 300
                            end = time

                        part = mel_spec[begin:end, :]
                        delta11 = delta1[begin:end, :]
                        delta21 = delta2[begin:end, :]

                        train_data[train_num, :, :, 0] = (part - dataset_mean1[dataset_index]) / (
                                    dataset_std1[dataset_index] + eps)
                        train_data[train_num, :, :, 1] = (delta11 - dataset_mean2[dataset_index]) / (
                                    dataset_std2[dataset_index] + eps)
                        train_data[train_num, :, :, 2] = (delta21 - dataset_mean3[dataset_index]) / (
                                    dataset_std3[dataset_index] + eps)

                        em = generate_label(emotion, 6)
                        train_label[train_num] = em
                        train_emt[emotion] = train_emt[emotion] + 1
                        train_num = train_num + 1
            else:
                em = generate_label(emotion, 6)
                if (is_test_set):
                    # test_set
                    dataset_test_nums[dataset_index] = dataset_test_nums[dataset_index] + 1
                    test_label[tnum] = em
                    if (time <= 300):
                        pernums_test[tnum] = 1
                        part = mel_spec
                        delta11 = delta1
                        delta21 = delta2
                        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant', constant_values=0)
                        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant', constant_values=0)
                        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant', constant_values=0)

                        test_data[test_num, :, :, 0] = (part - dataset_mean1[dataset_index]) / (
                                dataset_std1[dataset_index] + eps)
                        test_data[test_num, :, :, 1] = (delta11 - dataset_mean2[dataset_index]) / (
                                dataset_std2[dataset_index] + eps)
                        test_data[test_num, :, :, 2] = (delta21 - dataset_mean3[dataset_index]) / (
                                dataset_std3[dataset_index] + eps)

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
                                end = time
                                begin = time - 300
                            part = mel_spec[begin:end, :]
                            delta11 = delta1[begin:end, :]
                            delta21 = delta2[begin:end, :]
                            test_data[test_num, :, :, 0] = (part - dataset_mean1[dataset_index]) / (
                                    dataset_std1[dataset_index] + eps)
                            test_data[test_num, :, :, 1] = (delta11 - dataset_mean2[dataset_index]) / (
                                    dataset_std2[dataset_index] + eps)
                            test_data[test_num, :, :, 2] = (delta21 - dataset_mean3[dataset_index]) / (
                                    dataset_std3[dataset_index] + eps)

                            test_emt[emotion] = test_emt[emotion] + 1
                            Test_label[test_num] = em
                            test_num = test_num + 1
                elif (is_validation_set):
                    # valid_set
                    dataset_valid_nums[dataset_index] = dataset_valid_nums[dataset_index] + 1
                    em = generate_label(emotion, 6)
                    valid_label[vnum] = em
                    if (time <= 300):
                        pernums_valid[vnum] = 1
                        part = mel_spec
                        delta11 = delta1
                        delta21 = delta2
                        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant', constant_values=0)
                        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant', constant_values=0)
                        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant', constant_values=0)

                        valid_data[valid_num, :, :, 0] = (part - dataset_mean1[dataset_index]) / (
                                dataset_std1[dataset_index] + eps)
                        valid_data[valid_num, :, :, 1] = (delta11 - dataset_mean2[dataset_index]) / (
                                dataset_std1[dataset_index] + eps)
                        valid_data[valid_num, :, :, 2] = (delta21 - dataset_mean3[dataset_index]) / (
                                dataset_std1[dataset_index] + eps)

                        valid_emt[emotion] = valid_emt[emotion] + 1
                        Valid_label[valid_num] = em
                        valid_num = valid_num + 1
                        vnum = vnum + 1
                    else:
                        pernums_valid[vnum] = 2
                        vnum = vnum + 1
                        for i in range(2):
                            if (i == 0):
                                begin = 0
                                end = begin + 300
                            else:
                                end = time
                                begin = time - 300
                            part = mel_spec[begin:end, :]
                            delta11 = delta1[begin:end, :]
                            delta21 = delta2[begin:end, :]
                            valid_data[valid_num, :, :, 0] = (part - dataset_mean1[dataset_index]) / (
                                    dataset_std1[dataset_index] + eps)
                            valid_data[valid_num, :, :, 1] = (delta11 - dataset_mean2[dataset_index]) / (
                                    dataset_std1[dataset_index] + eps)
                            valid_data[valid_num, :, :, 2] = (delta21 - dataset_mean3[dataset_index]) / (
                                    dataset_std1[dataset_index] + eps)
                            valid_emt[emotion] = valid_emt[emotion] + 1
                            Valid_label[valid_num] = em
                            valid_num = valid_num + 1
    
    print("train_num: ", train_num)
    print("valid_num: ", valid_num)
    print("test_num: ", test_num)
    print("vnum: ", vnum)
    print("tnum: ", tnum)
    print("dataset train nums: ", dataset_train_nums)
    print("dataset validate nums: ", dataset_valid_nums)
    print("dataset test nums: ", dataset_test_nums)
    hap_index = np.arange(hapnum)
    neu_index = np.arange(neunum)
    sad_index = np.arange(sadnum)
    ang_index = np.arange(angnum)

    a0 = 0
    s1 = 0
    h2 = 0
    n3 = 0

    for l in range(train_num):
        if (train_label[l] == 0):
            ang_index[a0] = l
            a0 = a0 + 1
        elif (train_label[l] == 1):
            sad_index[s1] = l
            s1 = s1 + 1
        elif (train_label[l] == 2):
            hap_index[h2] = l
            h2 = h2 + 1
        elif (train_label[l] == 3):
            neu_index[n3] = l
            n3 = n3 + 1

    for m in range(1):
        np.random.shuffle(neu_index)
        np.random.shuffle(hap_index)
        np.random.shuffle(sad_index)
        np.random.shuffle(ang_index)
        # define emotional array
        hap_label = np.empty((pernum, 1), dtype=np.int8)
        ang_label = np.empty((pernum, 1), dtype=np.int8)
        sad_label = np.empty((pernum, 1), dtype=np.int8)
        neu_label = np.empty((pernum, 1), dtype=np.int8)

        hap_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)
        neu_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)
        sad_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)
        ang_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)

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
        print train_label.shape
        print train_emt
        print test_emt
        print valid_emt
        output = './CombinedEmotions_4_datasets_combined_SI40.pkl'
        f = open(output, 'wb')
        cPickle.dump((Train_data, Train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label,
                      pernums_test, pernums_valid), f)
        f.close()
    return


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    read_combined_dataset()
