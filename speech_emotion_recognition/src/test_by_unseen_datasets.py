#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Usage: test_by_unseen_datasets.py

Description: test the model by IEMOCAP (cross corpus, seen language), CASIA (cross-corpus, unseen language)
"""

# from __future__ import absolute_import
from __future__ import division

import os
import cPickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix as confusion

from acrnn import acrnn
from support import dense_to_one_hot

tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 60, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')

tf.app.flags.DEFINE_string('testdata_path', './test_data/IEMOCAP.pkl', 'data from IEMOCAP normalized by neutral')
# tf.app.flags.DEFINE_string('testdata_path', './test_data/CASIA.pkl', 'data from CASIA normalized by neutral')

FLAGS = tf.app.flags.FLAGS


def load_data(in_dir):
    f = open(in_dir, 'rb')
    test_data, test_label, Test_label, pernums_test = cPickle.load(f)
    return test_data, test_label, Test_label, pernums_test


def test():
    # load data
    test_data, test_label, Test_label, pernums_test = load_data(FLAGS.testdata_path)

    test_label = dense_to_one_hot(test_label, FLAGS.num_classes)
    Test_label = dense_to_one_hot(Test_label, FLAGS.num_classes)
    test_size = test_data.shape[0]
    tnum = pernums_test.shape[0]

    # load model
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    Y = tf.placeholder(tf.int32, shape=[None, FLAGS.num_classes])
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = acrnn(X, is_training=is_training, dropout_keep_prob=keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Ylogits)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    folder_path = "../pre_trained_models/model_optimize_UA_times_sad"
    models_name = ["model4.ckpt-1881"]

    with tf.Session() as sess:
        sess.run(init)
        for model_name in models_name:
            model_path = os.path.join(folder_path, model_name)
            saver.restore(sess, model_path)

            # test on combined testing set
            test_iter = divmod(test_size, FLAGS.batch_size)[0]
            y_pred_test = np.empty((test_size, FLAGS.num_classes), dtype=np.float32)
            y_test = np.empty((tnum, 4), dtype=np.float32)
            index = 0
            cost_test = 0
            if test_size < FLAGS.batch_size:
                loss, y_pred_test = sess.run([cross_entropy, Ylogits],
                                             feed_dict={X: test_data, Y: Test_label, is_training: False, keep_prob: 1})
                cost_test = cost_test + np.sum(loss)
            for v in range(test_iter):
                v_begin = v * FLAGS.batch_size
                v_end = (v + 1) * FLAGS.batch_size
                if v == test_iter - 1:
                    if v_end < test_size:
                        v_end = test_size
                loss, y_pred_test[v_begin:v_end, :] = sess.run([cross_entropy, Ylogits],
                                                               feed_dict={X: test_data[v_begin:v_end],
                                                                          Y: Test_label[v_begin:v_end],
                                                                          is_training: False, keep_prob: 1})
                cost_test = cost_test + np.sum(loss)
            cost_test = cost_test / test_size
            for s in range(tnum):
                y_test[s, :] = np.max(y_pred_test[index:index + pernums_test[s], :], 0)
                index = index + pernums_test[s]

            # generate confusion matrix
            test_conf = confusion(np.argmax(test_label, 1), np.argmax(y_test, 1))

            # calculate macro average metrics
            test_recall_ua = recall(np.argmax(test_label, 1), np.argmax(y_test, 1), average='macro')
            test_precision_ua = precision(np.argmax(test_label, 1), np.argmax(y_test, 1), average='macro')

            # there are two ways of calculating macro f1 score
            # way 1: f1 of unweighted average recall and precision
            # test_f1_ua = 2 * test_recall_ua * test_precision_ua / (test_recall_ua + test_precision_ua)
            # way 2: unweighted average of class-wise f1 scores
            test_f1_ua = f1(np.argmax(test_label, 1), np.argmax(y_test, 1), average='macro')

            # calculate class-wise metrics
            emotion_indices = {"Angry": 0, "Sad": 1, "Happy": 2, "Neutral": 3}
            test_emotion_recalls = recall(np.argmax(test_label, 1), np.argmax(y_test, 1), average=None)
            test_emotion_precisions = precision(np.argmax(test_label, 1), np.argmax(y_test, 1), average=None)
            test_emotion_F1_scores = f1(np.argmax(test_label, 1), np.argmax(y_test, 1), average=None)

            print ("*****************************************************************")
            print ("Model: ", model_path)
            print ("Test cost: %2.3g" % cost_test)
            print ('Test Confusion Matrix:["ang","sad","hap","neu"]')
            print (test_conf)
            print ("*****************************************************************")
            print("Test size: ", tnum)
            for emotion in ["Angry", "Sad", "Happy", "Neutral"]:
                index = emotion_indices[emotion]
                print emotion, ":{ Recall:", test_emotion_recalls[index], ", Precision:", test_emotion_precisions[
                    index], ", F1 Score:", test_emotion_F1_scores[index], "}"
            print("UA: { Recall:", test_recall_ua, ", Precision:", test_precision_ua, ", F1 Score:", test_f1_ua, "}")
            print ("*****************************************************************")


if __name__ == '__main__':
    test()
