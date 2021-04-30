#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2017

@author: hxj
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
from acrnn1 import acrnn
import cPickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# paper set: num_epoch = 5000; learning_rate = 0.00001
tf.app.flags.DEFINE_integer('num_epoch', 2001, 'The number of epoches for training.')
tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 60, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('is_adam', True, 'whether to use adam optimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.00005, 'learning rate of Adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1, 'the prob of every unit keep in dropout layer')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')

tf.app.flags.DEFINE_string('traindata_path', './CombinedEmotions_40.pkl', 'total dataset includes training set')
tf.app.flags.DEFINE_string('validdata_path', 'inputs/valid.pkl', 'total dataset includes valid set')
tf.app.flags.DEFINE_string('checkpoint', './checkpoint_lr_0.00005_corrected/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('model_name', 'model4.ckpt', 'model name')

FLAGS = tf.app.flags.FLAGS


def load_data(in_dir):
    f = open(in_dir, 'rb')
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = cPickle.load(
        f)
    # train_data,train_label,test_data,test_label,valid_data,valid_label = cPickle.load(f)
    return train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # print("num_labels", num_labels)
    # print("num_classes", num_classes)
    # print("index_offset", index_offset)
    # print("labels_dense", labels_dense)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def train():
    print("start training, number of epochs: ", FLAGS.num_epoch)
    ##### For plotting #####  
    colors = {}
    colors["ua"] = "tab:purple"
    colors["sad"] = "tab:blue"
    colors["happy"] = "tab:orange"
    colors["angry"] = "tab:red"
    colors["neutral"] = "tab:green"
    
    x_epoch = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.int)
    y_ua = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_sad_acc = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_happy_acc = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_neutral_acc = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_angry_acc = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    
    ##### Load data #####
    
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = load_data(
        FLAGS.traindata_path)

    print("num_classes", FLAGS.num_classes)
    print("valid_size", valid_data.shape[0])
    print("dataset_size", train_data.shape[0])
    print("vnum", pernums_valid.shape[0])

    train_label = dense_to_one_hot(train_label, FLAGS.num_classes)
    valid_label = dense_to_one_hot(valid_label, FLAGS.num_classes)
    Valid_label = dense_to_one_hot(Valid_label, FLAGS.num_classes)
    valid_size = valid_data.shape[0]
    dataset_size = train_data.shape[0]
    vnum = pernums_valid.shape[0]
    best_valid_uw = 0

    ##########tarin model###########
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    Y = tf.placeholder(tf.int32, shape=[None, FLAGS.num_classes])
    is_training = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = acrnn(X, is_training=is_training, dropout_keep_prob=keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Ylogits)
    cost = tf.reduce_mean(cross_entropy)
    var_trainable_op = tf.trainable_variables()
    if FLAGS.is_adam:
        # not apply gradient clipping
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    else:
        # apply gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, var_trainable_op), 5)
        opti = tf.train.AdamOptimizer(lr)
        train_op = opti.apply_gradients(zip(grads, var_trainable_op))
    correct_pred = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(FLAGS.num_epoch):
            # learning_rate = FLAGS.learning_rate
            start = (i * FLAGS.batch_size) % dataset_size
            end = min(start + FLAGS.batch_size, dataset_size)
            [_, tcost, tracc] = sess.run([train_op, cost, accuracy],
                                         feed_dict={X: train_data[start:end, :, :, :], Y: train_label[start:end, :],
                                                    is_training: True, keep_prob: FLAGS.dropout_keep_prob,
                                                    lr: FLAGS.learning_rate})
            if i % 5 == 0:
                # for valid data
                valid_iter = divmod((valid_size), FLAGS.batch_size)[0]
                y_pred_valid = np.empty((valid_size, FLAGS.num_classes), dtype=np.float32)
                y_valid = np.empty((vnum, 4), dtype=np.float32)
                index = 0
                cost_valid = 0
                if (valid_size < FLAGS.batch_size):
                    loss, y_pred_valid = sess.run([cross_entropy, Ylogits],
                                                  feed_dict={X: valid_data, Y: Valid_label, is_training: False,
                                                             keep_prob: 1})
                    cost_valid = cost_valid + np.sum(loss)
                for v in range(valid_iter):
                    v_begin = v * FLAGS.batch_size
                    v_end = (v + 1) * FLAGS.batch_size
                    if (v == valid_iter - 1):
                        if (v_end < valid_size):
                            v_end = valid_size
                    loss, y_pred_valid[v_begin:v_end, :] = sess.run([cross_entropy, Ylogits],
                                                                    feed_dict={X: valid_data[v_begin:v_end],
                                                                               Y: Valid_label[v_begin:v_end],
                                                                               is_training: False, keep_prob: 1})
                    cost_valid = cost_valid + np.sum(loss)
                cost_valid = cost_valid / valid_size
                for s in range(vnum):
                    y_valid[s, :] = np.max(y_pred_valid[index:index + pernums_valid[s], :], 0)
                    index = index + pernums_valid[s]

                valid_acc_uw = recall(np.argmax(valid_label, 1), np.argmax(y_valid, 1), average='macro')
                valid_conf = confusion(np.argmax(valid_label, 1), np.argmax(y_valid, 1))
                if valid_acc_uw > best_valid_uw:
                    best_valid_uw = valid_acc_uw
                    best_valid_conf = valid_conf
                    saver.save(sess, os.path.join(FLAGS.checkpoint, FLAGS.model_name), global_step=i + 1)
                print ("*****************************************************************")
                print ("Epoch: %05d" % (i + 1))
                print ("Training cost: %2.3g" % tcost)
                print ("Training accuracy: %3.4g" % tracc)
                print ("Valid cost: %2.3g" % cost_valid)
                print ("Valid_UA: %3.4g" % valid_acc_uw)
                print ("Best valid_UA: %3.4g" % best_valid_uw)
                print ('Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print (valid_conf)
                print ('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print (best_valid_conf)
                print ("*****************************************************************")
                num_validate = int(i / 5)
                total_angry = valid_conf[0][0] + valid_conf[0][1] + valid_conf[0][2] + valid_conf[0][3]
                total_sad = valid_conf[1][0] + valid_conf[1][1] + valid_conf[1][2] + valid_conf[1][3]
                total_happy = valid_conf[2][0] + valid_conf[2][1] + valid_conf[2][2] + valid_conf[2][3]
                total_neutral = valid_conf[3][0] + valid_conf[3][1] + valid_conf[3][2] + valid_conf[3][3]
                x_epoch[num_validate] = i
                y_ua[num_validate] = valid_acc_uw
                y_angry_acc[num_validate] = valid_conf[0][0] / total_angry
                y_sad_acc[num_validate] = valid_conf[1][1] / total_sad
                y_happy_acc[num_validate] = valid_conf[2][2] / total_happy
                y_neutral_acc[num_validate] = valid_conf[3][3] / total_neutral
                print("UA: %3.4g" % y_ua[num_validate])
                print("Angry: %3.4g" % y_angry_acc[num_validate])
                print("Sad: %3.4g" % y_sad_acc[num_validate])
                print("Happy: %3.4g" % y_happy_acc[num_validate])
                print("Neutral: %3.4g" % y_neutral_acc[num_validate])      
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize = (12,18))
    
    # ax1 is the complete plot with 5 lines
    ax1.plot(x_epoch, y_ua, color=colors["ua"], label="UA")
    ax1.plot(x_epoch, y_sad_acc, color=colors["sad"], label="Sad")
    ax1.plot(x_epoch, y_happy_acc, color=colors["happy"], label="Happy")
    ax1.plot(x_epoch, y_angry_acc, color=colors["angry"], label="Angry")
    ax1.plot(x_epoch, y_neutral_acc, color=colors["neutral"], label="Neutral")
    ax1.set_title('Accuracy')
    ax1.legend() 
    
    # ax2 is Sad vs. UA
    ax2.plot(x_epoch, y_sad_acc, color=colors["sad"], label="Sad")
    ax2.plot(x_epoch, y_ua, color=colors["ua"], label="UA")
    ax2.set_title('Sad vs. UA Acc')
    
    # ax3 is Sad vs. Happy
    ax3.plot(x_epoch, y_sad_acc, color=colors["sad"], label="Sad")
    ax3.plot(x_epoch, y_happy_acc, color=colors["happy"], label="Happy")
    ax3.set_title('Sad vs. Happy Acc')
    
    # ax4 is Sad vs. Angry
    ax4.plot(x_epoch, y_sad_acc, color=colors["sad"], label="Sad")
    ax4.plot(x_epoch, y_angry_acc, color=colors["angry"], label="Angry")
    ax4.set_title('Sad vs. Angry Acc')
    
    # ax5 is Sad vs. Neutral
    ax5.plot(x_epoch, y_sad_acc, color=colors["sad"], label="Sad")
    ax5.plot(x_epoch, y_neutral_acc, color=colors["neutral"], label="Neutral")
    ax5.set_title('Sad vs. Neutral Acc')
    
    fig_name = "corrected_epochs_" + str(i) + "_all_accuracy_plots.png"
    plt.savefig(fig_name)
    # plt.show()


if __name__ == '__main__':
    train()
