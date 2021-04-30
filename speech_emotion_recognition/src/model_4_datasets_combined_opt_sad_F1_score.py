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
tf.app.flags.DEFINE_integer('num_epoch', 3001, 'The number of epoches for training.')
tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 60, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('is_adam', True, 'whether to use adam optimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'learning rate of Adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1, 'the prob of every unit keep in dropout layer')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')

tf.app.flags.DEFINE_string('traindata_path', './CombinedEmotions_4_datasets_combined_SI40.pkl', 'total dataset includes training set')
tf.app.flags.DEFINE_string('checkpoint1', './checkpoint_lr_0.00001_optimize_ua_4_dataset_combined_no_casia_iemocap_round3/', 'the checkpoint1 dir')
tf.app.flags.DEFINE_string('checkpoint2', './checkpoint_lr_0.00001_optimize_ua_times_sad_4_dataset_combined_no_casia_iemocap_round3/', 'the checkpoint2 dir')
tf.app.flags.DEFINE_string('checkpoint3', './checkpoint_lr_0.00001_optimize_sad_F1_score_4_dataset_combined_no_casia_iemocap_round3/', 'the checkpoint2 dir')

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

def calculate_metric(confusion_matrix, emotion_num_actual, emotion_num_predicted, emotion):
    emotion_recall = confusion_matrix[emotion][emotion] / emotion_num_actual[emotion]
    if emotion_num_predicted[emotion] == 0:
        emotion_precision = 0
        emotion_F1_score = 0
    else:
        emotion_precision = confusion_matrix[emotion][emotion] / emotion_num_predicted[emotion]
        if emotion_recall == 0:
            emotion_F1_score = 0
        else:
            emotion_F1_score = 2 * emotion_precision * emotion_recall / (emotion_precision + emotion_recall)
    return emotion_recall, emotion_precision, emotion_F1_score

def train():
    print("start training, number of epochs: ", FLAGS.num_epoch)
    print("learning rate: ", FLAGS.learning_rate)
    print("checkpoint path 1: ", FLAGS.checkpoint1)
    print("checkpoint path 2: ", FLAGS.checkpoint2)
    print("checkpoint path 3: ", FLAGS.checkpoint3)
    ##### For plotting #####  
    colors = {}
    colors["ua"] = "tab:purple"
    colors["sad"] = "tab:blue"
    colors["happy"] = "tab:orange"
    colors["angry"] = "tab:red"
    colors["neutral"] = "tab:green"
    
    x_epoch = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.int)
    training_acc = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    training_cost = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_ua = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_ua_f1 = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_sad_f1 = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_happy_f1 = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_neutral_f1 = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_angry_f1 = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_sad_recall = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_sad_precision = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    y_sad_F1_score = np.empty(int(FLAGS.num_epoch / 5 + 1), dtype=np.float)
    
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
    best_valid_uw_times_sad = 0
    best_valid_sad_F1_score = 0

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
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=30)
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

                valid_recall_uw = recall(np.argmax(valid_label, 1), np.argmax(y_valid, 1), average='macro')
                valid_conf = confusion(np.argmax(valid_label, 1), np.argmax(y_valid, 1))
                if valid_recall_uw >= best_valid_uw:
                    best_valid_uw = valid_recall_uw
                    best_valid_conf = valid_conf
                    saver.save(sess, os.path.join(FLAGS.checkpoint1, FLAGS.model_name), global_step=i + 1)
                    
                num_validate = int(i / 5)
                
                total_actual_angry = valid_conf[0][0] + valid_conf[0][1] + valid_conf[0][2] + valid_conf[0][3]
                total_actual_sad = valid_conf[1][0] + valid_conf[1][1] + valid_conf[1][2] + valid_conf[1][3]
                total_actual_happy = valid_conf[2][0] + valid_conf[2][1] + valid_conf[2][2] + valid_conf[2][3]
                total_actual_neutral = valid_conf[3][0] + valid_conf[3][1] + valid_conf[3][2] + valid_conf[3][3]
                
                total_predicted_angry = valid_conf[0][0] + valid_conf[1][0] + valid_conf[2][0] + valid_conf[3][0]
                total_predicted_sad = valid_conf[0][1] + valid_conf[1][1] + valid_conf[2][1] + valid_conf[3][1]
                total_predicted_happy = valid_conf[0][2] + valid_conf[1][2] + valid_conf[2][2] + valid_conf[3][2]
                total_predicted_neutral = valid_conf[0][3] + valid_conf[1][3] + valid_conf[2][3] + valid_conf[3][3]
                
                # Calculate Recall, Precision, F1 Score
                num_emotion = 4
                unweighted_average_recall = 0
                unweighted_average_precision = 0
                unweighted_average_F1_score = 0
                
                emotion_recalls = {
                    "Angry": 0,
                    "Sad": 0,
                    "Happy": 0, 
                    "Neutral": 0
                }
                
                emotion_precisions = {
                    "Angry": 0,
                    "Sad": 0,
                    "Happy": 0, 
                    "Neutral": 0
                }
                
                emotion_F1_scores = {
                    "Angry": 0,
                    "Sad": 0,
                    "Happy": 0, 
                    "Neutral": 0
                }
                
                emotion_num_actual = [
                    valid_conf[0][0] + valid_conf[0][1] + valid_conf[0][2] + valid_conf[0][3],
                    valid_conf[1][0] + valid_conf[1][1] + valid_conf[1][2] + valid_conf[1][3],
                    valid_conf[2][0] + valid_conf[2][1] + valid_conf[2][2] + valid_conf[2][3],
                    valid_conf[3][0] + valid_conf[3][1] + valid_conf[3][2] + valid_conf[3][3]
                ]

                emotion_num_predicted = [
                    valid_conf[0][0] + valid_conf[1][0] + valid_conf[2][0] + valid_conf[3][0],
                    valid_conf[0][1] + valid_conf[1][1] + valid_conf[2][1] + valid_conf[3][1],
                    valid_conf[0][2] + valid_conf[1][2] + valid_conf[2][2] + valid_conf[3][2],
                    valid_conf[0][3] + valid_conf[1][3] + valid_conf[2][3] + valid_conf[3][3]
                ]

                emotion_index = {
                    "Angry": 0,
                    "Sad": 1,
                    "Happy": 2,
                    "Neutral": 3
                }
                
                for emotion in ["Angry", "Sad", "Happy", "Neutral"]:
                    emotion_recall, emotion_precision, emotion_F1_score = calculate_metric(
                        valid_conf, emotion_num_actual, emotion_num_predicted, emotion_index[emotion])
                    emotion_F1_scores[emotion] = emotion_F1_score
                    emotion_recalls[emotion] = emotion_recall
                    emotion_precisions[emotion] = emotion_precision
                    unweighted_average_recall += emotion_recall
                    unweighted_average_precision += emotion_precision
                    unweighted_average_F1_score += emotion_F1_score

                unweighted_average_recall = unweighted_average_recall / num_emotion
                unweighted_average_precision = unweighted_average_precision / num_emotion
                unweighted_average_F1_score = unweighted_average_F1_score / num_emotion
                
                
                # For plotting
                x_epoch[num_validate] = i
                training_cost[num_validate] = tcost
                training_acc[num_validate] = tracc
                y_ua[num_validate] = valid_recall_uw
                y_ua_f1[num_validate] = unweighted_average_F1_score
                y_angry_f1[num_validate] = emotion_F1_scores["Angry"]
                y_sad_f1[num_validate] = emotion_F1_scores["Sad"]
                y_happy_f1[num_validate] = emotion_F1_scores["Happy"]
                y_neutral_f1[num_validate] = emotion_F1_scores["Neutral"]
                sad_F1_score = emotion_F1_scores["Sad"]
                
                # UA * Sad
                valid_recall_uw_times_sad = valid_recall_uw * emotion_recalls["Sad"]
                
                if valid_recall_uw_times_sad >= best_valid_uw_times_sad:
                    best_valid_uw_times_sad = valid_recall_uw_times_sad
                    best_valid_conf_uw_times_sad = valid_conf
                    saver.save(sess, os.path.join(FLAGS.checkpoint2, FLAGS.model_name), global_step=i + 1)
                    
                y_sad_recall[num_validate] = emotion_recalls["Sad"]
                y_sad_precision[num_validate] = emotion_precisions["Sad"]
                y_sad_F1_score[num_validate] = emotion_F1_scores["Sad"]
                
                if sad_F1_score >= best_valid_sad_F1_score:
                    best_valid_sad_F1_score = sad_F1_score
                    best_valid_conf_sad_F1_score = valid_conf
                    saver.save(sess, os.path.join(FLAGS.checkpoint3, FLAGS.model_name), global_step=i + 1)
                
                if i == FLAGS.num_epoch - 1:
                    saver.save(sess, os.path.join(FLAGS.checkpoint3, FLAGS.model_name), global_step=i + 1)
                    
                print("*****************************************************************")
                print("Epoch: %05d" % (i + 1))
                print("Training cost: %2.3g" % tcost)
                print("Training accuracy: %3.4g" % tracc)
                print("Valid cost: %2.3g" % cost_valid)
                print("Valid_UA_Recall: %3.4g" % valid_recall_uw)
                print("Best Valid UA Recall: %3.4g" % best_valid_uw)
                print("Best Valid UA * Sad: %3.4g" % best_valid_uw_times_sad)
                print("Best Valid F1 Score: %3.4g" % best_valid_sad_F1_score)
                print('Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print(valid_conf)
                print('Best Valid Confusion Matrix Optimize Sad F1 Score:["ang","sad","hap","neu"]')
                print(best_valid_conf_sad_F1_score)
                print("*****************************************************************")
                print("Valid size: ", total_actual_angry + total_actual_sad + total_actual_happy + total_actual_neutral)
                print("UA: %3.4g" % y_ua[num_validate])
                print("Sad Recall: %3.4g" % y_sad_recall[num_validate]) 
                print("Sad Precision: %3.4g" % y_sad_precision[num_validate]) 
                print("Sad F1 Score: %3.4g" % y_sad_F1_score[num_validate])
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, figsize = (12,36))
    fig.tight_layout()
    
    # ax1 is the complete plot with 5 lines
    ax1.plot(x_epoch, y_ua_f1, color=colors["ua"], label="UA")
    ax1.plot(x_epoch, y_sad_f1, color=colors["sad"], label="Sad")
    ax1.plot(x_epoch, y_happy_f1, color=colors["happy"], label="Happy")
    ax1.plot(x_epoch, y_angry_f1, color=colors["angry"], label="Angry")
    ax1.plot(x_epoch, y_neutral_f1, color=colors["neutral"], label="Neutral")
    ax1.set_title('Validation F1 Rate of Each Emotion')
    ax1.legend() 
    
    # ax2 is Sad vs. UA
    ax2.plot(x_epoch, y_sad_f1, color=colors["sad"], label="Sad")
    ax2.plot(x_epoch, y_ua_f1, color=colors["ua"], label="UA")
    ax2.set_title('Validation Sad vs. UA F1')
    
    # ax3 is Sad vs. Happy
    ax3.plot(x_epoch, y_sad_f1, color=colors["sad"], label="Sad")
    ax3.plot(x_epoch, y_happy_f1, color=colors["happy"], label="Happy")
    ax3.set_title('Validation Sad vs. Happy F1')
    
    # ax4 is Sad vs. Angry
    ax4.plot(x_epoch, y_sad_f1, color=colors["sad"], label="Sad")
    ax4.plot(x_epoch, y_angry_f1, color=colors["angry"], label="Angry")
    ax4.set_title('Validation Sad vs. Angry F1')
    
    # ax5 is Sad vs. Neutral
    ax5.plot(x_epoch, y_sad_f1, color=colors["sad"], label="Sad")
    ax5.plot(x_epoch, y_neutral_f1, color=colors["neutral"], label="Neutral")
    ax5.set_title('Validation Sad vs. Neutral F1')
    
    # ax6 Training Accuracy
    ax6.plot(x_epoch, training_acc, color=colors["ua"], label="UA")
    ax6.set_title('Training Accuracy')
    
    # ax7 Training Cost
    ax7.plot(x_epoch, training_cost, color=colors["ua"], label="UA")
    ax7.set_title('Training Cost')
    
    # ax8 is Happy vs. UA
    ax8.plot(x_epoch, y_happy_f1, color=colors["happy"], label="Happy")
    ax8.plot(x_epoch, y_ua_f1, color=colors["ua"], label="UA")
    ax8.set_title('Validation Happy vs. UA F1')
    
    # ax9 is Angry vs. UA
    ax9.plot(x_epoch, y_angry_f1, color=colors["angry"], label="Angry")
    ax9.plot(x_epoch, y_ua_f1, color=colors["ua"], label="UA")
    ax9.set_title('Validation Angry vs. UA F1')
    
    # ax10 is Neutral vs. UA
    ax10.plot(x_epoch, y_neutral_f1, color=colors["neutral"], label="Neutral")
    ax10.plot(x_epoch, y_ua_f1, color=colors["ua"], label="UA")
    ax10.set_title('Validation Neutral vs. UA F1')
    
    fig_name = "corrected_epochs_" + str(i) + "_4_dataset_combined_optimize_sad_F1_score_no_casia_iemocap_0.00001_round3.png"
    plt.savefig(fig_name)
    # plt.show()


if __name__ == '__main__':
    train()
