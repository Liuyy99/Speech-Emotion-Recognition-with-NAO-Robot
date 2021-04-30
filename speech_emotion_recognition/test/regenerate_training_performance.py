from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
from src.acrnn1 import acrnn
import cPickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os

# paper set: num_epoch = 5000; learning_rate = 0.00001
tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 60, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('is_adam', True, 'whether to use adam optimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate of Adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1, 'the prob of every unit keep in dropout layer')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')

tf.app.flags.DEFINE_string('traindata_path', './CombinedEmotions_4_datasets_combined_SI40.pkl', 'total dataset includes training set')

FLAGS = tf.app.flags.FLAGS


def load_data(in_dir):
    f = open(in_dir, 'rb')
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = cPickle.load(f)
    return valid_data,valid_label,Valid_label,pernums_valid


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
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

def test():
    #####load data##########

    valid_data,valid_label,Valid_label,pernums_valid = load_data(FLAGS.traindata_path)

    valid_label = dense_to_one_hot(valid_label, FLAGS.num_classes)
    Valid_label = dense_to_one_hot(Valid_label, FLAGS.num_classes)
    valid_size = valid_data.shape[0]
    vnum = pernums_valid.shape[0]

    ##########tarin model###########
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    Y = tf.placeholder(tf.int32, shape=[None, FLAGS.num_classes])
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = acrnn(X, is_training=is_training, dropout_keep_prob=keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Ylogits)

    correct_pred = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

#     folder_path = "./checkpoint_lr_0.00001_corrected"
#     models_name = ["model4.ckpt-1001", "model4.ckpt-821", "model4.ckpt-791", "model4.ckpt-756", "model4.ckpt-676"]
#     folder_path = "./checkpoint_lr_0.00001_optimize_ua"
#     models_name = ["model4.ckpt-1591", "model4.ckpt-1551", "model4.ckpt-1211", "model4.ckpt-1151", "model4.ckpt-1106", "model4.ckpt-946", "model4.ckpt-906"]
#     folder_path = "./checkpoint_lr_0.00001_optimize_ua_times_sad"
#     models_name = ["model4.ckpt-1991","model4.ckpt-1591", "model4.ckpt-1551", "model4.ckpt-1391", "model4.ckpt-1266", "model4.ckpt-1206", "model4.ckpt-1196", "model4.ckpt-1191", "model4.ckpt-1081"]

#     folder_path = "./checkpoint_lr_0.00001_optimize_ua_times_sad_all_dataset_combined_not_fully_SI"
#     models_name = ["model4.ckpt-3496", "model4.ckpt-3261", "model4.ckpt-2781", "model4.ckpt-2721", "model4.ckpt-2596", "model4.ckpt-2061", "model4.ckpt-1881"]
#     folder_path = "./checkpoint_lr_0.00001_optimize_ua_all_dataset_combined_not_fully_SI"
#     models_name = ["model4.ckpt-3261", "model4.ckpt-3016", "model4.ckpt-3011", "model4.ckpt-1941", "model4.ckpt-1766"]
    folder_path = "./checkpoint_lr_0.00001_optimize_ua_times_sad_4_dataset_combined_no_casia_iemocap"
    models_name = ["model4.ckpt-5001", "model4.ckpt-1281", "model4.ckpt-1196", "model4.ckpt-1081", "model4.ckpt-1076", "model4.ckpt-986"]
#     folder_path = "./checkpoint_lr_0.00001_optimize_sad_F1_score_4_dataset_combined_no_casia_iemocap"
#     models_name = ["model4.ckpt-741", "model4.ckpt-591", "model4.ckpt-561", "model4.ckpt-546", "model4.ckpt-461"]
    
#     folder_path = "./checkpoint_lr_0.00001_optimize_ua_times_sad_all_dataset_combined"
#     models_name = ["model4.ckpt-4471", "model4.ckpt-3431", "model4.ckpt-3021", "model4.ckpt-2906", "model4.ckpt-2901", "model4.ckpt-2151", "model4.ckpt-1801"]
    
#     folder_path = "./checkpoint_lr_0.00001_optimize_ua_times_sad_all_dataset_combined_2nd_round"
#     models_name = ["model4.ckpt-2556", "model4.ckpt-1921", "model4.ckpt-1166", "model4.ckpt-1161"]
    with tf.Session() as sess:
        sess.run(init)
        for model_name in models_name:
            model_path = os.path.join(folder_path, model_name)
            saver.restore(sess, model_path)

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
            
            print ("*****************************************************************")
            print ("Model: ", model_path)
            print ("Test cost: %2.3g" % cost_valid)
            print ('Test Confusion Matrix:["ang","sad","hap","neu"]')
            print (valid_conf)
            print ("*****************************************************************")
            print("Valid size: ", sum(emotion_num_actual))
            # Calculate Recall, Precision, F1 Score
            num_emotion = 4
            unweighted_average_recall = 0
            unweighted_average_precision = 0
            unweighted_average_F1_score = 0
            
            for emotion in ["Angry", "Sad", "Happy", "Neutral"]:
                emotion_recall, emotion_precision, emotion_F1_score = calculate_metric(
                    valid_conf, emotion_num_actual, emotion_num_predicted, emotion_index[emotion])
                unweighted_average_recall += emotion_recall
                unweighted_average_precision += emotion_precision
                unweighted_average_F1_score += emotion_F1_score
                print(emotion, ":{ Recall:", emotion_recall, ", Precision:", emotion_precision, ", F1 Score:", emotion_F1_score, "}")
            
            unweighted_average_recall = unweighted_average_recall / num_emotion
            unweighted_average_precision = unweighted_average_precision / num_emotion
            unweighted_average_F1_score = unweighted_average_F1_score / num_emotion
            print("UA: { Recall:", unweighted_average_recall, ", Precision:", unweighted_average_precision, ", F1 Score:", unweighted_average_F1_score, "}")

if __name__ == '__main__':
    test()