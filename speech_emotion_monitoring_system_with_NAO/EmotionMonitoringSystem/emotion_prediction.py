import numpy as np
import tensorflow as tf
import time

from acrnn1 import acrnn
from feature_extraction import extract_speech_feature

tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 60, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')

FLAGS = tf.app.flags.FLAGS

Emotions = {0: "Angry", 1: "Sad", 2: "Happy", 3: "Neutral"}

X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
Ylogits = acrnn(X, is_training=False)

def predict(record_folder_path, num_of_record, num_of_segments):
    start = time.time()

    # Extract Mel-spec Feature of Records
    record_list, speech_data, pernums = extract_speech_feature(record_folder_path, num_of_record, num_of_segments)
    time_feature_extraction = time.time()
    print("Feature Extraction Time:", time_feature_extraction - start)

    speech_segment_num = speech_data.shape[0]
    speech_num = pernums.shape[0]

    model_path = "../SERModel/model4.ckpt-1881"

    with tf.Session() as sess:
        # load model
        tf.train.Saver().restore(sess, model_path)
        time_model_loading = time.time()
        print("Model Loading Time:", time_model_loading - time_feature_extraction)

        # predict emotion
        y_pred_segment = np.empty((speech_segment_num, FLAGS.num_classes), dtype=np.float32)
        y_pred = np.empty((speech_num, 4), dtype=np.float32)
        index = 0

        y_pred_segment = sess.run(Ylogits, feed_dict={X: speech_data})

        for s in range(speech_num):
            y_pred[s, :] = np.max(y_pred_segment[index:index + pernums[s], :], 0)
            index = index + pernums[s]

        predicted_emotion_label = np.argmax(y_pred, 1)
        predicted_emotion = [Emotions[label] for label in predicted_emotion_label]

        time_emotion_prediction = time.time()
        print("Emotion Prediction:", time_emotion_prediction - time_model_loading)

        end = time.time()
        print("Total time used for do SER on 20 seconds' record:", end - start)

        print ("*****************************************************************")
        print ("Emotion Prediction: ", predicted_emotion)
        print ("*****************************************************************")

    return record_list, predicted_emotion
