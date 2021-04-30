import shutil
import os
import csv
from speech_pre_processing import pre_process_record, get_record_length
from feature_extraction import get_segments_num
from emotion_prediction import predict
from report_generation import generate_report
import logging
logging.getLogger().setLevel("ERROR")

emo_record_path = './SER_emotion_record.csv'
record_length = 20  # seconds

record_name = '1614315147.66.wav'
record_path = '../SpeechRecording/1614315147.66.wav'

if not os.path.exists(emo_record_path):
    with open(emo_record_path, 'a') as csvfile:
        emo_record_writer = csv.writer(csvfile)
        emo_record_writer.writerow(('Timestamp', 'Emotions', 'Sad', 'Neutral', 'Angry', 'Happy'))

for i in range(4):
    speech_folder, num_speech = pre_process_record(record_path)
    if num_speech != 0:
        num_2s_segments = get_segments_num(speech_folder)
        print("num_speech: ", num_speech)
        print("num_2s_segments: ", num_2s_segments)
        record_list, emotions = predict(speech_folder, num_speech, num_2s_segments)
        emotion_durations = {
            "Angry": 0,
            "Sad": 0,
            "Happy": 0,
            "Neutral": 0
        }

        for i in range(len(emotions)):
            emo = emotions[i]
            speech_length = get_record_length(os.path.join(speech_folder, record_list[i]))
            emotion_durations[emo] = emotion_durations[emo] + speech_length

        with open(emo_record_path, 'a') as csvfile:
            emo_record_writer = csv.writer(csvfile)
            timestamp = record_name.split(".")[0] + "." + record_name.split(".")[1]
            fields = (timestamp, str(emotions), emotion_durations["Sad"], emotion_durations["Neutral"],
                      emotion_durations["Angry"], emotion_durations["Happy"])
            emo_record_writer.writerow(fields)
    else:
        print("Silent 20 seconds")

    shutil.rmtree(speech_folder)

generate_report(emo_record_path)