import os
# export PYTHONPATH=${PYTHONPATH}:/Users/yiyangliu/Desktop/pynaoqi-python2.7-2.8.6.23-mac64/lib/python2.7/site-packages
# export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Users/yiyangliu/Desktop/pynaoqi-python2.7-2.8.6.23-mac64/lib
# export QI_SDK_PREFIX=/Users/yiyangliu/Desktop/pynaoqi-python2.7-2.8.6.23-mac64
import sys
import shutil
import time

import csv
import logging
logging.getLogger().setLevel("ERROR")

from naoqi import ALProxy
from speech_pre_processing import pre_process_record, get_record_length
from feature_extraction import get_segments_num
from emotion_prediction import predict
from report_generation import generate_report

# ONE_DAY = 86400  # seconds

ONE_DAY = 40

def establish_connection(robot_ip, robot_port):
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
    recorder = ALProxy("ALAudioRecorder", robot_ip, robot_port)
    return tts, recorder


def live_monitor_emotion(robot_ip, robot_port, tts, recorder, speech_root_path, emo_record_root, loop_length):
    recorder.stopMicrophonesRecording()
    print 'emotion monitoring system started...'
    tts.say("hi I am your nao robot.")
    tts.say("start emotion recognition system...")

    while True:
        day_start = time.time()
        day_end = day_start + ONE_DAY
        emo_record_path = os.path.join(emo_record_root, str(day_start) + ".csv")
        monitor_one_day_emotion(day_end, recorder, speech_root_path, emo_record_path, loop_length)
        generate_report(emo_record_path)


def monitor_one_day_emotion(day_end, recorder, nao_speech_root_path, emo_record_path, loop_length):
    if not os.path.exists(emo_record_path):
        with open(emo_record_path, 'a') as csvfile:
            emo_record_writer = csv.writer(csvfile)
            emo_record_writer.writerow(('Timestamp', 'Emotions', 'Sad', 'Neutral', 'Angry', 'Happy'))

    while time.time() < day_end:
        print '----------------------------------------------------------------------'
        print 'record start'
        recorder.stopMicrophonesRecording()
        current_timestamp = time.time()
        readable_timestamp = time.ctime(current_timestamp)
        record_name = str(current_timestamp) + ".wav"
        record_path = os.path.join(nao_speech_root_path, record_name)
        recorder.startMicrophonesRecording(record_path, "wav", 16000, (0, 0, 1, 0))
        time.sleep(loop_length)
        recorder.stopMicrophonesRecording()
        print 'record over'

        audio_transfer_start = time.time()
        nao_server = "nao@" + robot_ip
        record_path_nao = nao_server + ":" + record_path
        record_path_local = os.path.join("../SpeechRecording", record_name)
        command = "scp " + record_path_nao + " " + record_path_local
        print command
        os.system(command)
        audio_transfer_end = time.time()
        print("Audio Transfer Time: ", audio_transfer_end - audio_transfer_start)

        print 'start emotion recognition'
        # Step 1: Speech Pre-processing
        speech_folder, num_speech = pre_process_record(record_path_local)

        # Step 2: Emotion Recognition
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

            shutil.rmtree(speech_folder)
        else:
            print("Silent 20 seconds")

        print '---------Total time for data processing of 20 seconds record----------'
        print time.time() - audio_transfer_start


if __name__ == '__main__':
    robot_ip = ""
    robot_port = 9559
    if len(sys.argv) == 1:
        print ("Robot IP not provided. Usage: python main.py <robot-ip>")
        exit()

    robot_ip = sys.argv[1]
    tts, recorder = establish_connection(robot_ip, robot_port)
    nao_speech_root_path = '/home/nao/yiyang/speech_recording'
    emo_record_root = '../SERRecord'
    loop_length = 20  # seconds
    print ("Start Live Emotion Monitoring. Exit by keyboard interrupt.")
    live_monitor_emotion(robot_ip, robot_port, tts, recorder, nao_speech_root_path, emo_record_root, loop_length)
