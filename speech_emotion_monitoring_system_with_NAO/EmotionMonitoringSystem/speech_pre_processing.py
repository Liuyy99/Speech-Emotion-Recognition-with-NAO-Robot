#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Pre-process speech signal.
"""
import wave
import os

import numpy as np
import speech_recognition as sr
from pydub import AudioSegment

SILENCE_THRESHOLD = -50.0
CHUNK_SIZE = 20
TIME_THRESHOLD = 400


def get_record_length(filename):
    sound = AudioSegment.from_file(filename, format="wav")
    return len(sound)


def read_file(filename):
    sound = AudioSegment.from_file(filename, format="wav")
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    speech_time = np.arange(0, wav_length) * (1.0/framerate)
    file.close()
    return sound, wavedata, speech_time, framerate


def is_silence(sound, silence_threshold=SILENCE_THRESHOLD):
    if sound.dBFS < silence_threshold:
        return True
    else:
        return False


def detect_sound_segment(sound, chunk_size=CHUNK_SIZE):
    '''
    sound: pydub.AudioSegment
    silence_threshold: in dB (the loudest is 0; -50 is 1/100000 of the full volumn)
    chunk_size: in ms, must be larger than 0

    iterate over chunks until finding the first one with sound
    '''
    # a list of (start, end) pair; if the whole record is silence, the list will be empty
    sound_segment = []
    trim_start_ms = 0  # ms

    print("sound length: ", len(sound))

    while trim_start_ms < len(sound):
        start = trim_start_ms
        end = trim_start_ms + chunk_size
        if is_silence(sound[start:end]):
            trim_start_ms = end
        else:
            pair = (start, end)
            sound_segment.append(pair)
            trim_start_ms = end

    # merge sound segment
    merged_sound_segment = []
    start_index, end_index = sound_segment[0]
    for i in range(1, len(sound_segment)):
        this_start, this_end = sound_segment[i]
        if this_start == end_index:
            end_index = this_end
        else:
            merged_sound_segment.append((start_index, end_index))
            start_index = this_start
            end_index = this_end

    # print("merged sound segment:", merged_sound_segment)

    return merged_sound_segment


def split_sound_into_segements(sound, segments_start_end):
    segments = []
    for start, end in segments_start_end:
        if end - start < TIME_THRESHOLD:
            continue
        segment = sound[start:end]
        segments.append(segment)
    return segments


def check_speech_text(speech_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(speech_file) as source:
        audio_listened = recognizer.listen(source)
    try:
        rec_text = recognizer.recognize_google(audio_listened)
        print("speech recognition text:", rec_text)
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    return True


def pre_process_record(speech_path):
    # Step 1: read record
    sound, wavedata, speech_time, framerate = read_file(speech_path)

    sound_folder = None
    num_segments = 0
    # Step 2: split record by silence, remove silence at the beginning and end
    if not is_silence(sound):
        merged_sound_segment = detect_sound_segment(sound)
        segments = split_sound_into_segements(sound, merged_sound_segment)

        # Step 3: save segments into .wav file
        num_segments = len(segments)
        segment_root_folder = '../SpeechRecording'
        sound_name = speech_path.split("/")[-1].split(".")[0]
        sound_folder = os.path.join(segment_root_folder, sound_name)
        os.mkdir(sound_folder)

        for i in range(len(segments)):
            segment = segments[i]
            segment_name = "segment" + str(i) + ".wav"
            segment_path = os.path.join(sound_folder, segment_name)
            segment.export(segment_path, format="wav")

    return sound_folder, num_segments
