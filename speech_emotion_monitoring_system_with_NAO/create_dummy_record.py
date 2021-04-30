import time
import os
import csv
import random

emo_record_path = './dummy_SER_emotion_record.csv'

nao_record_length = 20  # seconds

if not os.path.exists(emo_record_path):
    with open(emo_record_path, 'a') as csvfile:
        emo_record_writer = csv.writer(csvfile)
        emo_record_writer.writerow(('Timestamp', 'Emotions', 'Sad', 'Neutral', 'Angry', 'Happy'))

start_time = time.time() - 36000
record_length = 20
record_length_ms = 20000
to_add = 0 # second
total_time = 86400 # 24 hours --> 86400 seconds
ran_sad_range = 0
emotions = ["Sad", "Neutral", "Angry", "Happy"]

# 20000
while (to_add <= total_time):
    sad_duration = 0
    neutral_duration = 0
    angry_duration = 0
    happy_duration = 0
    silence_duration = 0
    if to_add < 10000:
        sad_duration = random.randrange(0, 2000, 20)
        neutral_duration = random.randrange(0, 4000, 20)
        angry_duration = random.randrange(0, 2000, 20)
        happy_duration = random.randrange(0, 2000, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration
    elif to_add < 20000:
        sad_duration = random.randrange(0, 3000, 20)
        neutral_duration = random.randrange(0, 4000, 20)
        angry_duration = random.randrange(0, 1000, 20)
        happy_duration = random.randrange(0, 1000, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration
    elif to_add < 30000:
        sad_duration = random.randrange(0, 5000, 20)
        neutral_duration = random.randrange(0, 1000, 20)
        angry_duration = random.randrange(0, 500, 20)
        happy_duration = random.randrange(0, 500, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration
    elif to_add < 40000:
        sad_duration = random.randrange(0, 2000, 20)
        neutral_duration = random.randrange(0, 3000, 20)
        angry_duration = random.randrange(0, 1000, 20)
        happy_duration = random.randrange(0, 1000, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration
    elif to_add < 65000:
        sad_duration = random.randrange(0, 100, 20)
        neutral_duration = random.randrange(0, 100, 20)
        angry_duration = random.randrange(0, 100, 20)
        happy_duration = random.randrange(0, 100, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration
    elif to_add < 75000:
        sad_duration = random.randrange(0, 1000, 20)
        neutral_duration = random.randrange(0, 2000, 20)
        angry_duration = random.randrange(0, 4000, 20)
        happy_duration = random.randrange(0, 1000, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration
    else:
        sad_duration = random.randrange(0, 1000, 20)
        neutral_duration = random.randrange(0, 4000, 20)
        angry_duration = random.randrange(0, 1000, 20)
        happy_duration = random.randrange(0, 1000, 20)
        silence_duration = record_length_ms - sad_duration - neutral_duration - angry_duration - happy_duration

    emotions = ["dummy"]
    record_timestamp = start_time + to_add
    with open(emo_record_path, 'a') as csvfile:
        emo_record_writer = csv.writer(csvfile)
        fields = (str(record_timestamp), str(emotions),
                  sad_duration, neutral_duration, angry_duration, happy_duration)
        emo_record_writer.writerow(fields)
    process_time = random.uniform(1.5, 2.5)
    to_add = to_add + process_time + record_length