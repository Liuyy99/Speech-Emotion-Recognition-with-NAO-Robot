import pandas as pd
import time
import os
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt

ONE_HOUR = 3600  # SECONDS


def date_parser(timestamps):
    return [time.ctime(float(x)) for x in timestamps]


def get_dayhour(datehour):
    datehour = datehour[1:-1]
    date = datehour.split(",")[0]
    hour = datehour.split(",")[1].strip()
    month = date.split("-")[1]
    day = date.split("-")[-1]
    dayhour = hour
    return dayhour


def generate_report(emo_record):
    record_data = pd.read_csv(emo_record, date_parser=date_parser, parse_dates=['Timestamp'])
    record_data = record_data.rename(columns={
        "Timestamp": "datetime",
        "Sad": "sad_length",
        "Neutral": "neutral_length",
        "Angry": "angry_length",
        "Happy": "happy_length"})

    record_data["datetime"] = pd.to_datetime(record_data["datetime"])
    start_time = record_data["datetime"].min()
    end_time = record_data["datetime"].max()
    plot_header = "FROM " + str(start_time) + " TO " + str(end_time)

    emotion_names = ["sad", "happy", "neutral", "angry"]

    for emo in emotion_names:
        emo_length = emo + "_length"
        emo_percent = emo + "_percent"
        record_data[emo_length] = record_data[emo_length].apply(lambda x: float(x) / 1000)
        record_data[emo_percent] = record_data[emo_length].apply(lambda x: (x / ONE_HOUR) * 100)

    record_data["date"] = record_data["datetime"].dt.date
    record_data["hour"] = record_data["datetime"].dt.hour
    record_data = record_data.set_index("datetime")

    sad_length_of_each_hour = record_data.groupby(["date", "hour"])["sad_length"].sum()
    sad_percent_of_each_hour = record_data.groupby(["date", "hour"])["sad_percent"].sum()
    emotion_length_of_each_hour = record_data.groupby(["date", "hour"])["sad_length", "neutral_length", "angry_length",
                                                                        "happy_length"].sum()
    emotion_percent_of_each_hour = record_data.groupby(["date", "hour"])["sad_percent", "neutral_percent", "angry_percent",
                                                                        "happy_percent"].sum()


    folder_name = str(start_time) + "to" + str(end_time)
    folder_path = os.path.join("../ReportFigure", folder_name)
    os.mkdir(folder_path)

    # Plot 1: bar chart, the length of sad emotion (in seconds) in every one hour
    fig, (ax1) = plt.subplots(1, figsize=(12, 6))
    sad_length_of_each_hour.plot(kind='bar', rot=0, ax=ax1)
    ax1.set_title(plot_header + "\nSad Emotion Appearance")
    ax1.set_xlabel("Hour of the day")
    ax1.set_ylabel("Sad emotion detected in one hour (in seconds)")
    new_x_labels = [get_dayhour(item.get_text()) for item in ax1.get_xticklabels()]
    ax1.set_xticklabels(new_x_labels)
    # plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.savefig(os.path.join(folder_path, "plot1.png"))

    # Plot 2: bar chart, the length of each emotion (in seconds) in every one hour
    fig, (ax2) = plt.subplots(1, figsize=(12, 6))
    emotion_length_of_each_hour.plot(kind='bar', rot=0, ax=ax2)
    ax2.set_title(plot_header + "\n Emotion Appearance (second)")
    ax2.set_xlabel("Hour of the day")
    ax2.set_ylabel("Emotions detected in every one hour (in seconds)")
    new_x_labels = [get_dayhour(item.get_text()) for item in ax2.get_xticklabels()]
    ax2.set_xticklabels(new_x_labels)
    # plt.setp(ax2.get_xticklabels(), rotation=45)
    plt.savefig(os.path.join(folder_path, "plot2.png"))

    # Plot 3: line chart, the length of each emotion (percentage) in every one hour
    fig, (ax3) = plt.subplots(1, figsize=(12, 6))
    emotion_percent_of_each_hour.plot(kind='line', rot=0, ax=ax3)
    ax3.set_title(plot_header + "\n Emotion Changes (percentage)")
    ax3.set_xlabel("Hour of the day")
    ax3.set_ylabel("Emotion changes in 24 hours (percentage)")
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.savefig(os.path.join(folder_path, "plot3.png"))
