# Speech-Emotion-Recognition-with-NAO-Robot
Cross-corpus Language-independent Speech Emotion Recognition (SER) is a challenging task due to the complexity of 
emotion and the effects of emotional irrelevant factors, such as acoustic environment and language differences. 
Most of the current SER systems focus on classifying emotions in a single monolingual corpus. In real-life applications, 
these systems show poor ability to handle inputs in different environments and languages. 

This project aims to propose a cross-corpus language-independent SER system and build an emotion monitoring system 
with NAO robot. A 3-D Attention-based CRNN model (ACRNN) was proposed with a normalization step to eliminate the effects 
of corpus and language specific factors. Corpora in different languages were combined for model training to get better 
generalizability. The created model was then incorporated into a 24/7 real-time speech emotion monitoring system with 
the NAO robot. 

There are two main components in this project: 
* **speech_emotion_recognition**
* **speech_emotion_monitoring_system_with_NAO**

## speech_emotion_recognition
This component contains implementation for feature extraction, normalization, model training and model testing. 

The code has been tested on Linux and on MacOS. 

#### Setup
Python2.7, Tensorflow1.3.0, numpy, python-speech-features, sklearn, scipy

#### Dataset
* Combined dataset (resampled and combined German Emo-db, SAVEE, RAVDESS, URDU datasets)

  4 languages -- German, British English, American English, Urdu; for training and validation

* IEMOCAP
  
  English; for cross-corpus testing

* CASIA
  
  Chinese; for cross-corpus and cross-language testing

#### Build model
* Put combined dataset into folder as *speech_emotion_recognition/combined_4_dataset*
* Calculate neutral mean and standard deviation
```
cd speech_emotion_recognition
cd src
python zscore_mean_std.py
```
* Extract 3d mel-spectrogram and do z-score normalization
```
python extract_normalize_mel.py
```
* Build and train model
```
python build_model.py
```

#### Pre-trained model
Two pre-trained models are available in pre_trained_models folder. 
model_optimize_UA contains model with good 4-emotion recognition performance.
model_optimize_UA_times_sad contains model with good sad recognition performance, as well as
good overall performance. This model is selected as the model for speech emotion monitoring system.

#### Test model
* Put IEMOCAP dataset and formatted CASIA dataset into folder as 
*speech_emotion_recognition/test_data/IEMOCAP_full_release* and *speech_emotion_recognition/test_data/CASIA_formatted*
* Extract features and do normalization
```
cd speech_emotion_recognition
cd test_data
python zscore_IEMOCAP.py
python extract_mel_IEMOCAP.py
python zscore_CASIA.py
python extract_mel_CASIA.py
```
* Set the model to be tested in src/test_by_unseen_datasets.py
```
# Example
folder_path = "../pre_trained_models/model_optimize_UA_times_sad"
models_name = ["model4.ckpt-1881"]
```
* Set the testing dataset in src/test_by_unseen_datasets.py
```
# Example 1
tf.app.flags.DEFINE_string('testdata_path', '../test_data/IEMOCAP.pkl', 'data from IEMOCAP normalized by neutral')
# Example 2
tf.app.flags.DEFINE_string('testdata_path', '../test_data/CASIA.pkl', 'data from CASIA normalized by neutral')
```
* Test model
```
cd speech_emotion_recognition
cd src
python test_by_unseen_datasets.py
```
#### Notice
The reported performance in the dissertation can be regenerated by models in the pre_trained_models folder. 

## speech_emotion_monitoring_system_with_NAO