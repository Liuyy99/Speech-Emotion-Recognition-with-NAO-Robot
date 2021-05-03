# Speech-Emotion-Recognition-with-NAO-Robot
Cross-corpus Language-independent Speech Emotion Recognition (SER) is a challenging task due to the complexity of 
emotion and the effects of emotional irrelevant factors, such as acoustic environment and language differences. 
Most of the current SER systems focus on classifying emotions in a single monolingual corpus. In real-life applications, 
these systems show poor ability to handle inputs in different environments and languages. 

This project aims to propose a cross-corpus language-independent SER system and build an emotion monitoring system 
with NAO robot. We proposed a 3-D Attention-based CRNN model (ACRNN) with normalization step to eliminate the effects 
of corpus and language specific factors. Corpora in different languages were combined for model training to get better 
generalizability. The created model was then incorporated into a 24/7 real-time speech emotion monitoring system with 
the NAO robot. 

There are two main components in this project: **speech_emotion_recognition** and 
**speech_emotion_monitoring_system_with_NAO**. 
For more details, please refer to the README file of each component.
