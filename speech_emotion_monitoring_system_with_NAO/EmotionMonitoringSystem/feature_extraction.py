import wave
import numpy as np
import python_speech_features as ps
import os
import cPickle

eps = 1e-5
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def getlogspec(signal,samplerate=16000,winlen=0.02,winstep=0.01,
               nfilt=26,nfft=399,lowfreq=0,highfreq=None,preemph=0.97,
               winfunc=lambda x:np.ones((x,))):
    highfreq = highfreq or samplerate/2
    signal = ps.sigproc.preemphasis(signal,preemph)
    frames = ps.sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = ps.sigproc.logpowspec(frames,nfft)
    return pspec

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    speech_time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, speech_time, framerate

def normalization(data, mean, std, eps):
    data = (data - mean) / (std + eps)
    return data

def load_data():
    f = open('./zscore_Casia_40.pkl','rb')
    mean1, std1, mean2, std2, mean3, std3 = cPickle.load(f)
    return mean1, std1, mean2, std2, mean3, std3

def extract_speech_feature(rootdir, num_of_record, num_of_segments):
    eps = 1e-5
    speech_num = num_of_record
    speech_segment_num = num_of_segments
    filter_num = 40
    pernums = np.arange(speech_num)
    
    mean1, std1, mean2, std2, mean3, std3 = load_data()

    speech_data = np.empty((speech_segment_num, 300, filter_num, 3), dtype= np.float32)
    record_list = []

    speech_num = 0
    speech_segment_num = 0

    for speech in os.listdir(rootdir):
        if speech.startswith('.'):
            continue

        speech_path = os.path.join(rootdir, speech)
        data, speech_time, rate = read_file(speech_path)
        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)
        speech_time = mel_spec.shape[0]
        record_list.append(speech)

        if (speech_time <= 300):
            pernums[speech_num] = 1
            part = mel_spec
            delta11 = delta1
            delta21 = delta2
            part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant', constant_values=0)
            delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant', constant_values=0)
            delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant', constant_values=0)
            speech_data[speech_segment_num, :, :, 0] = normalization(part, mean1, std1, eps)
            speech_data[speech_segment_num, :, :, 1] = normalization(delta11, mean2, std2, eps)
            speech_data[speech_segment_num, :, :, 2] = normalization(delta21, mean3, std3, eps)
            speech_segment_num = speech_segment_num + 1
            speech_num = speech_num + 1
        else:
            frames = divmod(speech_time - 300, 100)[0] + 1
            pernums[speech_num] = frames
            speech_num = speech_num + 1
            for i in range(frames):
                begin = 100 * i
                end = begin + 300
                part = mel_spec[begin:end, :]
                delta11 = delta1[begin:end, :]
                delta21 = delta2[begin:end, :]
                speech_data[speech_segment_num, :, :, 0] = normalization(part, mean1, std1, eps)
                speech_data[speech_segment_num, :, :, 1] = normalization(delta11, mean2, std2, eps)
                speech_data[speech_segment_num, :, :, 2] = normalization(delta21, mean3, std3, eps)
                speech_segment_num = speech_segment_num + 1

    return record_list, speech_data, pernums

def get_segments_num(rootdir):
    filter_num = 40
    speech_segment_num = 0

    for speech in os.listdir(rootdir):
        if speech.startswith('.'):
            continue
        speech_path = os.path.join(rootdir, speech)
        data, speech_time, rate = read_file(speech_path)
        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
        speech_time = mel_spec.shape[0]

        if (speech_time <= 300):
            speech_segment_num = speech_segment_num + 1
        else:
            frames = divmod(speech_time - 300, 100)[0] + 1
            speech_segment_num = speech_segment_num + frames

    return speech_segment_num