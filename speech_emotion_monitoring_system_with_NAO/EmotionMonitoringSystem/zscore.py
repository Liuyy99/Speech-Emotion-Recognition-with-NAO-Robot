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
    file = wave.open(filename,'r')    
    params = file.getparams()
    # nchannels: number of audio channels, 1 for mono, 2 for stereo
    # sampwidth: sample width in byte
    # framerate: sampling frequency
    # wav_length: number of audio frames
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    # wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate
        
def read_CASIA():
    
    train_num = 810
    filter_num = 40
    rootdir = '/Users/yiyangliu/Desktop/Unseen_dataset_for_testing/Formatted_casia_4_emotions'
    traindata1 = np.empty((train_num*300,filter_num),dtype=np.float32)
    traindata2 = np.empty((train_num*300,filter_num),dtype=np.float32)
    traindata3 = np.empty((train_num*300,filter_num),dtype=np.float32)
    train_num = 0

    all_emotions = ["Angry", "Happy", "Sad", "Neutral"]

    for emotion_name in os.listdir(rootdir):
        # emotion = ''
        # if (emotion_name[0] == 'A'):  # angry
        #     emotion = 'ang'
        # if (emotion_name[0] == 'H'):  # happy
        #     emotion = 'hap'
        # if (emotion_name[0] == 'S'):  # sad
        #     emotion = 'sad'
        # if (emotion_name[0] == 'N'):  # neu
        #     emotion = 'neu'

        if emotion_name not in all_emotions:
            continue

        sub_dir = os.path.join(rootdir, emotion_name)
        for record in os.listdir(sub_dir):
            if record.startswith('.'):
                continue

            record_path = os.path.join(sub_dir, record)

            data, time, rate = read_file(record_path)
            mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
            delta1 = ps.delta(mel_spec, 2)
            delta2 = ps.delta(delta1, 2)
            time = mel_spec.shape[0]

            if (time <= 300):
                part = mel_spec
                delta11 = delta1
                delta21 = delta2
                part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant', constant_values=0)
                delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant', constant_values=0)
                delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant', constant_values=0)
                traindata1[train_num * 300:(train_num + 1) * 300] = part
                traindata2[train_num * 300:(train_num + 1) * 300] = delta11
                traindata3[train_num * 300:(train_num + 1) * 300] = delta21

                train_num = train_num + 1
            else:
                frames = divmod(time - 300, 100)[0] + 1
                for i in range(frames):
                    begin = 100 * i
                    end = begin + 300
                    part = mel_spec[begin:end, :]
                    delta11 = delta1[begin:end, :]
                    delta21 = delta2[begin:end, :]
                    traindata1[train_num * 300:(train_num + 1) * 300] = part
                    traindata2[train_num * 300:(train_num + 1) * 300] = delta11
                    traindata3[train_num * 300:(train_num + 1) * 300] = delta21
                    train_num = train_num + 1


    print("train_num", train_num)
    mean1 = np.mean(traindata1,axis=0)
    std1 = np.std(traindata1,axis=0)
    mean2 = np.mean(traindata2,axis=0)
    std2 = np.std(traindata2,axis=0)
    mean3 = np.mean(traindata3,axis=0)
    std3 = np.std(traindata3,axis=0)
    output = './zscore_Casia_'+str(filter_num)+'.pkl'
    f=open(output,'wb')
    cPickle.dump((mean1,std1,mean2,std2,mean3,std3),f)
    f.close()
    return

if __name__=='__main__':
    read_CASIA()
    #print "test_num:", test_num
    #print "train_num:", train_num
#    n = wgn(x, 6)
#    xn = x+n
