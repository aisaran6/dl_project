from Model import CleanUNet
from Model import network_config
import torch
from Model import signal2pytorch
import numpy as np
import math


def getprediction(noisy_audio_test):
    model = CleanUNet(**network_config).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load("model_param.pth"))
    # noisy_audio_test, ntsamplerate = librosa.load("p232_001.wav", mono=False, sr=None)
    # print(noisy_audio_test.shape)
    # print(noisy_audio_test)
    # print()
    noisy_audio_norm_test = noisy_audio_test/np.abs(noisy_audio_test.max())
    noisy_audio_norm_test_q=signal2pytorch(noisy_audio_norm_test).to(device)
    predictions=model(noisy_audio_norm_test_q)
    predictions=predictions.cpu().detach()
    predictions=np.array(predictions)
    xrek=predictions[:,0,:]
    # print(xrek[0].shape)
    # print()
    # print(xrek[0])
    return xrek
def signalPower(x):
    # print(x)
    return np.average(x**2)
# def SNR(signal, noise):
#     powS = signalPower(signal)
#     powN = signalPower(noise)
#     return 10*math.log10(math.abs(powS-powN)/powN)
def SNRsystem(inputSig, outputSig):
    noise = outputSig-inputSig

    powS = signalPower(outputSig)
    powN = signalPower(noise)
    return 10*math.log10(abs((powS-powN))/powN)

def calculate_snr(clean_audio, noisy_audio):
    method2 = SNRsystem(clean_audio,noisy_audio)
    # print("Result Method 2: {} dB".format(method2))
    return method2