import basic_audio_proc
import numpy as np
import warnings
import pickle
#import matplotlib.pyplot as plt

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
playplot_audio = False
plot_audio = False
play_audio = False


amps = np.array([1000, 600])
freqz = np.array([200, 600])
fs = 44100

sinSignal = np.int16(basic_audio_proc.generateSinSignal(amps,freqz,5,fs))
basic_audio_proc.play_audio(sinSignal, fs)
