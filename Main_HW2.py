import basic_audio_proc
import numpy as np
import warnings
import pickle
import matplotlib.pyplot as plt

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
playplot_audio = False
plot_audio = False
play_audio = False


amps = np.array([1000, 600,1000])
freqz = np.array([3500, 800, 1200])
fs = 44100

sinSignal = np.int16(basic_audio_proc.generateSinSignal(amps,freqz,60,fs))
#basic_audio_proc.play_audio(sinSignal, fs)

spect = np.fft.fft(sinSignal,1024)
freq = np.fft.fftfreq(spect.shape[-1])
print(spect.shape)
print(freq.shape)
plt.plot(freq,spect.real,freq,spect.imag)
plt.show()
