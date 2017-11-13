import basic_audio_proc
import numpy as np
import warnings
import pickle
import matplotlib.pyplot as plt
import scipy.signal as sig

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
playplot_audio = False
plot_audio = False
play_audio = False


amps = np.array([1000, 1000])
freqz = np.array([200, 600])
fs = 44100

sinSignal = np.int16(basic_audio_proc.generateSinSignal(amps,freqz,5,fs))

if plot_audio:
    plt.plot(sinSignal)
    plt.show()
if play_audio:
    basic_audio_proc.play_audio(sinSignal, fs)
    
    
f_stft_hz, t_stft, Zxx_stft = sig.stft(sinSignal, fs, nperseg=2048, nfft=2048)

if plot_audio:
    plt.pcolormesh(t_stft, f_stft_hz, np.abs(Zxx_stft), vmin=0, vmax=amps[1])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

f_stft_brk = basic_audio_proc.hz2bark(f_stft_hz)
W = basic_audio_proc.mapping2barkmat(fs, f_stft_brk, 0.5)
power_stft = np.square(np.abs(Zxx_stft))
power_in_brk_band = np.dot(W, power_stft)
spl_in_brk_band = 10 * np.log10(power_in_brk_band)

spread = basic_audio_proc.spreadingfunctionmat(fs/2, 1, spl_in_brk_band)
test = True
