import basic_audio_proc
import numpy as np
<<<<<<< HEAD
import warnings
import pickle
import matplotlib.pyplot as plt
=======
import matplotlib.pyplot as plt
import scipy.signal as sig
>>>>>>> master

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
<<<<<<< HEAD
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
=======
plot_audio = True
play_audio = False


length_audio = 5
amps = np.array([1000, 1000])
freqz = np.array([200, 600])
fs = 44100
n_brkbands = 48

sinSignal = np.int16(basic_audio_proc.generateSinSignal(amps,freqz,length_audio,fs))

if play_audio:
    basic_audio_proc.play_audio(sinSignal, fs)

f_stft_hz, t_stft, Zxx_stft = sig.stft(sinSignal, fs, nperseg=2048, nfft=2048)
power_stft = np.square(np.abs(Zxx_stft))

if plot_audio:
    plt.pcolormesh(t_stft, f_stft_hz, power_stft, vmin=0, vmax=np.amax(power_stft))
    plt.title('STFT Power')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

f_stft_brk = basic_audio_proc.hz2bark(f_stft_hz)
W = basic_audio_proc.mapping2barkmat(fs, f_stft_brk, n_brkbands)
power_stft = np.square(np.abs(Zxx_stft))
power_in_brk_band = np.dot(W, power_stft)
spl_in_brk_band = 10 * np.log10(power_in_brk_band)

brk_bandwise_axis = np.linspace(0, (n_brkbands-1)/2.0, n_brkbands)
hz_bandwise_axis = basic_audio_proc.bark2hz(brk_bandwise_axis)

if plot_audio:
    plt.pcolormesh(t_stft, brk_bandwise_axis, power_in_brk_band, vmin=0, vmax=np.amax(power_in_brk_band))
    plt.title('STFT Power')
    plt.ylabel('Frequency [Bark]')
    plt.xlabel('Time [sec]')
    plt.show()

spreadingfunc_brk = basic_audio_proc.calc_spreadingfunc_brk(1, spl_in_brk_band, plot=plot_audio)
maskingthresh = basic_audio_proc.nonlinear_superposition(spreadingfunc_brk[100,:,:], alpha=0.3)

thresh_quiet = 3.64 * (hz_bandwise_axis/1000.) **(-0.8) - 6.5*np.exp( -0.6 * (hz_bandwise_axis/1000. - 3.3) ** 2.) + 1e-3*((hz_bandwise_axis/1000.) ** 4.)
thresh_quiet = np.clip(thresh_quiet, -20, 60)
thresh_quiet = thresh_quiet - 60 # convert from SPL to dB Full Scale (digital)

overall_thresh = np.maximum(maskingthresh, thresh_quiet)

plt.plot(hz_bandwise_axis, thresh_quiet)
plt.show()

plt.plot(brk_bandwise_axis, maskingthresh)
plt.xlabel('frequency [Bark]')
plt.ylabel('Masking Threshold [dB]')
plt.show()

plt.plot(hz_bandwise_axis, maskingthresh, 'b', alpha=0.5, lw=1, label='Signal Masking')
plt.plot(hz_bandwise_axis, overall_thresh, 'r', alpha=0.5, lw=1, label='Signal Masking and Threshold in Quiet')
plt.xlabel('frequency [Hz]')
plt.ylabel('Masking Threshold [dB]')
plt.legend()
plt.show();
test = True
>>>>>>> master
