import encframework as enc
import decframework as dec
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np



########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
plot_filter = True
dumpHuffman = True
plotSpectra = True

quantized_audio, norm_audio, fs, dump_fname = enc.read_and_quantize('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel, n_bits=8)

lpAudio, bp1Audio, bp2Audio, hpAudio = enc.applyFilterBank(quantized_audio,fs,plotFilter=plot_filter,plotAudio=plot_audio,playAudio=play_audio)


if plotSpectra:
    g = plt.figure(1)
    w, H = signal.freqz(quantized_audio)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('dB')
    plt.title('Quantized Audio Spectrum')
    #plt.show()

    f = plt.figure(2)
    w, H = signal.freqz(lpAudio)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    # plt.show()

    w, H = signal.freqz(bp1Audio)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    # plt.show()

    w, H = signal.freqz(bp2Audio)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    # plt.show()

    w, H = signal.freqz(hpAudio)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    plt.title('Spectrum of the 4 components')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('dB')
    plt.legend(['lowpass', 'bp 1', 'bp 2', 'highpass'])
    plt.show()