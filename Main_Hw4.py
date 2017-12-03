import encframework as enc
import decframework as dec
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import basic_audio_proc as bap



########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
plot_filter = False
dumpHuffman = False
plotSpectra = False

quantized_audio, norm_audio, fs, dump_fname = enc.read_and_quantize('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel, n_bits=8)

# decompose into 4 subbands
lpAudio, bp1Audio, bp2Audio, hpAudio = enc.applyAnalysisFilterBank(quantized_audio,fs,plotFilter=plot_filter,plotAudio=plot_audio,playAudio=play_audio)
# downsampling
sFactor = 4
lpAudioDs = bap.downsample(lpAudio, N=sFactor)
bp1AudioDs = bap.downsample(bp1Audio, N=sFactor)
bp2AudioDs = bap.downsample(bp2Audio, N=sFactor)
hpAudioDs = bap.downsample(hpAudio, N=sFactor)


# Here is where processing (Psychoacoustic model, Hufmann Coding etc can take place)


# Synthesis of signal components

# upsampling

lpAudioUs = bap.upsample(lpAudioDs, N=sFactor)
bp1AudioUs = bap.upsample(bp1AudioDs, N=sFactor)
bp2AudioUs = bap.upsample(bp2AudioDs, N=sFactor)
hpAudioUs = bap.upsample(hpAudioDs, N=sFactor)

# reconstruct original signal

reconstructedAudio = enc.applySynthesisFilterBank(lpAudioUs, bp1AudioUs, bp2AudioUs, hpAudioUs, fs)

f1 = plt.figure(1)
w, H = signal.freqz(quantized_audio)
plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
plt.xlabel('Normalized Frequency')
plt.ylabel('dB')
plt.title('Quantized Audio Spectrum')

f2 = plt.figure(2)
w, H = signal.freqz(reconstructedAudio)
plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
plt.xlabel('Normalized Frequency')
plt.ylabel('dB')
plt.title('Reconstructed Audio Spectrum')

f3= plt.figure(3)
plt.plot(quantized_audio)
plt.title('Original Audio Signal')

f4 = plt.figure(4)
plt.plot(reconstructedAudio)
plt.title('Reconstructed Audio Signal')
plt.show()

bap.play_audio(quantized_audio, fs)
bap.play_audio(reconstructedAudio, fs)


if play_audio:
    print("playing lowpass component")
    print("fs = ", fs, " Hz")
    bap.play_audio(lpAudio, fs)
    print("fs = ", fs/sFactor, " Hz")
    bap.play_audio(lpAudioDs, fs/sFactor)
    print("playing bandpass 1 component")
    print("fs = ", fs, " Hz")
    bap.play_audio(bp1Audio, fs)
    print("fs = ", fs/sFactor, " Hz")
    bap.play_audio(bp1AudioDs, fs/sFactor)
    print("playing bandpass 2 component")
    print("fs = ", fs, " Hz")
    bap.play_audio(bp2Audio, fs)
    print("fs = ", fs / sFactor, " Hz")
    bap.play_audio(bp2AudioDs, fs/sFactor)
    print("playing highpass component")
    print("fs = ", fs, " Hz")
    bap.play_audio(hpAudio, fs)
    print("fs = ", fs / sFactor, " Hz")
    bap.play_audio(hpAudioDs, fs/sFactor)


if plotSpectra:
    f1 = plt.figure(1)
    w, H = signal.freqz(quantized_audio)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('dB')
    plt.title('Quantized Audio Spectrum')
    #plt.show()

    f2 = plt.figure(2)
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