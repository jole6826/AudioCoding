import encframework as enc
import decframework as dec
import matplotlib.pyplot as plt
import basic_audio_proc as bap
import plotting



########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = True
plot_filter = True
plotSpectra = True
play_audio = True
dumpHuffman = False
n_bands = 4
sFactor = 4

quantized_audio, raw_audio, norm_audio, fs, dump_fname = enc.read_and_quantize('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel, n_bits=16)

# decompose into 4 subbands
audio_in_bands, filterbank = enc.applyAnalysisFilterBank(raw_audio, n_bands, fs)
if plot_filter:
    plotting.plot_filterbank(filterbank)

# downsampling
audio_in_bands_ds = [bap.downsample(band, N=sFactor) for band in audio_in_bands]

# Here is where processing (Psychoacoustic model, Hufmann Coding etc can take place)

# Synthesis of signal components

# upsampling
audio_in_bands_us = [bap.upsample(band, N=sFactor) for band in audio_in_bands_ds]

# reconstruct original signal

reconstructed_audio = dec.applySynthesisFilterBank(audio_in_bands, filterbank)


if play_audio:
    print 'playing lowpass component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_in_bands[0], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_in_bands_ds[0], fs/sFactor)
    
    print 'playing bandpass 1 component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_in_bands[1], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_in_bands_ds[1], fs/sFactor)
    
    print 'playing bandpass 2 component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_in_bands[2], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_in_bands_ds[2], fs/sFactor)
    
    print 'playing highpass component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_in_bands[3], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_in_bands_ds[3], fs/sFactor)
    
    print 'Play original audio'
    bap.play_audio(raw_audio, fs)
    print 'Play reconstructed audio'
    bap.play_audio(reconstructed_audio, fs)


if plotSpectra:
    f1 = plotting.plot_spectrum(raw_audio, 'Original Audio Spectrum')
    f2 = plotting.plot_spectrum(audio_in_bands, 'Spectrum of the 4 subband components', 
                                ['lowpass', 'bp 1', 'bp 2', 'highpass'], 'lower center')
    
    f3 = plotting.plot_spectrum(reconstructed_audio, 'Reconstructed Audio Spectrum')
    plt.show()