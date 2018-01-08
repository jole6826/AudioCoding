import encframework as enc
import decframework as dec
import matplotlib.pyplot as plt
import basic_audio_proc as bap
import plotting
import scipy.signal as sig
import numpy as np

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = True
play_audio = False
dump_files = False
n_bands = 4
sFactor = 4
n_brkbands = 48

raw_audio, norm_audio, org_dtype, fs = enc.read_segment('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel)
quantized8_audio = enc.quantize(norm_audio, org_dtype, 8)

# decompose into 4 subbands
audio_bands, filterbank = enc.applyAnalysisFilterBank(raw_audio, n_bands, fs)
if plot_audio:
    f1 = plotting.plot_filterbank(filterbank)
    plt.show()

# downsampling
audio_bands_ds = [bap.downsample(band, N=sFactor) for band in audio_bands]


# Quantization
norm_audio_bands_ds = [bap.normalize(band) for band in audio_bands_ds]
quantized16_audio_bands_ds = [enc.quantize(band, org_dtype, 16) for band in norm_audio_bands_ds]
quantized8_audio_bands_ds = [enc.quantize(band, org_dtype, 8) for band in norm_audio_bands_ds]

# Psychoacoustics
f_stft_hz, t_stft, Zxx_stft = zip(*(sig.stft(band, fs, nperseg=2048, nfft=2048) for band in audio_bands))

if plot_audio:
    f2 = plotting.plot_spectrogram(audio_bands, fs)

f_stft_brk = bap.hz2bark(f_stft_hz[0])
W = bap.mapping2barkmat(fs, f_stft_brk, n_brkbands)
power_stft = [np.square(np.abs(Zxx_band)) for Zxx_band in Zxx_stft]
power_in_brk_band = [np.dot(W, power_stft_band) for power_stft_band in power_stft]
spl_in_brk_band = [10 * np.log10(band) for band in power_in_brk_band]

spreadingfunc_brk = [bap.calc_spreadingfunc_brk(1, band, plot=plot_audio) for band in spl_in_brk_band]
maskingthresh = [bap.nonlinear_superposition(band[100,:,:], alpha=0.3) for band in spreadingfunc_brk]

brk_bandwise_axis = np.linspace(0, (n_brkbands-1)/2.0, n_brkbands)
hz_bandwise_axis = bap.bark2hz(brk_bandwise_axis)
thresh_quiet = 3.64 * (hz_bandwise_axis/1000.) **(-0.8) - 6.5*np.exp( -0.6 * (hz_bandwise_axis/1000. - 3.3) ** 2.) + 1e-3*((hz_bandwise_axis/1000.) ** 4.)
thresh_quiet = np.clip(thresh_quiet, -20, 60)
thresh_quiet = thresh_quiet - 60 # convert from SPL to dB Full Scale (digital)

overall_thresh = [np.maximum(maskingthresh_band, thresh_quiet) for maskingthresh_band in maskingthresh]

if plot_audio:
    plotting.plot_maskingthresh(overall_thresh, hz_bandwise_axis)
    plt.show()

# Huffmann 
cb_bands, cb_tree_bands, data_binstring_bands = zip(*(enc.enc_huffman(quantized_audio) for quantized_audio in quantized8_audio_bands_ds))

if dump_files:
    enc.dump_quantized(quantized8_audio_bands_ds, 'encoded8bit_bands.bin')
    enc.dump_huffman(data_binstring_bands, cb_bands, 'encoded8bit_huffman_bands.bin')

# Synthesis of signal components
quantized8_decoded_bands = dec.load_single_binary_bandwise('encoded8bit_bands.bin', n_bands)
binary_decoded_bands, cb_decoded_bands = dec.load_double_binary_bandwise('encoded8bit_huffman_bands.bin', n_bands)
huffman_decoded_bands = [dec.dec_huffman(band, cb_decoded_bands[idx_band]) for idx_band, band 
                         in enumerate(binary_decoded_bands)]
huffman_decoded_bands = [band.astype(np.int8) for band in huffman_decoded_bands]

# upsampling
huff_audio_bands_us = [bap.upsample(band, N=sFactor) for band in huffman_decoded_bands]
quantized8_bands_us = [bap.upsample(band, N=sFactor) for band in quantized8_decoded_bands]

# reconstruct original signal
huff_reconstructed_audio = dec.applySynthesisFilterBank(huff_audio_bands_us, filterbank)
quantized8_reconstructed_audio = dec.applySynthesisFilterBank(quantized8_bands_us, filterbank)

if play_audio:
    print 'playing lowpass component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[0], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_bands_ds[0], fs/sFactor)
      
    print 'playing bandpass 1 component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[1], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_bands_ds[1], fs/sFactor)
      
    print 'playing bandpass 2 component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[2], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_bands_ds[2], fs/sFactor)
      
    print 'playing highpass component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[3], fs)
    print 'fs = {} Hz'.format(fs/sFactor)
    bap.play_audio(audio_bands_ds[3], fs/sFactor)
    
    print 'Play original 8 bit audio'
    bap.play_audio(quantized8_audio, fs)
    print 'Play reconstructed 8 bit decoded audio'
    bap.play_audio(huff_reconstructed_audio, fs)


if plot_audio:
    f1 = plotting.plot_spectrum(raw_audio, 'Original Audio Spectrum')
    f2 = plotting.plot_spectrum(audio_bands, 'Spectrum of the 4 subband components', 
                                ['lowpass', 'bp 1', 'bp 2', 'highpass'], 'lower center')
    
    f3 = plotting.plot_spectrum(huff_reconstructed_audio, 'Reconstructed Audio Spectrum')
    plt.show()