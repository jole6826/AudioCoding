import encframework as enc
import decframework as dec
import matplotlib.pyplot as plt
import basic_audio_proc as bap
import plotting
import scipy.signal as sig
import numpy as np
import os

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = True
play_filterbank = False
dump_files = True
n_bands = 4
sFactor = 4
n_brkbands = 48

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')

raw_audio, norm_audio, org_dtype, fs = enc.read_segment('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel)

quantized8_audio = enc.quantize(norm_audio, org_dtype, 8).astype(np.int8)
twoscomp8_data_binstring = enc.enc_twos_complement(quantized8_audio, 8)
quantized16_audio = enc.quantize(norm_audio, org_dtype, 16).astype(np.int16)
twoscomp16_data_binstring = enc.enc_twos_complement(quantized16_audio, 16)

if dump_files:
    enc.dump_twos_complement(twoscomp8_data_binstring, 8, 'encoded_8bit.bin')
    enc.dump_twos_complement(twoscomp16_data_binstring, 16, 'encoded_16bit.bin')

'''
Build Pychoacoustic model with whole audio signal, see block diagram lecture 7: p10
'''
f_stft_hz, t_stft, Zxx_stft = sig.stft(raw_audio, fs, nperseg=2048, nfft=2048)

if plot_audio:
    f2 = plotting.plot_spectrogram(raw_audio, fs)

f_stft_brk = bap.hz2bark(f_stft_hz)
W = bap.mapping2barkmat(fs, f_stft_brk, n_brkbands)
power_stft = np.square(np.abs(Zxx_stft))
power_in_brk_band = np.dot(W, power_stft)
spl_in_brk_band = 10 * np.log10(power_in_brk_band)

spreadingfunc_brk = bap.calc_spreadingfunc_brk(1, spl_in_brk_band, plot=plot_audio)
maskingthresh = bap.nonlinear_superposition(spreadingfunc_brk[100,:,:], alpha=0.3)

brk_bandwise_axis = np.linspace(0, (n_brkbands-1)/2.0, n_brkbands)
hz_bandwise_axis = bap.bark2hz(brk_bandwise_axis)
thresh_quiet = 3.64 * (hz_bandwise_axis/1000.) **(-0.8) - 6.5*np.exp( -0.6 * (hz_bandwise_axis/1000. - 3.3) ** 2.) + 1e-3*((hz_bandwise_axis/1000.) ** 4.)
thresh_quiet = np.clip(thresh_quiet, -20, 60)
thresh_quiet = thresh_quiet - 60 # convert from SPL to dB Full Scale (digital)

overall_thresh = np.maximum(maskingthresh, thresh_quiet)

if plot_audio:
    plotting.plot_maskingthresh(overall_thresh, hz_bandwise_axis)
    plt.show()

'''
decompose audio signal into 4 subbands and process each subband individually:
- each subband has different quant stepsize according to psychoacoustic model
- each subband has different huffman codes according to histogram
'''
# decompose into 4 subbands
audio_bands, filterbank = enc.applyAnalysisFilterBank(raw_audio, n_bands, fs)
if plot_audio:
    f1 = plotting.plot_filterbank(filterbank)
    plt.show()

# downsampling
audio_bands_ds = [bap.downsample(band, N=sFactor) for band in audio_bands]

# Quantization
norm_audio_bands_ds = [bap.normalize(band) for band in audio_bands_ds]
bitdemand = bap.bitdemand_from_masking(overall_thresh, n_bands, org_dtype)
print 'Bit demand for each subband: {}'.format(bitdemand)

quantized_audio_bands_ds = [enc.quantize(band, org_dtype, bitdemand[idx]) 
                            for idx, band in enumerate(norm_audio_bands_ds)]

# Regular 2s complement coding
twoscomp_data_binstring_bands = [enc.enc_twos_complement(quantized_audio, bitdemand[idx])
                                 for idx, quantized_audio in enumerate(quantized_audio_bands_ds)]

# Huffmann Encoding
cb_bands, cb_tree_bands, huff_data_binstring_bands = zip(*(enc.enc_huffman(quantized_audio, bitdemand[idx]) 
                                                      for idx, quantized_audio in enumerate(quantized_audio_bands_ds)))

if dump_files:
    enc.dump_twos_complement(twoscomp_data_binstring_bands, bitdemand, 'encoded_bands.bin')
    enc.dump_huffman(huff_data_binstring_bands, cb_bands, bitdemand, 'encoded_huffman_bands.bin')

'''
Decode & Synthesize again
'''
# Decode and Dequantize
twoscomp_bin_decoded_bands, twoscomp_n_bits = dec.load_twoscomp_binary_bandwise('encoded_bands.bin', n_bands)
twoscomp_decoded_bands = [dec.dec_twoscomp(band, twoscomp_n_bits[idx]) 
                          for idx, band in enumerate(twoscomp_bin_decoded_bands)]
twoscomp_decoded_bands = [dec.dequantize(band, twoscomp_n_bits[idx]) 
                          for idx, band in enumerate(twoscomp_decoded_bands)]

huffman_bin_decoded_bands, cb_decoded_bands, huffman_n_bits = dec.load_huffman_binary_bandwise('encoded_huffman_bands.bin', n_bands)
huffman_decoded_bands = [dec.dec_huffman(band, cb_decoded_bands[idx_band]) 
                         for idx_band, band in enumerate(huffman_bin_decoded_bands)]

huffman_decoded_bands = [dec.dequantize(band, huffman_n_bits[idx]) 
                         for idx, band in enumerate(huffman_decoded_bands)]

# upsampling
huff_audio_bands_us = [bap.upsample(band, N=sFactor) for band in huffman_decoded_bands]
twoscomp_bands_us = [bap.upsample(band, N=sFactor) for band in twoscomp_decoded_bands]

# reconstruct original signal
huff_reconstructed_audio = dec.applySynthesisFilterBank(huff_audio_bands_us, filterbank)
twoscomp_reconstructed_audio = dec.applySynthesisFilterBank(twoscomp_bands_us, filterbank)

if play_filterbank:
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

if play_audio:    
    print 'Play Original audio'
    bap.play_audio(raw_audio, fs)
    print 'Play 8 bit audio'
    bap.play_audio(quantized8_audio, fs)
    print 'Play reconstructed (variable bit demand in bands) decoded audio'
    bap.play_audio(huff_reconstructed_audio, fs)


if plot_audio:
    f1 = plotting.plot_spectrum(raw_audio, 'Original Audio Spectrum')
    f2 = plotting.plot_spectrum(audio_bands, 'Spectrum of the 4 subband components', 
                                ['lowpass', 'bp 1', 'bp 2', 'highpass'], 'lower center')
    
    f3 = plotting.plot_spectrum(huff_reconstructed_audio, 'Reconstructed Audio Spectrum')
    plt.show()