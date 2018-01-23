import encframework as enc
import decframework as dec
import matplotlib.pyplot as plt
import basic_audio_proc as bap
import plotting
import scipy.signal as sig
import numpy as np
import filterBanks as fb
import os
from os.path import basename
import time

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = True
plot_psycho = False
play_audio = True
play_filterbank = False
dump_files = True
n_bands = 128 # was at 128
aNLS = 0.3 # alpha for nonlinear superposition (was at 0.3)
aSF = 1.0 # alpha for spreading function (was 1.0)  1 for tonal, 0 for noiselike
n_brkbands = 48
fs = 44100

# define audio files in a list
audioFolder = 'audio'
f = ['rockyou_16.wav']
f.append('castanets_16.wav')
f.append('speech_16.wav')
f.append('imagine_Dragons_Thunder_short_32khz.wav')
settings = '_{}bands_aNLS_{}_aTon_{}'.format(n_bands, aNLS, aSF)

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')

# read in audio files
file= f[0]
base = basename(file)
name = os.path.splitext(base)[0]

#raw_audio, norm_audio, org_dtype, fs = enc.read_segment('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel)
raw_audio, norm_audio, org_dtype, fs = enc.read_segment(os.path.join(audioFolder,file), length_segment, channel)
raw_audio = bap.pad_zeros_to_blocklength(raw_audio, n_bands)
norm_audio = bap.pad_zeros_to_blocklength(norm_audio, n_bands)


'''
Regular 8bit and 16bit quantization for comparison
'''
quantized8_audio = enc.quantize(norm_audio, org_dtype, 8).astype(np.int8)
twoscomp8_data_binstring = enc.enc_twos_complement(quantized8_audio, 8)
quantized16_audio = enc.quantize(norm_audio, org_dtype, 16).astype(np.int16)
twoscomp16_data_binstring = enc.enc_twos_complement(quantized16_audio, 16)


start_time = time.time()

'''
Build Pychoacoustic model with whole audio signal, see block diagram lecture 7: p10
'''
f_stft_hz, t_stft, Zxx_stft = sig.stft(raw_audio, fs, nperseg=2048, nfft=2048)

if plot_psycho:
    f2 = plotting.plot_spectrogram(raw_audio, fs)

f_stft_brk = bap.hz2bark(f_stft_hz)
W = bap.mapping2barkmat(fs, f_stft_brk, n_brkbands)
power_stft = np.square(np.abs(Zxx_stft))
power_in_brk_band = np.dot(W, power_stft)
spl_in_brk_band = 10 * np.log10(power_in_brk_band)

spreadingfunc_brk = bap.calc_spreadingfunc_brk(aSF, spl_in_brk_band, plot=plot_psycho)
maskingthresh = bap.nonlinear_superposition(spreadingfunc_brk[100,:,:], alpha=aNLS)

brk_bandwise_axis = np.linspace(0, (n_brkbands-1)/2.0, n_brkbands)
hz_bandwise_axis = bap.bark2hz(brk_bandwise_axis)
thresh_quiet = 3.64 * (hz_bandwise_axis/1000.) **(-0.8) - 6.5*np.exp( -0.6 * (hz_bandwise_axis/1000. - 3.3) ** 2.) + 1e-3*((hz_bandwise_axis/1000.) ** 4.)
thresh_quiet = np.clip(thresh_quiet, -20, 60)
thresh_quiet = thresh_quiet - 60 # convert from SPL to dB Full Scale (digital)

overall_thresh = np.maximum(maskingthresh, thresh_quiet)


# create MDCT Filter Bank
mdct_fb_analysis, mdct_fb_synthesis = fb.create_mdct_filterbank(n_bands)

'''
decompose audio signal into subbands and process each subband individually:
- each subband has different quant stepsize according to psychoacoustic model
- each subband has different huffman codes according to histogram
'''
# decompose into subbands
audio_bands = enc.apply_mdct_analysis_filterbank(raw_audio, mdct_fb_analysis)

# downsampling
audio_bands_ds = [bap.downsample(band, N=n_bands) for band in audio_bands]

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
    enc.dump_twos_complement(twoscomp_data_binstring_bands, bitdemand, name + settings + '_encoded_bands.bin')
    enc.dump_huffman(huff_data_binstring_bands, cb_bands, bitdemand, name+ settings + '_encoded_huffman_bands.bin')



if dump_files:
    enc.dump_twos_complement(twoscomp8_data_binstring, 8, name + settings + '_encoded_8bit.bin')
    enc.dump_twos_complement(twoscomp16_data_binstring, 16, name + settings + '_encoded_16bit.bin')


size_original = os.stat(os.path.join('bin', (name + settings + '_encoded_16bit.bin'))).st_size
size_encoded = os.stat(os.path.join('bin', (name + settings + '_encoded_huffman_bands.bin'))).st_size
print("Encoded file is {}% of original file size.".format(100 * size_encoded / float(size_original)))

'''
Decode & Synthesize again
'''
# Decode and Dequantize
twoscomp_bin_decoded_bands, twoscomp_n_bits = dec.load_twoscomp_binary_bandwise(name + settings + '_encoded_bands.bin', n_bands)
twoscomp_decoded_bands = [dec.dec_twoscomp(band, twoscomp_n_bits[idx])
                          for idx, band in enumerate(twoscomp_bin_decoded_bands)]
twoscomp_decoded_bands = [dec.dequantize(band, twoscomp_n_bits[idx])
                          for idx, band in enumerate(twoscomp_decoded_bands)]

huffman_bin_decoded_bands, cb_decoded_bands, huffman_n_bits = dec.load_huffman_binary_bandwise(name + settings + '_encoded_huffman_bands.bin', n_bands)
huffman_decoded_bands = [dec.dec_huffman(band, cb_decoded_bands[idx_band])
                         for idx_band, band in enumerate(huffman_bin_decoded_bands)]

huffman_decoded_bands = [dec.dequantize(band, huffman_n_bits[idx])
                         for idx, band in enumerate(huffman_decoded_bands)]

# upsampling
huff_audio_bands_us = [bap.upsample(band, N=n_bands) for band in huffman_decoded_bands]
twoscomp_bands_us = [bap.upsample(band, N=n_bands) for band in twoscomp_decoded_bands]

# reconstruct original signal
huff_reconstructed_audio = dec.apply_mdct_synthesis_filterbank(huff_audio_bands_us, mdct_fb_synthesis)
twoscomp_reconstructed_audio = dec.apply_mdct_synthesis_filterbank(twoscomp_bands_us, mdct_fb_synthesis)

print ("Time for encoding and decoding: {}".format(time.time() - start_time))

audio_fb_without_quant = enc.apply_mdct_analysis_filterbank(raw_audio, mdct_fb_analysis)
audio_fb_without_quant = [bap.downsample(band, N=n_bands) for band in audio_fb_without_quant]
audio_fb_without_quant = [bap.upsample(band, N=n_bands) for band in audio_fb_without_quant]
audio_fb_without_quant = dec.apply_mdct_synthesis_filterbank(audio_fb_without_quant, mdct_fb_synthesis)

noise = audio_fb_without_quant - huff_reconstructed_audio
noise_power = np.sum(np.square(noise).astype(np.float) / len(noise))
signal_power = np.sum(np.square(audio_fb_without_quant).astype(np.float) / len(audio_fb_without_quant))
snr = 10 * np.log10(signal_power / noise_power)

f4 = plotting.plot_time([audio_fb_without_quant, huff_reconstructed_audio], title='Reconstruction of MDCT filterbank', 
 
                        legend_names=['Original', 'Reconstructed'])
axislimits = plt.ylim()
plt.show()

f3 = plotting.plot_time(noise, title='Noise')
plt.ylim(axislimits)
plt.show()

print ("Signal to noise ratio: {} dB".format(snr))

if play_audio:
    print 'Play Original audio'
    bap.play_audio(raw_audio, fs)
    print 'Play reconstructed (variable bit demand in bands) decoded audio'
    bap.play_audio(huff_reconstructed_audio, fs)
    print 'Play quantization noise'
    bap.play_audio(noise, fs)