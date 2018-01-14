import encframework as enc
import decframework as dec
import matplotlib.pyplot as plt
import basic_audio_proc as bap
import plotting
import scipy.signal as sig
import numpy as np
import filterBanks as fb
import os

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
n_bands = 128
n_brkbands = 48
fs = 44100

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')
    

##############
''' 
Test mdct filterbank with ramp function: expected outcome: only delay of 2*N-1
'''
##########
mdct_fb_analysis, mdct_fb_synthesis = fb.create_mdct_filterbank(n_bands)

ramp = np.arange(0, np.iinfo(np.int16).max + 1)
ramp = np.append(np.zeros(fs, dtype=np.int16), ramp)
ramp = np.append(ramp, np.iinfo(np.int16).max * np.ones(fs, dtype=np.int16)).astype(np.int16)

# decompose into subbands
ramp_bands = enc.apply_mdct_analysis_filterbank(ramp, mdct_fb_analysis)
 
# downsampling
ramp_bands_ds = [bap.downsample(band, N=n_bands) for band in ramp_bands]
 
# upsampling
ramp_bands_us = [bap.upsample(band, N=n_bands) for band in ramp_bands_ds]
 
# reconstruct original signal
ramp_reconstructed = dec.apply_mdct_synthesis_filterbank(ramp_bands_us, mdct_fb_synthesis)

f1 = plotting.plot_time(mdct_fb_analysis[n_bands/2], title='IR of MDCT Band {}'.format(n_bands/2))
f2 = plotting.plot_spectrum(mdct_fb_analysis[n_bands/2], title='Frequency Response of MDCT Band {}'.format(n_bands/2))
#f3 = plotting.plot_filterbank(mdct_fb_analysis) #only uncomment for small n_bands to get meaningful plot
f4 = plotting.plot_time([ramp, ramp_reconstructed], title='Reconstruction of MDCT filterbank', 
                        legend_names=['Original Ramp', 'Reconstructed Ramp'])

plt.show()

###########
'''
Use MDCT Filterbank in Audio Coder
'''
###########

raw_audio, norm_audio, org_dtype, fs = enc.read_segment('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel)

# fixed quantization without psychoacoustic model
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

if plot_psycho:
    f2 = plotting.plot_spectrogram(raw_audio, fs)

f_stft_brk = bap.hz2bark(f_stft_hz)
W = bap.mapping2barkmat(fs, f_stft_brk, n_brkbands)
power_stft = np.square(np.abs(Zxx_stft))
power_in_brk_band = np.dot(W, power_stft)
spl_in_brk_band = 10 * np.log10(power_in_brk_band)

spreadingfunc_brk = bap.calc_spreadingfunc_brk(1, spl_in_brk_band, plot=plot_psycho)
maskingthresh = bap.nonlinear_superposition(spreadingfunc_brk[100,:,:], alpha=0.3)

brk_bandwise_axis = np.linspace(0, (n_brkbands-1)/2.0, n_brkbands)
hz_bandwise_axis = bap.bark2hz(brk_bandwise_axis)
thresh_quiet = 3.64 * (hz_bandwise_axis/1000.) **(-0.8) - 6.5*np.exp( -0.6 * (hz_bandwise_axis/1000. - 3.3) ** 2.) + 1e-3*((hz_bandwise_axis/1000.) ** 4.)
thresh_quiet = np.clip(thresh_quiet, -20, 60)
thresh_quiet = thresh_quiet - 60 # convert from SPL to dB Full Scale (digital)

overall_thresh = np.maximum(maskingthresh, thresh_quiet)

if plot_psycho:
    plotting.plot_maskingthresh(overall_thresh, hz_bandwise_axis)
    plt.show()

'''
decompose audio signal into subbands and process each subband individually:
- each subband has different quant stepsize according to psychoacoustic model
- each subband has different huffman codes according to histogram
'''
# decompose into 4 subbands
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
huff_audio_bands_us = [bap.upsample(band, N=n_bands) for band in huffman_decoded_bands]
twoscomp_bands_us = [bap.upsample(band, N=n_bands) for band in twoscomp_decoded_bands]

# reconstruct original signal
huff_reconstructed_audio = dec.apply_mdct_synthesis_filterbank(huff_audio_bands_us, mdct_fb_synthesis)
twoscomp_reconstructed_audio = dec.apply_mdct_synthesis_filterbank(twoscomp_bands_us, mdct_fb_synthesis)

if play_filterbank:
    print 'playing lowpass component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[0], fs)
      
    print 'playing bandpass component nr. {}'.format(n_bands/2)
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[n_bands/2], fs)

    print 'playing highpass component'
    print 'fs = {} Hz'.format(fs)
    bap.play_audio(audio_bands[-1], fs)

if play_audio:    
    print 'Play Original audio'
    bap.play_audio(raw_audio, fs)
    print 'Play 8 bit audio'
    bap.play_audio(quantized8_audio, fs)
    print 'Play reconstructed (variable bit demand in bands) decoded audio'
    bap.play_audio(huff_reconstructed_audio, fs)
