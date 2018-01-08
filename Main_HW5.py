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
play_audio = True
dump_files = True
n_bands = 128
n_brkbands = 48
fs = 44100

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')
    
mdct_filterbank = fb.create_mdct_filterbank(n_bands)

''' Test mdct filterbank with ramp function: expected outcome: only delay of 2*N-1
'''

ramp = np.arange(0, np.iinfo(np.int16).max + 1)
ramp = np.append(np.zeros(fs, dtype=np.int16), ramp)
ramp = np.append(ramp, np.iinfo(np.int16).max * np.ones(fs, dtype=np.int16))

# decompose into subbands
ramp_bands = fb.apply_filters(ramp, mdct_filterbank)

# downsampling
ramp_bands_ds = [bap.downsample(band, N=n_bands) for band in ramp_bands]

# upsampling
ramp_bands_us = [bap.upsample(band, N=n_bands) for band in ramp_bands_ds]

# reconstruct original signal
ramp_reconstructed = dec.applySynthesisFilterBank(ramp_bands_us, mdct_filterbank)

f1 = plotting.plot_time(mdct_filterbank[n_bands/2], title='IR of MDCT Band {}'.format(n_bands/2))
f2 = plotting.plot_spectrum(mdct_filterbank[n_bands/2], title='Frequency Response of MDCT Band {}'.format(n_bands/2))
#f3 = plotting.plot_filterbank(mdct_filterbank) #only uncomment for small n_bands to get meaningful plot
f4 = plotting.plot_time([ramp, ramp_reconstructed], title='Reconstruction of MDCT filterbank', 
                        legend_names=['Original Ramp', 'Reconstructed Ramp'])

plt.show()