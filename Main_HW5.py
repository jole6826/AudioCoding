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

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')
    

mdct_filterbank = fb.create_mdct_filterbank(n_bands)

f1 = plotting.plot_time(mdct_filterbank[n_bands/2], title='IR of MDCT Band {}'.format(n_bands/2))
f2 = plotting.plot_spectrum(mdct_filterbank[n_bands/2], title='Frequency Response of MDCT Band {}'.format(n_bands/2))
#f3 = plotting.plot_filterbank(mdct_filterbank) #only uncomment for small n_bands to get meaningful plot
plt.show()