import numpy as np
import scipy.signal as signal

def createFilterBank(fs, n_bands):
    # uses a filter bank with 4 subbands to decompose a signal into 4 parts (lowpass, 2 bandpass and highpass signal)
    fNy = fs/2.0
    bandwidth = fNy/n_bands
    fb = [None]*n_bands
    
    # create filter coefficents for subband filtering
    # georg: I don't understand how to find a suitable number of taps, with 8*nSubbands as in Schuller's lecture, you get terrible
    # overshoots for Highpass filters (apparently due to remez filter design), internet says: odd number of samples solves this, it does!
    # also I changed from 1 -> 0.9 in passband to avoid clipping
    fb[0] = signal.remez(32*n_bands - 1, [0, bandwidth, bandwidth+500, fNy], [0.9, 0], [1, 100], Hz=fs)
    fb[-1] = signal.remez(32*n_bands - 1, [0, fNy-bandwidth-500, fNy-bandwidth, fNy], [0, 0.9], [100, 1], Hz=fs)
    
    for idx_band in range(1, n_bands-1):
        lower_bound = idx_band * bandwidth
        upper_bound = (idx_band+1) * bandwidth
        
        fb[idx_band] = signal.remez(32*n_bands - 1, [0, lower_bound-500, lower_bound, upper_bound, upper_bound+500, fNy],
                                    [0, 0.9, 0], [100, 1, 100], Hz=fs)
    
    return fb


def apply_filters(audio, filterbank):
    orig_type = audio.dtype
    min_dtype = np.iinfo(orig_type).min
    max_dtype = np.iinfo(orig_type).max
    
    audio_in_bands = [None] * len(filterbank)
    
    for idx_band, band_filter in enumerate(filterbank):
        audio_in_bands[idx_band] = signal.lfilter(band_filter, 1, audio)
        np.clip(audio_in_bands[idx_band], min_dtype, max_dtype)
        audio_in_bands[idx_band] = audio_in_bands[idx_band].astype(orig_type)

    return audio_in_bands


def applyFiltersSynthesis(audio_in_bands, filterbank):
    orig_type = audio_in_bands[0].dtype
    min_dtype = np.iinfo(orig_type).min
    max_dtype = np.iinfo(orig_type).max
    
    reconstructed_audio = np.zeros(len(audio_in_bands[0]))
    
    for idx_band, band_filter in enumerate(filterbank):
        reconstructed_audio += signal.lfilter(band_filter, 1, audio_in_bands[idx_band])
        
    np.clip(reconstructed_audio, min_dtype, max_dtype)
    reconstructed_audio = reconstructed_audio.astype(orig_type)

    return reconstructed_audio
