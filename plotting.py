import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

def plot_spectrum(audio_signals, title='Spectrum', legend_names=None, legend_pos='best'):
    fig = plt.figure()
    
    if type(audio_signals).__name__ == 'list': #list of ndarrays, ergo multiple signals to plot
        for signal in audio_signals:
            w, H = sig.freqz(signal)
            plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    elif type(audio_signals).__name__ == 'ndarray': #only one signal to plot
        w, H = sig.freqz(audio_signals)
        plt.plot(w, 20 * np.log10(abs(H) + 1e-6))
    else: #error
        raise TypeError('Either pass list of ndarrays or single ndarray. You passed {}.'.format(type(audio_signals).__name__))
        
    plt.xlabel('Normalized Frequency')
    plt.ylabel('dB')
    plt.title(title)
    
    if legend_names != None:
        plt.legend(legend_names, loc=legend_pos)
    
    return fig

def plot_time(audio_signals, title='Time Domain', legend_names=None, legend_pos='best'):
    fig = plt.figure()
    
    if type(audio_signals).__name__ == 'list': #list of ndarrays, ergo multiple signals to plot
        for signal in audio_signals:
            plt.plot(signal)
    elif type(audio_signals).__name__ == 'ndarray': #only one signal to plot
        plt.plot(audio_signals)
    else: #error
        raise TypeError('Either pass list of ndarrays or single ndarray. You passed {}.'.format(type(audio_signals).__name__))
        
    plt.xlabel('Time in Samples')
    plt.title(title)
    
    if legend_names != None:
        plt.legend(legend_names, loc=legend_pos)
    
    return fig

def plot_filterbank(fb):    
    legend_names = [None] * len(fb)
    legend_names[0] = 'Low pass filter'
    legend_names[-1] = 'High pass filter'
    
    for bp_idx, __ in enumerate(fb[1:-1]):
        legend_names[bp_idx+1] = 'Band pass filter {}'.format(bp_idx+1)
        
    ftime = plot_time(fb, 'Filter Impulse Responses', 
                       legend_names, 'upper right')
    #('low pass', 'bandpass 1', 'bandpass 2','highpass')
    fspec = plot_spectrum(fb, 'Filter Magnitude Response', 
                           legend_names, 'lower center')
    
    return ftime, fspec
    
def plot_spectrogram(audio_signals, fs):
    if type(audio_signals).__name__ == 'list': #list of ndarrays, ergo multiple signals to plot
        for idx_band, signal in enumerate(audio_signals):
            f_stft_hz, t_stft, Zxx_stft = sig.stft(signal, fs, nperseg=2048, nfft=2048)
            plt.pcolormesh(t_stft, f_stft_hz, np.abs(Zxx_stft), vmin=0, vmax=max(signal)/20) 
            plt.title('STFT Magnitude Band {}'.format(idx_band))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()          
    elif type(audio_signals).__name__ == 'ndarray': #only one signal to plot
        f_stft_hz, t_stft, Zxx_stft = sig.stft(audio_signals, fs, nperseg=2048, nfft=2048)
        plt.pcolormesh(t_stft, f_stft_hz, np.abs(Zxx_stft), vmin=0, vmax=max(signal))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()     
    else: #error
        raise TypeError('Either pass list of ndarrays or single ndarray. You passed {}.'.format(type(audio_signals).__name__))
    
def plot_maskingthresh(masking_thresh, hz_bandwise_axis):
    fig = plt.figure()
    
    if type(masking_thresh).__name__ == 'list': #list of ndarrays, ergo multiple signals to plot
        for idx_band, band in enumerate(masking_thresh):
            plt.plot(hz_bandwise_axis, band, label='Subband {}'.format(idx_band))           
    elif type(masking_thresh).__name__ == 'ndarray': #only one signal to plot
        plt.plot(hz_bandwise_axis, masking_thresh)
    else: #error
        raise TypeError('Either pass list of ndarrays or single ndarray. You passed {}.'.format(type(masking_thresh).__name__))

    plt.xlabel('frequency [Hz]')
    plt.ylabel('Masking Threshold [dB]')
    plt.title('Masking Threshold incl. threshold in quiet')
    plt.legend()
    
    return fig