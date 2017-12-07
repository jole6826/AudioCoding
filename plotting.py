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
        
    plot_time(fb, 'Filter Impulse Responses', 
                       legend_names, 'upper right')
    #('low pass', 'bandpass 1', 'bandpass 2','highpass')
    plot_spectrum(fb, 'Filter Magnitude Response', 
                           legend_names, 'lower center')
    plt.show()