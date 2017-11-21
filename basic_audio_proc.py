import scipy.io.wavfile as wv
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

def normalize(audio):
    # normalizes single channel audio data with maximum value that can be stored in the datatype

    datatype = audio.dtype
    maxval_datatype = np.abs(np.iinfo(datatype).min)

    # take care of overflows / make sure normalization can be performed (needs floating points)
    audio = audio.astype(np.float32)

    normaudio = audio / maxval_datatype
    return normaudio

def quantize(audio, org_dtype, wordlength):
    '''
    quantizes single channel audio data to 16 bit

    depending on the datatype of the original wav, the prior normalization by the maximum value of
    the datatype will result in a different maximum number that has to be taken into account for calculating
    the proper stepsize of the quantizer
    (e.g. int16 goes from -32768...32767 -> max value is 32767/32768,
    int32 goes from -2147483648...-2147483647 -> different max value)
    '''

    min_dtype = float(np.iinfo(org_dtype).min)
    max_dtype = float(np.iinfo(org_dtype).max)

    stepsize = ((max_dtype/np.abs(min_dtype)) - (min_dtype/np.abs(min_dtype))) / (2**wordlength)
    quantized = np.round(audio/stepsize)

    output_datatypes = {8: np.int8,
                        16: np.int16,
                        32: np.int32}
    quantized = quantized.astype(output_datatypes[wordlength])
    return quantized

def huffmanCoding(audio,cBook):
    '''huffmannCoding performs huffman coding on 8 bit quantized audio data
    Input:
        audio -     int8 quantized data (maybe in blocks of size 1024 or similar)
        cBook -     huffman codebook that stores the respective "bitstream" for each possible symbol of audio
                    in data range of int8

    Output:
        coded -     string (?, decide data structure) containing binary values
        '''
    import numpy as np

    nSamples = len(audio)
    maxVal = float(np.max(audio))
    minVal = float(np.min(audio))
    min_dtype = float(np.iinfo(np.int8).min)
    max_dtype = float(np.iinfo(np.int8).max)


    if maxVal > max_dtype or minVal < min_dtype:
        print ("data contains values outside of int8 range")

    print("error, not implemented yet")

    coded = ""

    for ix in xrange(0,nSamples,1):
        # apply codebook for each sample
        cBookIx = cBook[:, 0] == audio[ix]
        binSymbol = cBook[cBookIx,1]
        #binSymbol = "10"
        coded += binSymbol
    return coded

def read_segment(filename, duration, channel):
    '''
    Reads arbitrary wav file.

    Duration has to be specified in seconds.
    Channel has to be specified according to .wav specification (0 = left, 1 = right, 2 = center, 3 = LFE,
    4 = rear left, 5 = rear right)
    '''
    [fs, audio] = wv.read(filename)
    [n_samples, _] = audio.shape

    samples_segment = duration * fs # conversion from seconds to samples

    if n_samples/2 >= samples_segment:
        start_segment = n_samples/2;
        end_segment = n_samples/2 + samples_segment
    elif n_samples >= samples_segment:
        start_segment = n_samples - samples_segment
        end_segment = n_samples/2 + samples_segment
    else:
        start_segment = 0
        end_segment = n_samples

    audio_segment = audio[start_segment:end_segment, :]

    # different maximum for each channel
    mono_norm_segment = normalize(audio_segment[:,channel])
    mono_raw_segment = audio_segment[:, channel]
    return mono_norm_segment, mono_raw_segment, audio.dtype, fs

def frame_audio(audio, frame_length):
    '''
    frames single channel audio into consecutive blocks/frames of length frame_length
    '''
    n_zeros = audio.size % frame_length
    audio_zeropadded = np.append(audio, np.zeros(frame_length-n_zeros))
    framed_audio = np.reshape(audio_zeropadded, (-1, frame_length))
    return framed_audio

def write_wav(filename, rate, data):
    wv.write(filename, rate, data)

def generateSinSignal(amps,freqs,d,fs):
    """
    Create a signal consisting of a number of sinuses with amplitudes in numpy array amps and frequencies in numpy array freqs
    It has duration d in [s] and sampling frequency fs in [Hz]
    """
    t = np.arange(0, d, 1.0/fs)
    s = np.zeros(t.shape)
    for i in range(amps.shape[0]):
        s = s + amps[i] * np.sin(2 * np.pi * freqs[i] * t)

    return s


def play_audio(audio, fs):
    '''
    plays single channel audio data
    '''
    p = pyaudio.PyAudio()

    datatype = audio.dtype
    print datatype
    if datatype == np.int8:
        width = 1
    elif datatype == np.int16:
        width = 2
    elif datatype == np.int32:
        width = 4
    elif datatype == np.float32:
        width = 4

    output_format = pyaudio.get_format_from_width(width, False)
    stream = p.open(format = output_format, channels = 1, rate = fs, output = True)
    stream.write(audio.tostring())

### Functions for psycho acoustic model

def hz2bark(f):
    """ Method to compute Bark from Hz. Based on :
    https://github.com/stephencwelch/Perceptual-Coding-In-Python
    Args
    :
        f   : (ndarray)     Array containing frequencies in Hz.
    Returns :
        Brk : (ndarray)     Array containing Bark scaled values.
    """

    Brk = 6. * np.arcsinh(f/600.)
    return Brk

def bark2hz(Brk):
    """ Method to compute Bark from Hz. Based on :
    https://github.com/stephencwelch/Perceptual-Coding-In-Python
    Args:
        Brk : (ndarray)     Array containing Bark scaled values.
    Returns:
        f   : (ndarray)     Array containing frequencies in Hz.
    """

    Fhz = 600. * np.sinh(Brk/6.)
    return Fhz

def mapping2barkmat(fs, bin_brk, n_brkbands):
    #Constructing matrix W which has 1's for each Bark subband, and 0's else:
    nfft = bin_brk.shape[0]
    step_brk = 24.0 / n_brkbands
    
    W = np.zeros((n_brkbands, nfft))
    for i in xrange(n_brkbands):
        W[i,:] = (np.floor(bin_brk/step_brk)== i)
    return W

def calc_spreadingfunc_brk(alpha, spl_in_brk_band, plot):    
    n_bands = spl_in_brk_band.shape[0] #number of bark bands
    n_segments = spl_in_brk_band.shape[1] #number of time segments
    size_bands = 24.0 / n_bands #size of each band in bark
    spreading_func_brk = np.zeros((n_segments, n_bands, n_bands*2+1))
    
    for idx_brk_band, val_spl_brk in enumerate(spl_in_brk_band):
        band_upper_frqz_brk = (idx_brk_band+1)*size_bands
        band_lower_frqz_brk = idx_brk_band*size_bands
        band_center_brk = band_lower_frqz_brk + (band_upper_frqz_brk - band_lower_frqz_brk)*0.5
        bark_quarter_idx = int(band_center_brk*4) # peaks of spreading func shall be at half of band -> 0.25
        
        band_upper_frqz_hz = bark2hz(band_upper_frqz_brk)
        band_lower_frqz_hz = bark2hz(band_lower_frqz_brk)
        band_center_hz =  band_lower_frqz_hz + (band_upper_frqz_hz - band_lower_frqz_hz)*0.5
        
        O_f = alpha*(14.5 + idx_brk_band) + (1-alpha)*5.5  # Simultaneous masking for tones at Bark band 12
        slope_up = +27.0 * np.ones(n_segments)  # rising slope of spreading func
        slope_down = -(24.0 + 0.23*np.power(band_center_hz/1000, -1) - 0.2*val_spl_brk)  # Lower slope of spreading function
                
        peak = val_spl_brk - O_f
        for idx_t, _ in enumerate(val_spl_brk):
            spreading_func_brk[idx_t, idx_brk_band, 0:bark_quarter_idx+1] = np.linspace(-band_center_brk*slope_up[idx_t], 0, bark_quarter_idx+1) + peak[idx_t]
            spreading_func_brk[idx_t, idx_brk_band, bark_quarter_idx:n_bands*2+2] = np.linspace(0, ((n_bands-idx_brk_band)*size_bands)*slope_down[idx_t], n_bands*2-bark_quarter_idx+1) + peak[idx_t]
            
    if plot:
        brk_axis = np.linspace(0, n_bands/2, n_bands*2+1)
        for spreading_in_band in spreading_func_brk[n_segments/2,:,:]:
            plt.plot(brk_axis, spreading_in_band)
        plt.axis([0, n_bands/2, -100, 48])
        plt.xlabel('frequency [Bark]')
        plt.ylabel('L [dB]')
        plt.show()
    
    return spreading_func_brk

def nonlinear_superposition(spreadingfunc_brk, alpha):
    spreadingfunc_brk_linear = np.power(10, spreadingfunc_brk/10) # convert dB back to power
    exponentiated_bands = np.power(spreadingfunc_brk_linear, alpha)
    summed_bands = np.sum(exponentiated_bands, axis=0) # sum up bands (= columns)
    
    # because of centering the peaks of spreadingfunc in middle of band we have to add up two consecutive values to get a vector of
    # the length n_bands back
    summed_bands = (summed_bands + np.roll(summed_bands,-1))[:-1:2]
    
    exponentiaded_sum = np.power(summed_bands, (1-alpha))
    superposition_dB = 10 * np.log10(exponentiaded_sum) # convert back to dB
    return superposition_dB