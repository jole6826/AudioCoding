import scipy.io.wavfile as wv
import numpy as np
import pyaudio

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
    framed_audio = np.reshape(audio_zeropadded, (frame_length, -1))
    return framed_audio

def write_wav(filename, rate, data):
    wv.write(filename, rate, data)

def generateSinSignal(amps,freqs,d,fs):
	"""
	Create a signal consisting of a number of sinuses with amplitudes in numpy array amps and frequencies in numpy array freqs
	It has duration d in [s] and sampling frequency fs in [Hz]
	"""
	t = np.linspace(0,d,d*fs)
	s = np.zeros(t.shape)
	for i in range(amps.shape[0]):
		s = s + amps[i]*np.sin(2*np.pi*freqs[i]*t)

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

    Fhz = 600. * np.sinh(f/6.)
    return Fhz

def mapping2barkmat(fs):
    #Constructing matrix W which has 1’s for each Bark subband, and 0’s else:
    nfft=1024; nfilts=48; nfreqs=nfft/2
    binbarks = hz2bark(np.linspace(0,(nfft/2),(nfft/2)+1)*fs/nfft)
    W = np.zeros((nfilts, nfft))
    for i in xrange(nfilts):
        W[i,0:(nfft/2)+1] = (np.round(binbarks/step_barks)== i)
    return W
