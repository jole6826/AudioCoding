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
    
def play_audio(audio, fs):
    '''
    plays single channel audio data
    '''
    p = pyaudio.PyAudio()
    
    datatype = audio.dtype
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