import scipy.io.wavfile as wv
import numpy as np
import pyaudio

def normalize(audio):
    # normalizes single channel audio data with maximum value that can be stored in the datatype
    
    datatype = audio.dtype;
    maxval_datatype = np.abs(np.iinfo(datatype).min);
    
    # take care of overflows / make sure normalization can be performed (needs floating points)
    audio = audio.astype(np.float32)
    
    normaudio = audio / maxval_datatype;
    return normaudio;

def quantize(audio, org_dtype, wordlength):
    ''' 
    quantizes single channel audio data to 16 bit
    
    depending on the datatype of the original wav, the prior normalization by the maximum value of
    the datatype will result in a different maximum number that has to be taken into account for calculating
    the proper stepsize of the quantizer 
    (e.g. int16 goes from -32768...32767 -> max value is 32767/32768, 
    int32 goes from -2147483648...-2147483647 -> different max value)
    '''
    
    min_dtype = float(np.iinfo(org_dtype).min);
    max_dtype = float(np.iinfo(org_dtype).max);
    
    stepsize = ((max_dtype/np.abs(min_dtype)) - (min_dtype/np.abs(min_dtype))) / (2**wordlength);
    quantized = np.round(audio/stepsize);
    
    output_datatypes = {8: np.int8,
                        16: np.int16,
                        32: np.int32};
    quantized = quantized.astype(output_datatypes[wordlength]);
    return quantized

def read_segment(filename, duration, channel):
    '''
    Reads arbitrary wav file. 
    
    Duration has to be specified in seconds. 
    Channel has to be specified according to .wav specification (0 = left, 1 = right, 2 = center, 3 = LFE, 
    4 = rear left, 5 = rear right)
    '''
    [fs, audio] = wv.read(filename);
    [n_samples, _] = audio.shape;
    
    samples_segment = duration * fs; # conversion from seconds to samples
    
    if n_samples/2 >= samples_segment:
        start_segment = n_samples/2;
        end_segment = n_samples/2 + samples_segment;
    elif n_samples >= samples_segment:
        start_segment = n_samples - samples_segment;
        end_segment = n_samples/2 + samples_segment;
    else:
        start_segment = 0;
        end_segment = n_samples

    audio_segment = audio[start_segment:end_segment, :];

    # different maximum for each channel
    mono_norm_segment = normalize(audio_segment[:,channel]);
    mono_raw_segment = audio_segment[:, channel];
    return mono_norm_segment, mono_raw_segment, audio.dtype;

def write_wav(filename, rate, data):
    wv.write(filename, rate, data)

def sound(s, FS,bitdepth):
	"This function plays out a vector s as a sound at sampling rate FS, like on Octave or Matlab, with: import soundfloat; soundfloat.sound(s,FS)" 
	
	CHUNK = 1024 #Blocksize
	#WIDTH = 2 #2 bytes per sample
	CHANNELS = 1 #2
	RATE = FS  #Sampling Rate in Hz
	p = pyaudio.PyAudio()

	if bitdepth == 8:
		pyFormat = pyaudio.paInt8
		dtype = np.int8
	else:
		pyFormat = pyaudio.paInt16
		dtype = np.int16

	stream = p.open(format=pyFormat,
				channels=CHANNELS,
                rate=RATE,
                #input=False,
                output=True,
                #input_device_index=10,
                #frames_per_buffer=CHUNK
                )
	stream.write(s.astype(dtype))
	"""
	for i in range(0, int(len(s) / CHUNK) ):
		#print "i=", i
		#Putting samples into blocks of length CHUNK:
		samples=s[i*CHUNK:((i+1)*CHUNK)];
		samples=clip(samples,-1,1)
		#print samples[1]
		#print "len(samples)= ", len(samples)
		#Writing data back to audio output stream: 
		stream.write(samples.astype(np.float32))
	"""

	stream.stop_stream()
	stream.close()

	p.terminate()
	print("* done")
