import scipy.io.wavfile as wav
import pyaudio
import numpy as np
#import matplotlib.pyplot as plt


def playAudio(data,samplingrate,channels):
	
	p = pyaudio.PyAudio()
	
	# open audio stream

	stream = p.open(format=pyaudio.paInt16, # paFloat32, # paInt16,
					channels=channels,
					rate=samplingrate,
					output=True)


	sound = (data.astype(np.int16).tostring())
	stream.write(sound)

    # close stream and terminate audio object
	stream.stop_stream()
	stream.close()
	p.terminate()
	return

def getFragment(data,tShift,t,RATE):

	nShiftSamples = tShift * RATE
	nSamples = t*RATE # seconds*Hz

	fragment = wavData[nShiftSamples:nShiftSamples+nSamples,:]
	return fragment



### Homework 1 Audio Coding ###

RATE, wavData = wav.read("Track48.wav")

print (len(wavData), RATE, wavData.dtype, "Shape", wavData.shape)

audio = wavData #input_quantized #quant_stereo #left #wavArray left_quantized

channels = 2

#playAudio(audio, RATE, channels)


# extract 8 seconds 

tShift = 2 # seconds
t = 8 #seconds

fragment = getFragment(wavData,tShift,t,RATE)

max_value = np.iinfo(fragment.dtype).max


fragmentFloat = fragment.astype(float)
normalizedFragment = fragmentFloat/max_value

# Play normalized fragment
playAudio(normalizedFragment*max_value, RATE, channels)

#Plot each channel to check its not silent
#plt.plot(normanormalizedFragment)

### 2 blockwise FFT processing ### 

BLCK_SZ = 1024
normChannel1 = normalizedFragment[:,0]
f = np.linspace(0,RATE/2,num=BLCK_SZ)
print(f)
for i in range(0,4):
	chunk = normChannel1[i*BLCK_SZ:((i+1)*BLCK_SZ)]
	fftChunk = np.fft.fft(chunk)
	magnitudeFFTChunk = np.absolute(fftChunk)
	# plot magnitudes (abs) of each fft on top of each other
	#plt.plot(magnitudeFFTChunk)
	#print(i+1,magnitudeFFTChunk[0:19])

#plt.show()
	






