
import scipy.io.wavfile as wav
import pyaudio
import numpy as np



def playFile(audio, samplingRate, channels):


    p = pyaudio.PyAudio()

    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)
    # play. May repeat with different volume values (if done interactively)


    sound = (audio.astype(numpy.int16).tostring())
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return sound

	
def cutAudio(name, start, lenge=8):	
	rate, data = wav.read(name)
	cutSoundDat = data[44100*start:44100*(start+lenge),:]
	return cutSoundDat

def getInfo(name):
	import wave
	wavObj = wave.open(name, 'r')

	channels = wavObj.getnchannels()
	sampleWidthBytes = wavObj.getsampwidth()
	
	#Steves lesson: Reihenfolge ist wichtig
	return channels, sampleWidthBytes
	
def doFFT(data, samplerate):
	from scipy.fftpack import fft
	import numpy as np
	from scipy import arange
	import matplotlib.pyplot as plt
	chunk = 1024
	i=0
	for i in range(0, 4):
		chimp = data[i*chunk:(i+1)*chunk]
		# get just the one sided real part of the fft
		re = (np.abs(np.fft.rfft(chimp)))		
		re = np.fft.rfft(chimp)
		# re = 20*np.log10(np.abs(np.fft.rfft(chunkCh)))
		f = np.linspace(0, samplerate/2, len(re))
		plt.plot(f, re)
	
	plt.show()
	return
	
# def encrframewk(name, BitsPerSample):    

	# Wave_write.setsampwidth(2) #16bit




















