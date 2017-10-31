import pickle
import scipy 
import scipy.io.wavfile as siow 
import numpy as np

def decode(sourceFile, saveFile, sampleRate = 44100):

	inputBin = open(sourceFile, 'rb')
	sound = pickle.load(inputBin)
	snd = siow.write(saveFile, sampleRate, sound)
	inputBin.close()
	return snd
	
def decode8bit(sourceFile, saveFile, sampleRate = 44100):

	inputBin = open(sourceFile, 'rb')
	sound = pickle.load(inputBin)
	
	sound = sound.astype(np.int16)
	sound = sound * 2**8-1
	
	siow.write(saveFile, sampleRate, sound)
	inputBin.close()
	return sound