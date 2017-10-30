import pickle
import scipy
import scipy.io.wavfile as siow 
import numpy as np

def encode(sourceFile, saveFile, samplerate = 44100, dtype = np.int16):

	rate, sound = siow.read(sourceFile, False)
	outputBin = open(saveFile, 'wb')	
	pickle.dump(sound, outputBin)
	outputBin.close()
	
	return sound
	
def encode8bit(sourceFile, saveFile, samplerate = 44100):

	rate, sound = siow.read(sourceFile, False)
	outputBin = open(saveFile, 'wb')	
	
	sound = sound/(2**8-1)
	sound = sound.astype(dtype=np.uint8) 
	
	pickle.dump(sound, outputBin)
	outputBin.close()
	
	return sound