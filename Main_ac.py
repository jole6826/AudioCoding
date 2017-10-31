
import scipy.io.wavfile as wav
import pyaudio
import numpy as np
import play_wav as pl
import encframewk_ac as encframewk
import decframewk_ac as decframewk
import matplotlib.pyplot as plt
import pylab 

#######################################################
# 1.1 Seperate Channels

name = "Track48.wav"

samplerate, audio = wav.read("Track48.wav")

channels, sampleWidthBytes = pl.getInfo(name)

# zeig mir die Bits Bitch
print(sampleWidthBytes*8)
print(channels)
print(samplerate)

# soundData = pl.playFile(audio, samplerate, channels)
# string kommt zurueck  zum quantisieren in int 

cutSoundData = pl.cutAudio(name,10,1)
# pl.playFile(cutSoundData,samplerate,channels)

rate, dataInt = wav.read(name)

m1 = max(dataInt[:,0])
m2 = max(dataInt[:,1])
# print(m1,m2)
ch1 = dataInt[:,0]
ch2 = dataInt[:,1]
bereich = 2.**15-1
# norm1 = ch1/float(m1) 
# norm2 = ch2/float(m2)

#normalisiere auf Bereich

norm1 = ch1/float(bereich) 
norm2 = ch2/float(bereich)

# print(norm1)
# plt.plot(norm1)
# plt.plot(norm2)
# plt.show()

fragm1 = cutSoundData[:,0]
fragm2 = cutSoundData[:,1]

# pl.playFile(fragm1,samplerate,channels/2)
# pl.playFile(fragm2,samplerate,channels/2)


# plt.plot(fragm2)
# plt.plot(fragm1)sublime.log_commands(True)
# plt.show()

###########################################
# 1.2 FFT

sndData = cutSoundData[0::2]
# reArray = pl.doFFT(sndData, samplerate)

###########################################
# 1.3 quantization

encframewk.encode("Track48.wav", "encoded_ac.bin")    		
decframewk.decode("encoded_ac.bin", "decoded_ac.wav")		

encframewk.encode8bit("Track48.wav", "encoded8bit_ac.bin")
decframewk.decode8bit("encoded8bit_ac.bin", "decoded8bit_ac.wav")






# 