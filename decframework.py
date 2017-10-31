import pickle
import basic_audio_proc
import numpy as np

audio16bit = pickle.load(open('encoded16bit.bin', 'rb'));
audio8bit = pickle.load(open('encoded8bit.bin', 'rb'));
# 8 bit is unsigned: https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.io.wavfile.write.html
audio8bit = (audio8bit + 128).astype(np.uint8); 

basic_audio_proc.write_wav('decoded16bit.wav', 44100, audio16bit);
basic_audio_proc.write_wav('decoded8bit.wav', 44100, audio8bit);