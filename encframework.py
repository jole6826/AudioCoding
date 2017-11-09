import basic_audio_proc
import numpy as np
import warnings
import pickle
import matplotlib.pyplot as plt

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
playplot_audio = False

<<<<<<< HEAD
# read wav segment
[norm_audio, raw_audio, org_dtype] = basic_audio_proc.read_segment('Track48.wav', 5, 0);
=======

# read wav segment
[norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment('Track48.wav', length_segment, channel)
>>>>>>> master

encoded16bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 16)
encoded8bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 8)

<<<<<<< HEAD

# TODO play audio segment and quantized versions
print("Playing 8bit quantized:")
basic_audio_proc.sound(encoded8bit_audio, 44100,8);
print("Playing 16bit quantized:")
basic_audio_proc.sound(encoded16bit_audio, 44100,16);
# TODO plot both channels

#pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1);
#pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)
=======
pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1)
pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)
   
if playplot_audio:
    plt.plot(norm_audio, lw = 0.2)
    plt.show()
    basic_audio_proc.play_audio(norm_audio, fs)
    
    plt.plot(encoded16bit_audio, lw = 0.2)
    plt.show()    
    basic_audio_proc.play_audio(encoded16bit_audio, fs)
    
    plt.plot(encoded8bit_audio, lw = 0.2)
    plt.show()    
    basic_audio_proc.play_audio(encoded8bit_audio, fs)
    
    # block into 1024 sample blocks
    framed_audio = basic_audio_proc.frame_audio(norm_audio, 1024)
    fft_audio = np.abs(np.fft.fft(framed_audio, axis=1))
    frqz_resp = 20*np.log10(fft_audio)

    plt.semilogx(frqz_resp[:,0], 'b', lw=0.4, alpha=0.5)
    plt.semilogx(frqz_resp[:,1], 'r', lw=0.4, alpha=0.5)
    plt.semilogx(frqz_resp[:,2], 'g', lw=0.4, alpha=0.5)
    plt.semilogx(frqz_resp[:,3], 'y', lw=0.4, alpha=0.5)
    plt.show()
    
    
>>>>>>> master
