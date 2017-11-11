import basic_audio_proc
import numpy as np
import warnings
import pickle
#import matplotlib.pyplot as plt

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
playplot_audio = False
plot_audio = False
play_audio = False


# read wav segment
[norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment('Track48.wav', length_segment, channel)


encoded16bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 16)
encoded8bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 8)


#pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1)
#pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)
   

if play_audio:
	print("Playing original:")
	basic_audio_proc.play_audio(norm_audio, fs)
	print("Playing 16bit quantized:")
	basic_audio_proc.play_audio(encoded16bit_audio, fs)
	print("Playing 8bit quantized:")
	basic_audio_proc.play_audio(encoded8bit_audio, fs)
if plot_audio:
    plt.plot(norm_audio, lw = 0.2)
    plt.show()
    
    plt.plot(encoded16bit_audio, lw = 0.2)
    plt.show()    
    
    plt.plot(encoded8bit_audio, lw = 0.2)
    plt.show()    
    
    # block into 1024 sample blocks
    framed_audio = basic_audio_proc.frame_audio(norm_audio, 1024)
    fft_audio = np.abs(np.fft.fft(framed_audio, axis=1))
    frqz_resp = 20*np.log10(fft_audio)

    plt.semilogx(frqz_resp[:,0], 'b', lw=0.4, alpha=0.5)
    plt.semilogx(frqz_resp[:,1], 'r', lw=0.4, alpha=0.5)
    plt.semilogx(frqz_resp[:,2], 'g', lw=0.4, alpha=0.5)
    plt.semilogx(frqz_resp[:,3], 'y', lw=0.4, alpha=0.5)
    plt.show()
    
    

