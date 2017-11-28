import numpy as np
import matplotlib.pyplot as plt
import encframework as enc
import decframework as dec
import basic_audio_proc

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
dumpHuffman = True

# Encode
quantized_audio16, norm_audio16, fs, dump_fname16 = enc.read_and_quantize('Track48.wav', length_segment, channel, n_bits=16)
quantized_audio8, norm_audio8, fs, dump_fname8 = enc.read_and_quantize('Track48.wav', length_segment, channel, n_bits=8)

# Decode
dec_audio16 = dec.load_single_binary(dump_fname16)
dec_audio8 = dec.load_single_binary(dump_fname8)

# 8 bit is unsigned: https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.io.wavfile.write.html
dec_audio8_unsigned = (dec_audio8 + 128).astype(np.uint8)

# Homework 1: write 8bit/16bit encoded to wav
basic_audio_proc.write_wav('decoded16bit.wav', 44100, dec_audio16)
basic_audio_proc.write_wav('decoded8bit.wav', 44100, dec_audio8_unsigned)


plt.plot(basic_audio_proc.normalize(dec_audio8), 'r', lw = 0.5, alpha = 0.5, label = '8 Bit Quantization')
plt.plot(norm_audio16, 'b', lw = 0.5, alpha = 0.5, label = 'Original wav')
plt.legend()
plt.show()

if play_audio:
    print("Playing 16bit quantized:")
    basic_audio_proc.play_audio(quantized_audio16, fs)
    print("Playing 8bit quantized:")
    basic_audio_proc.play_audio(quantized_audio8, fs)

if plot_audio:
    t_axis = np.linspace(0, norm_audio16.shape[0]-1, norm_audio16.shape[0]) / float(fs)
    plt.plot(t_axis, norm_audio16, lw = 0.2)
    plt.title('Original Audio')
    plt.xlabel('t in seconds')
    plt.show()

    plt.plot(t_axis, quantized_audio16, lw = 0.2)
    plt.title('16bit encoded Audio')
    plt.xlabel('t in seconds')
    plt.show()

    plt.plot(t_axis, quantized_audio8, lw = 0.2)
    plt.title('8bit encoded Audio')
    plt.xlabel('t in seconds')
    plt.show()

    # block into 1024 sample blocks
    framed_audio = basic_audio_proc.frame_audio(norm_audio16, 1024)
    fft_audio = np.abs(np.fft.fft(framed_audio, axis=0))
    frqz_resp = 20*np.log10(fft_audio)

    plt.plot(frqz_resp[0,:], 'b', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[1,:], 'r', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[2,:], 'g', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[3,:], 'y', lw=0.4, alpha=0.5)
    plt.xlabel('f in frequency bins')
    plt.ylabel('Magnitude in dB')
    plt.show()
