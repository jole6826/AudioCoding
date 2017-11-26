import basic_audio_proc
import huffmanCoding as hc
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bitstring as bits

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
dumpFiles = False
dumpHuffman = True


# read wav segment
[norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment('Track48.wav', length_segment, channel)


encoded16bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 16)
encoded8bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 8)

# processing and coding goes here
# creating codebook like: https://gist.github.com/mreid/fdf6353ec39d050e972b

# create huffman codebook using original signal
cb = hc.createHuffmanCodebook(encoded8bit_audio)

# create codebook using gaussian distribution of probablities
testVasls = np.linspace(-128,127,num=256)
mu = 0
sig = 30
x = np.linspace(0,256,num=256)
testHist = np.exp(-np.power(testVasls - mu, 2.) / (2 * np.power(sig, 2.)))
cbGaussianDist = hc.createHuffmanCodebookFromHist(testHist, testVasls)

# plt.plot(testVasls,testHist)
# plt.show()

# encode using that codebooks
huffmanEncoded8bit_fullInfos = hc.huffmanEncoder(encoded8bit_audio, cb)
huffmanEncoded8bit_gaussianDist = hc.huffmanEncoder(encoded8bit_audio, cbGaussianDist)


# save files as binary data
if dumpHuffman:
    with open('huffEncoded8bit_fullInfos.bin', 'wb') as f:
        huffmanEncoded8bit_fullInfos.tofile(f)
        f.close()
    with open('huffEncoded8bit_gaussianDist.bin', 'wb') as f:
        huffmanEncoded8bit_gaussianDist.tofile(f)
        f.close()
        
    # dump codebooks, ideally this should be in the same file as the bitstream
    pickle.dump(cb, open('cb.bin', 'wb'), 1)
    pickle.dump(cbGaussianDist, open('cbGaussianDist.bin', 'wb'), 1)

if dumpFiles:
    pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1)
    pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)


if play_audio:
    print("Playing original:")
    basic_audio_proc.play_audio(raw_audio, fs)
    print("Playing 16bit quantized:")
    basic_audio_proc.play_audio(encoded16bit_audio, fs)
    print("Playing 8bit quantized:")
    basic_audio_proc.play_audio(encoded8bit_audio, fs)

if plot_audio:
    t_axis = np.linspace(0, norm_audio.shape[0]-1, norm_audio.shape[0]) / float(fs)
    plt.plot(t_axis, norm_audio, lw = 0.2)
    plt.title('Original Audio')
    plt.xlabel('t in seconds')
    plt.show()

    plt.plot(t_axis, encoded16bit_audio, lw = 0.2)
    plt.title('16bit encoded Audio')
    plt.xlabel('t in seconds')
    plt.show()

    plt.plot(t_axis, encoded8bit_audio, lw = 0.2)
    plt.title('8bit encoded Audio')
    plt.xlabel('t in seconds')
    plt.show()

    # block into 1024 sample blocks
    framed_audio = basic_audio_proc.frame_audio(norm_audio, 1024)
    fft_audio = np.abs(np.fft.fft(framed_audio, axis=0))
    frqz_resp = 20*np.log10(fft_audio)

    plt.plot(frqz_resp[0,:], 'b', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[1,:], 'r', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[2,:], 'g', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[3,:], 'y', lw=0.4, alpha=0.5)
    plt.xlabel('f in frequency bins')
    plt.ylabel('Magnitude in dB')
    plt.show()
