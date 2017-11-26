import pickle
import basic_audio_proc
import huffmanCoding as hc
import numpy as np
import matplotlib.pyplot as plt
import bitstring as bits


audio16bit = pickle.load(open('encoded16bit.bin', 'rb'))
audio8bit = pickle.load(open('encoded8bit.bin', 'rb'))
# 8 bit is unsigned: https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.io.wavfile.write.html
audio8bitUnsigned = (audio8bit + 128).astype(np.uint8)

# read bitstream from huffman encoded files
bitstreamHuffEnc = bits.BitArray(filename = 'huffEncoded8bit_fullInfos.bin')
bitstreamHuffEncGD = bits.BitArray(filename = 'huffEncoded8bit_gaussianDist.bin')


# Huffman decoding
# should be replaced: read codebooks from binary files
cb = pickle.load(open('cb.bin', 'rb'))
cbGaussianDist = pickle.load(open('cbGaussianDist.bin', 'rb'))

# create codebook using gaussian distribution of probablities
# testVasls = np.linspace(-128,127,num=256)
# mu = 0
# sig = 30
# x = np.linspace(0,256,num=256)
# testHist = np.exp(-np.power(testVasls - mu, 2.) / (2 * np.power(sig, 2.)))
# cbGaussianDist = hc.createHuffmanCodebookFromHist(testHist,testVasls)

# decoding
hcDecoded = hc.huffmanDecoder(bitstreamHuffEnc, cb)
hcDecodedGD = hc.huffmanDecoder(bitstreamHuffEncGD, cbGaussianDist)



basic_audio_proc.write_wav('decoded16bit.wav', 44100, audio16bit)
basic_audio_proc.write_wav('decoded8bit.wav', 44100, audio8bitUnsigned)

[norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment('Track48.wav', 8, 1)

plt.plot(basic_audio_proc.normalize(audio8bit), 'r', lw = 0.5, alpha = 0.5, label = '8 Bit Quantization')
plt.plot(norm_audio, 'b', lw = 0.5, alpha = 0.5, label = 'Original wav')
plt.legend()
plt.show()