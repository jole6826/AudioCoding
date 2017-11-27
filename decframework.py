import pickle
import basic_audio_proc
import huffmanCoding as hc
import numpy as np
import matplotlib.pyplot as plt
import bitstring as bits

# load unencoded 8bit or 16bit
audio16bit = pickle.load(open('encoded16bit.bin', 'rb'))
audio8bit = pickle.load(open('encoded8bit.bin', 'rb'))
# 8 bit is unsigned: https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.io.wavfile.write.html
audio8bitUnsigned = (audio8bit + 128).astype(np.uint8)

# read bitstream from huffman encoded files
bitstreamHuffEnc = bits.BitArray(filename = 'encoded8bit_huffman.bin')
# read Huffman codebook
cb = pickle.load(open('cb.bin', 'rb'))

''' Distribution Approach
bitstreamHuffEncGD = bits.BitArray(filename = 'huffEncoded8bit_gaussianDist.bin')
cbGaussianDist = pickle.load(open('cbGaussianDist.bin', 'rb'))

hcDecodedGD = hc.huffmanDecoder(bitstreamHuffEncGD, cbGaussianDist)
'''

# decoding
all_bits = bitstreamHuffEnc.bin
n_padded_zeros = int(all_bits[0:3], 2)
data_stream = all_bits[3:-n_padded_zeros]
hcDecoded = hc.fastHuffmanDecoder(data_stream, cb)

diff = hcDecoded - audio8bit
plt.plot(diff)
plt.show()

# Homework 1: write 8bit/16bit encoded to wav
basic_audio_proc.write_wav('decoded16bit.wav', 44100, audio16bit)
basic_audio_proc.write_wav('decoded8bit.wav', 44100, audio8bitUnsigned)

[norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment('Track48.wav', 8, 1)

plt.plot(basic_audio_proc.normalize(audio8bit), 'r', lw = 0.5, alpha = 0.5, label = '8 Bit Quantization')
plt.plot(norm_audio, 'b', lw = 0.5, alpha = 0.5, label = 'Original wav')
plt.legend()
plt.show()