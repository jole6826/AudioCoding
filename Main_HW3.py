import encframework as enc
import decframework as dec
import os
import matplotlib.pyplot as plt

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
dumpHuffman = True

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')

# Encoding  
quantized_audio, norm_audio, fs, dump_fname = enc.read_and_quantize('Track48.wav', length_segment, channel, n_bits=8)
cb, cb_tree, hdump_fname = enc.enc_huffman(quantized_audio)

# Decoding
data_binary, cb = dec.load_double_binary(hdump_fname)
huff_decoded_audio = dec.dec_huffman(data_binary, cb)

diff = huff_decoded_audio - quantized_audio
plt.plot(diff)
plt.show()