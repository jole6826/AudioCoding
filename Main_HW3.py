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
dump_files = True

# Handle folder structure
if not os.path.exists('bin'):
    os.makedirs('bin')

# Encoding  
raw_audio, norm_audio, org_dtype, fs = enc.read_segment('Track48.wav', length_segment, channel)
quantized_audio = enc.quantize(norm_audio, org_dtype, n_bits=8)
cb, cb_tree, data_binstring = enc.enc_huffman(quantized_audio)

if dump_files:
    enc.dump_quantized(quantized_audio, 'encoded8bit.bin')
    enc.dump_huffman(data_binstring, cb, 'encoded8bit_huffman.bin')

# Decoding
data_binary, cb = dec.load_double_binary('encoded8bit_huffman.bin')
huff_decoded_audio = dec.dec_huffman(data_binary, cb)

diff = huff_decoded_audio - quantized_audio
plt.plot(diff)
plt.show()