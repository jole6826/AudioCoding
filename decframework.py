import pickle
import huffmanCoding as hc
import os

def load_single_binary(fname):
    data_binary = pickle.load(open(os.path.join('bin', fname), 'rb'))
    return data_binary

def load_double_binary(fname):
    with open(os.path.join('bin', fname), 'rb') as f:
        cb = pickle.load(f)
        data_binary = pickle.load(f)
    return data_binary, cb
       
def dec_huffman(data_binary, cb):    
    # read bitstream from huffman encoded files
    data = hc.unpack_bytes_to_bits(data_binary)
    
    n_padded_zeros = int(data[0:3], 2)
    audio_data = data[3:-n_padded_zeros]
    decoded_audio = hc.huffmanDecoder(audio_data, cb)
    
    return decoded_audio