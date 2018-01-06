import pickle
import huffmanCoding as hc
import os
import filterBanks as fb
import basic_audio_proc as bap
import numpy as np

def load_twoscomp_binary(fname):
    with open(os.path.join('bin', fname), 'rb') as f:
        data_binary = pickle.load(f)
        n_bits = pickle.load(f)
    return data_binary, n_bits

def load_twoscomp_binary_bandwise(fname, nbands):
    data_binary = [None] * nbands
    n_bits = [None] * nbands
    with open(os.path.join('bin', fname), 'rb') as f:
        for i in range(nbands):
            data_binary[i] = pickle.load(f)
            n_bits[i] = pickle.load(f)
    return data_binary, n_bits

def load_huffman_binary(fname):
    with open(os.path.join('bin', fname), 'rb') as f:
        cb = pickle.load(f)
        data_binary = pickle.load(f)
        n_bits = pickle.load(f)
    return data_binary, cb, n_bits

def load_huffman_binary_bandwise(fname, nbands):
    data_binary_bands = [None] * nbands
    cb_bands = [None] * nbands
    n_bits = [None] * nbands
    with open(os.path.join('bin', fname), 'rb') as f:
        for i in range(nbands):
            cb_bands[i] = pickle.load(f)  
            data_binary_bands[i] = pickle.load(f) 
            n_bits[i] = pickle.load(f)
    return data_binary_bands, cb_bands, n_bits

def dequantize(quant_indices, n_bits):
    audio = bap.dequantize(quant_indices, n_bits)
    return audio

def applySynthesisFilterBank(audio_in_bands, filterbank):
    # uses a filter bank with 4 subbands to decompose a signal into 4 parts (lowpass, 2 bandpass and highpass signal)

    recon_audio = fb.applyFiltersSynthesis(audio_in_bands, filterbank)

    return recon_audio

def dec_twoscomp(data_binary, n_bits):
    data = hc.unpack_bytes_to_bits(data_binary)
    
    n_padded_zeros = int(data[0:3], 2)
    if n_padded_zeros == 0:
        audio_data = data[3:]
    else:
        audio_data = data[3:-n_padded_zeros]
    
    n_samples = len(audio_data) / n_bits
    
    # translate every n_bits bits back to decimal
    decoded_audio = [twos_complement_2_decimal(audio_data[i*n_bits:(i+1)*n_bits], n_bits) 
                     for i in np.arange(n_samples)]
    decoded_audio = np.array(decoded_audio)
    
    return decoded_audio
    
    
def twos_complement_2_decimal(twos_comp, n_bits):
    mask = 2**(n_bits-1)
    twos_comp = int(twos_comp, 2)
    return -(twos_comp & mask) + (twos_comp & ~mask)
      
def dec_huffman(data_binary, cb):    
    # read bitstream from huffman encoded files
    data = hc.unpack_bytes_to_bits(data_binary)
    
    n_padded_zeros = int(data[0:3], 2)
    if n_padded_zeros == 0:
        audio_data = data[3:]
    else:
        audio_data = data[3:-n_padded_zeros]
    decoded_audio = hc.huffmanDecoder(audio_data, cb)
    
    return decoded_audio