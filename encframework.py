import basic_audio_proc
import huffmanCoding as hc
import filterBanks as fb
import pickle
import os


def read_segment(audio_id, length_segment, channel):
    # read wav segment
    [norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment(audio_id, length_segment, channel)
    return raw_audio, norm_audio, org_dtype, fs

def quantize(norm_audio, org_dtype, n_bits):
    # Quantize with 16 or 8 bit
    quantized_audio = basic_audio_proc.quantize(norm_audio, org_dtype, n_bits)
    return quantized_audio

def applyAnalysisFilterBank(audio, n_bands, fs):
    # uses a filter bank with n_bands subbands to decompose a signal into n_bands parts (lowpass, n_bands-2 bandpass and highpass signal)
    filterbank = fb.createFilterBank(fs, n_bands)
    audio_in_bands = fb.apply_filters(audio, filterbank)

    return audio_in_bands, filterbank

def enc_huffman(audio, n_bits):
    cb, cb_tree = hc.createHuffmanCodebook(audio, n_bits)
    data_bits = hc.huffmanEncoder(audio, cb, n_bits)

    n_zero_pad = 8 - ((len(data_bits)+3) % 8)
    if n_zero_pad == 8:
        n_zero_pad = 0
    padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits + '{:b}'.format(0).zfill(n_zero_pad)
    data_binstring = hc.pack_bits_to_bytes(padded_bits)

    return cb, cb_tree, data_binstring

def enc_twos_complement(audio, n_bits):
# this is just regular coding as pickling of integers would do but enables arbitrary, e.g. 7 or 11 bit coding
# for which there are no datatypes in numpy/python
    
    binsymbols = [decimal_2_twos_complement(sample, n_bits) for sample in audio]
    coded = ''.join(binsymbols)
    
    n_zero_pad = 8 - ((len(coded)+3) % 8)
    if n_zero_pad == 8:
        n_zero_pad = 0
    padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + coded + '{:b}'.format(0).zfill(n_zero_pad)
    data_binstring = hc.pack_bits_to_bytes(padded_bits)        
    
    return data_binstring

def decimal_2_twos_complement(value, n_bits):
    if (value & (1 << (n_bits-1))) != 0: #check whether value is negative
        value = (1 << n_bits) + value
    
    coded = '{:b}'.format(value).zfill(n_bits)
    return coded
    
def dump_twos_complement(data_binstring, bitdemand, dump_fname):
    if type(data_binstring).__name__ == 'list': #list of strs, ergo multiple signals to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            for idx, band in enumerate(data_binstring):
                pickle.dump(band, f, 1)
                pickle.dump(bitdemand[idx], f, 1)
    elif type(data_binstring).__name__ == 'str': #only one signal to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            pickle.dump(data_binstring, f, 1)
            pickle.dump(bitdemand, f, 1)
    else: #error
        raise TypeError('Either pass list of strs or single str. You passed {}.'.format(type(data_binstring).__name__))

def dump_huffman(data_binstring, cb, bitdemand, dump_fname):  
    if type(data_binstring).__name__ == 'tuple': #list of strs, ergo multiple signals to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            for idx_band, binstring_band in enumerate(data_binstring):
                pickle.dump(cb[idx_band], f, 1)
                pickle.dump(binstring_band, f, 1)
                pickle.dump(bitdemand[idx_band], f, 1)
    elif type(data_binstring).__name__ == 'str': #only one signal to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            pickle.dump(cb, f, 1)
            pickle.dump(data_binstring, f, 1) 
            pickle.dump(bitdemand, f, 1)
    else: #error
        raise TypeError('Either pass tuple (list) of strs or single str. You passed {}.'.format(type(data_binstring).__name__)) 