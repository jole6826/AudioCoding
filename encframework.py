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

def enc_huffman(audio):
    cb, cb_tree = hc.createHuffmanCodebook(audio)
    data_bits = hc.huffmanEncoder(audio, cb)

    n_zero_pad = 8 - ((len(data_bits)+3) % 8)
    if n_zero_pad == 8:
        n_zero_pad = 0
    padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits + '{:b}'.format(0).zfill(n_zero_pad)
    data_binstring = hc.pack_bits_to_bytes(padded_bits)

    return cb, cb_tree, data_binstring

def dump_quantized(quantized_audio, dump_fname):
    if type(quantized_audio).__name__ == 'list': #list of ndarrays, ergo multiple signals to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            for band in quantized_audio:
                pickle.dump(band, f, 1)
    elif type(quantized_audio).__name__ == 'ndarray': #only one signal to dump
        pickle.dump(quantized_audio, open(os.path.join('bin', dump_fname), 'wb'), 1)
    else: #error
        raise TypeError('Either pass list of ndarrays or single ndarray. You passed {}.'.format(type(quantized_audio).__name__))

def dump_huffman(data_binstring, cb, dump_fname):  
    if type(data_binstring).__name__ == 'tuple': #list of ndarrays, ergo multiple signals to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            for idx_band, binstring_band in enumerate(data_binstring):
                pickle.dump(cb[idx_band], f, 1)
                pickle.dump(binstring_band, f, 1)
    elif type(data_binstring).__name__ == 'str': #only one signal to dump
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            pickle.dump(cb, f, 1)
            pickle.dump(data_binstring, f, 1) 
    else: #error
        raise TypeError('Either pass tuple (list) of strs or single str. You passed {}.'.format(type(data_binstring).__name__)) 