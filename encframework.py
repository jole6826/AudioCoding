import basic_audio_proc
import huffmanCoding as hc
import pickle
import os

dumpFiles = True


def read_and_quantize(audio_id, length_segment, channel, n_bits):
    # read wav segment
    [norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment(audio_id, length_segment, channel)

    # Quantize with 16 or 8 bit
    quantized_audio = basic_audio_proc.quantize(norm_audio, org_dtype, n_bits)

    if dumpFiles:
        dump_fname = 'encoded' + str(n_bits) + 'bit.bin'
        pickle.dump(quantized_audio, open(os.path.join('bin', dump_fname), 'wb'), 1)

    return quantized_audio, norm_audio, fs, dump_fname

def enc_huffman(audio):
    cb, cb_tree = hc.createHuffmanCodebook(audio)
    data_bits = hc.huffmanEncoder(audio, cb)

    n_zero_pad = 8 - ((len(data_bits)+3) % 8)
    padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits + '{:b}'.format(0).zfill(n_zero_pad)
    data_binstring = hc.pack_bits_to_bytes(padded_bits)

    if dumpFiles:
        dump_fname = 'encoded8bit_huffman.bin'
        with open(os.path.join('bin', dump_fname), 'wb') as f:
            pickle.dump(cb, f, 1)
            pickle.dump(data_binstring, f, 1)

    return cb, cb_tree, dump_fname
