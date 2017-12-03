import basic_audio_proc
import huffmanCoding as hc
import filterBanks as fb
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


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


def applyAnalysisFilterBank(audio, fs, plotFilter=False, plotAudio=False, playAudio=False):
    # uses a filter bank with 4 subbands to decompose a signal into 4 parts (lowpass, 2 bandpass and highpass signal)

    bLp, bBp1, bBp2, bHp = fb.createFilterBank(fs,plotFilter=plotFilter)
    lp_Audio, bp1_Audio, bp2_Audio, hp_Audio = fb.applyFilters(audio, bLp, bBp1, bBp2, bHp)

    # if playAudio:
    #     print('Playing original:')
    #     basic_audio_proc.play_audio(audio,fs)
    #     print('Playing lowpass signal:')
    #     basic_audio_proc.play_audio(lp_Audio,fs)
    #     print('Playing bandpass 1:')
    #     basic_audio_proc.play_audio(bp1_Audio * 10,fs)
    #     print('Playing bandpass 2:')
    #     basic_audio_proc.play_audio(bp2_Audio * 10, fs)
    #     print('Playing highpass:')
    #     basic_audio_proc.play_audio(hp_Audio * 10, fs)

    if plotAudio:
        plt.plot(audio)
        plt.plot(lp_Audio)
        plt.plot(bp1_Audio)
        plt.plot(bp2_Audio)
        plt.plot(hp_Audio)
        plt.title('Audio input and outputs of the filter banks (not downsampled)')
        plt.xlabel('Time in Samples')
        plt.legend(('original','lowpass','bandpass1','bandpass2','highpass'))
        plt.show()

    return lp_Audio, bp1_Audio, bp2_Audio, hp_Audio


def applySynthesisFilterBank(lp_Audio, bp1_Audio, bp2_Audio, hp_Audio, fs, plotFilter=False, plotAudio=False, playAudio=False):
    # uses a filter bank with 4 subbands to decompose a signal into 4 parts (lowpass, 2 bandpass and highpass signal)

    bLp, bBp1, bBp2, bHp = fb.createFilterBank(fs,plotFilter=plotFilter)
    lp_Audio, bp1_Audio, bp2_Audio, hp_Audio = fb.applyFiltersSynthesis(lp_Audio, bp1_Audio, bp2_Audio, hp_Audio, bLp, bBp1, bBp2, bHp)

    reconAudio = lp_Audio + bp1_Audio + bp2_Audio + hp_Audio


    if plotAudio:
        plt.plot(lp_Audio)
        plt.plot(bp1_Audio)
        plt.plot(bp2_Audio)
        plt.plot(hp_Audio)
        plt.title('Audio input and outputs of the filter banks (not downsampled)')
        plt.xlabel('Time in Samples')
        plt.legend(('original','lowpass','bandpass1','bandpass2','highpass'))
        plt.show()

    return reconAudio


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
