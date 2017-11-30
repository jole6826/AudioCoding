import encframework as enc
import decframework as dec
import os
import matplotlib.pyplot as plt
import basic_audio_proc as bap


########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
dumpHuffman = True

quantized_audio, norm_audio, fs, dump_fname = enc.read_and_quantize('imagine_Dragons_Thunder_short_32khz.wav', length_segment, channel, n_bits=8)

lpAudio, bp1Audio, bp2Audio, hpAudio = enc.applyFilterBank(quantized_audio,fs,plotFilter=True,plotAudio=True,playAudio=True)
