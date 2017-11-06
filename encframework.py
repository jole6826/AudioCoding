import basic_audio_proc
import numpy as np
import warnings
import pickle

# read wav segment
[norm_audio, raw_audio, org_dtype] = basic_audio_proc.read_segment('Track48.wav', 8, 0);

test = True;

encoded16bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 16);
encoded8bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 8);

pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1);
pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)
