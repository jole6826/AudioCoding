import basic_audio_proc
import numpy as np
import warnings
import pickle

# read wav segment
[norm_audio, raw_audio, org_dtype] = basic_audio_proc.read_segment('Track48.wav', 5, 0);

encoded16bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 16);
encoded8bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 8);


# TODO play audio segment and quantized versions
print("Playing 8bit quantized:")
basic_audio_proc.sound(encoded8bit_audio, 44100,8);
print("Playing 16bit quantized:")
basic_audio_proc.sound(encoded16bit_audio, 44100,16);
# TODO plot both channels

#pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1);
#pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)
