import basic_audio_proc
import numpy as np
import matplotlib.pyplot as plt
import pickle

########################################
# Adjust here before running the script#
########################################
channel = 1 # L = 0, R = 1
length_segment = 8 # in seconds
plot_audio = False
play_audio = False
dumpFiles = False


# read wav segment
[norm_audio, raw_audio, org_dtype, fs] = basic_audio_proc.read_segment('Track48.wav', length_segment, channel)


encoded16bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 16)
encoded8bit_audio = basic_audio_proc.quantize(norm_audio, org_dtype, 8)

# processing and coding goes here
# creating codebook like: https://gist.github.com/mreid/fdf6353ec39d050e972b
codebookTest = np.linspace(-128,127,256,dtype=np.int8)
codebookTest2 = ["" for x in range(256)]
for i in xrange(0,256,1):
    codebookTest2[i] = "{0:08b}".format(codebookTest[i] & 0xff)

codebookTestJoined = np.empty([256,2],dtype=np.object)
codebookTestJoined[:,0] = codebookTest
codebookTestJoined [:,1] = codebookTest2

signaltest = np.array([-127,-120,-50,40,0,32,1,127],dtype=np.int8)
bitsOut = basic_audio_proc.huffmanCoding(signaltest,codebookTestJoined)

# save files as binary data
if dumpFiles:
    pickle.dump(encoded16bit_audio, open('encoded16bit.bin', 'wb'), 1)
    pickle.dump(encoded8bit_audio, open('encoded8bit.bin', 'wb'), 1)


if play_audio:
    print("Playing original:")
    basic_audio_proc.play_audio(raw_audio, fs)
    print("Playing 16bit quantized:")
    basic_audio_proc.play_audio(encoded16bit_audio, fs)
    print("Playing 8bit quantized:")
    basic_audio_proc.play_audio(encoded8bit_audio, fs)

if plot_audio:
    t_axis = np.linspace(0, norm_audio.shape[0]-1, norm_audio.shape[0]) / float(fs)
    plt.plot(t_axis, norm_audio, lw = 0.2)
    plt.title('Original Audio')
    plt.xlabel('t in seconds')
    plt.show()

    plt.plot(t_axis, encoded16bit_audio, lw = 0.2)
    plt.title('16bit encoded Audio')
    plt.xlabel('t in seconds')
    plt.show()

    plt.plot(t_axis, encoded8bit_audio, lw = 0.2)
    plt.title('8bit encoded Audio')
    plt.xlabel('t in seconds')
    plt.show()

    # block into 1024 sample blocks
    framed_audio = basic_audio_proc.frame_audio(norm_audio, 1024)
    fft_audio = np.abs(np.fft.fft(framed_audio, axis=0))
    frqz_resp = 20*np.log10(fft_audio)

    plt.plot(frqz_resp[0,:], 'b', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[1,:], 'r', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[2,:], 'g', lw=0.4, alpha=0.5)
    plt.plot(frqz_resp[3,:], 'y', lw=0.4, alpha=0.5)
    plt.xlabel('f in frequency bins')
    plt.ylabel('Magnitude in dB')
    plt.show()
