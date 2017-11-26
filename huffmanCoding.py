import scipy.io.wavfile as wv
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import bitstring as bits


def huffmanEncoder(audio,cBook):
    '''huffmannCoding performs huffman coding on 8 bit quantized audio data
    Input:
        audio -     int8 quantized data (maybe in blocks of size 1024 or similar)
        cBook -     huffman codebook that stores the respective "bitstream" for each possible symbol of audio
                    in data range of int8

    Output:
        coded -     string (?, decide data structure) containing binary values
        '''

    nSamples = audio.shape[0]
    maxVal = float(np.max(audio))
    minVal = float(np.min(audio))
    min_dtype = float(np.iinfo(np.int8).min)
    max_dtype = float(np.iinfo(np.int8).max)


    if maxVal > max_dtype or minVal < min_dtype:
        print ("data contains values outside of int8 range")

    coded = '0b'

    for samp in audio:
        # apply codebook for each sample
        binSymbol = cBook[str(np.float(samp))]
        #binSymbol = cBook[str(np.float(audio[ix]))]
        coded += binSymbol
    bitStreamOut = bits.BitArray(bin=coded)
    return bitStreamOut

def huffmanDecoder(bitstream,cBook):
    bits = bitstream.bin
    inv_cBook = {bit_code: value for value, bit_code in cBook.iteritems()}
    nSamples_max = len(bits) / min([len(key) for key in inv_cBook.keys()]) + 1
    decoded = np.zeros((nSamples_max, 1))
    
    idx_sample = 0
    while (len(bits) > 0):
        for idx_bitstream in np.arange(1, len(bits)):
            code_to_find = bits[0:idx_bitstream]
            try:
                decoded_val = inv_cBook[code_to_find]
                # this is only executed if values was found
                decoded[idx_sample] = decoded_val
                idx_sample += 1
                bits = bits[idx_bitstream:] #delete the found bits
                break #break for loop and start new one
            except KeyError:
                # if code not found and no more bits in stream, end of bitstream is reached (zero padded...)
                if idx_bitstream == len(bits) - 1:
                    bits = []
                    break
                # if code until now not found, just go on (the beauty of prefix codes)
                pass 
            
    return decoded

def createHuffmanCodebook(audio):
    # function to create huffman codebook using probabilities of each symbol
    # p is an array that contains the symbols p[:,0] and probabilities p[:,1]
    # based on https://gist.github.com/mreid/fdf6353ec39d050e972b

    org_dtype = audio.dtype
    min_dtype = float(np.iinfo(org_dtype).min)
    max_dtype = float(np.iinfo(org_dtype).max)
    nSamples = audio.shape[0]
    nPossibleVals = int(max_dtype - min_dtype + 1)
    hist, __ = np.histogram(audio, nPossibleVals, [min_dtype, max_dtype])
    prob = np.float32(hist) / nSamples
    prob = prob[hist != 0]
    vals = np.linspace(min_dtype,max_dtype,num=256)[hist != 0]
    p = zip(map(str,vals),prob)
    p = dict(p)

    c = huffmanCb(p)
    return c

def createHuffmanCodebookFromHist(hist,vals):
    # function to create huffman codebook using probabilities of each symbol
    # p is an array that contains the symbols p[:,0] and probabilities p[:,1]
    # based on https://gist.github.com/mreid/fdf6353ec39d050e972b

    nSamples = hist.size
    prob = np.float32(hist) / nSamples
    prob = prob[hist != 0]
    vals = vals[hist != 0]
    p = zip(map(str,vals),prob)
    p = dict(p)

    c = huffmanCb(p)
    return c

def huffmanCb(p):
    '''used recursively to create hufmann codebook from input data
    input:
        - p is a dictionary containing the signal values as str
        and the corresponding probability'''

    #assert (sum(p.values()) == 1.0)  # make sure probabilities add up to 1.0
    # Base case of only two symbols, assign 0 or 1 arbitrarily

    if (len(p) == 2):
        return dict(zip(p.keys(), ['0', '1']))

    # Create a new distribution by merging lowest prob. pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = huffmanCb(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + '0', ca1a2 + '1'

    return c

def lowest_prob_pair(p):
    '''Return pair of symbols from distribution p with lowest probabilities.'''
    assert(len(p) >= 2) # Ensure there are at least 2 symbols in the dist.

    sorted_p = sorted(p.items(), key=lambda (i,pi): pi)
    return sorted_p[0][0], sorted_p[1][0]
