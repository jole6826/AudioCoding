import scipy.io.wavfile as wv
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import bitstring as bits

class HufmannTree(object):
    def __init__(self, node0, node1, data, p):
        self.node0 = node0
        self.node1 = node1
        self.data = data
        self.p = p

def huffmanEncoder(audio,cBook):
    '''huffmannCoding performs huffman coding on 8 bit quantized audio data
    Input:
        audio -     int8 quantized data (maybe in blocks of size 1024 or similar)
        cBook -     huffman codebook that stores the respective "bitstream" for each possible symbol of audio
                    in data range of int8

    Output:
        coded -     string (?, decide data structure) containing binary values
        '''

    maxVal = float(np.max(audio))
    minVal = float(np.min(audio))
    min_dtype = float(np.iinfo(np.int8).min)
    max_dtype = float(np.iinfo(np.int8).max)


    if maxVal > max_dtype or minVal < min_dtype:
        print ("data contains values outside of int8 range")

    coded = ''

    for samp in audio:
        # apply codebook for each sample
        binSymbol = cBook[str(np.float(samp))]
        #binSymbol = cBook[str(np.float(audio[ix]))]
        coded += binSymbol
    # bitstream pads zeros to nearest number of bytes -> we have to know how many to throw away: maximum 7 bits
    # we have to transmit the number of padded zeros: we need 3 more bits for that (2^3 = 8)
    n_zero_pad = 8 - ((len(coded)+3) % 8)
    padded_code = '{0:03b}'.format(n_zero_pad) + coded
    
    bitStreamOut = bits.BitArray(bin=padded_code)
    return bitStreamOut

def fastHuffmanDecoder(bitstream, cb_tree):
    decoded = np.zeros(len(bitstream))
     
    idx_sample = 0
    cb_tree_iterator = cb_tree
    for bit in bitstream:
        if (bit == '0') & (cb_tree_iterator.node0 != None) & (cb_tree_iterator.node1 != None):
            cb_tree_iterator = cb_tree_iterator.node0
        elif (bit == '1') & (cb_tree_iterator.node0 != None) & (cb_tree_iterator.node1 != None):
            cb_tree_iterator = cb_tree_iterator.node1

        if (cb_tree_iterator.node0 == None) & (cb_tree_iterator.node1 == None):
            decoded[idx_sample] = cb_tree_iterator.data
            idx_sample += 1
            cb_tree_iterator = cb_tree

    return decoded[0:idx_sample] 

''' this is way too slow...        
def huffmanDecoder(bitstream, cBook):
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
'''

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
    probs = np.float32(hist) / nSamples
    probs = probs[hist != 0]
    vals = np.linspace(min_dtype,max_dtype,num=nPossibleVals)[hist != 0]
    
    init_tree = [HufmannTree(None, None, value, probs[idx]) for idx, value in enumerate(vals)]

    codebook_tree = build_huffman_tree(init_tree)
    codebook = tree2codebook(codebook_tree, '')
    return codebook, codebook_tree

def build_huffman_tree(init_tree):
    while len(init_tree) >= 2:
        lowest_left, lowest_right = lowest_prob_nodes(init_tree)
        init_tree.remove(lowest_left)
        init_tree.remove(lowest_right)
        init_tree.append(HufmannTree(lowest_left, lowest_right, str(lowest_left.data) + str(lowest_right.data), lowest_left.p + lowest_right.p))
    return init_tree[0]

def tree2codebook(huffman_tree, current_code):
    if (huffman_tree.node0 != None) & (huffman_tree.node1 != None):
        cb_left = tree2codebook(huffman_tree.node0, current_code + '0')
        cb_right = tree2codebook(huffman_tree.node1, current_code + '1')
        cb = cb_left.copy()
        cb.update(cb_right)
        return cb
    else:
        return dict({str(huffman_tree.data): current_code})   
   
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

def lowest_prob_nodes(huffman_tree):
    sorted_tree = sorted(huffman_tree, key=lambda node: node.p)
    return sorted_tree[0], sorted_tree[1]
    
def lowest_prob_pair(p):
    '''Return pair of symbols from distribution p with lowest probabilities.'''
    assert(len(p) >= 2) # Ensure there are at least 2 symbols in the dist.

    sorted_p = sorted(p.items(), key=lambda (i,pi): pi)
    return sorted_p[0][0], sorted_p[1][0]
