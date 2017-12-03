import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def createFilterBank(fs, plotFilter=False):
    # uses a filter bank with 4 subbands to decompose a signal into 4 parts (lowpass, 2 bandpass and highpass signal)
    nSubbands = 4 # subbands
    fNy = fs/2.0
    bandwidth = fNy/nSubbands
    lpLBound = 0 # lower boundary for lowpass
    lpUBound = bandwidth # upper boundary for lowpass
    bp1LBound = bandwidth
    bp1UBound = 2* bandwidth
    bp2LBound = 2*bandwidth
    bp2UBound = 3*bandwidth
    hpLBound = 3*bandwidth
    hpUBound = fNy

    # create filter coefficents for subband filtering
    bLp=signal.remez(24*nSubbands,[lpLBound,lpUBound,lpUBound+500,fNy],[1,0],[1,100],Hz=32000, type='bandpass')
    bBp1=signal.remez(24*nSubbands,[0,bp1LBound-500,bp1LBound,bp1UBound,bp1UBound+500,fNy],[0,1,0],[100,1,100],Hz=32000, type='bandpass')
    bBp2=signal.remez(24*nSubbands,[0,bp2LBound-500,bp2LBound,bp2UBound,bp2UBound+500,fNy],[0,1,0],[100,1,100],Hz=32000, type='bandpass')
    bHp=signal.remez(24*nSubbands,[0,hpLBound-500,hpLBound,hpUBound],[0,1],[100,1],Hz=32000, type='bandpass')

    if plotFilter:
        plotFilterbank(bLp,bBp1,bBp2,bHp)

    return bLp, bBp1, bBp2, bHp


def applyFilters(audio,bLp,bBp1,bBp2, bHp):
    origType = audio.dtype
    lp_Audio = signal.lfilter(bLp, 1, audio)
    bp1_Audio = signal.lfilter(bBp1, 1, audio)
    bp2_Audio = signal.lfilter(bBp2, 1, audio)
    hp_Audio = signal.lfilter(bHp, 1, audio)
    np.clip(lp_Audio, -128, 127, out=lp_Audio)
    np.clip(bp1_Audio, -128, 127, out=bp1_Audio)
    np.clip(bp2_Audio, -128, 127, out=bp2_Audio)
    np.clip(hp_Audio, -128, 127, out=hp_Audio)


    lp_Audio = lp_Audio.astype(origType)
    bp1_Audio = bp1_Audio.astype(origType)
    bp2_Audio = bp2_Audio.astype(origType)
    hp_Audio = hp_Audio.astype(origType)

    return lp_Audio, bp1_Audio, bp2_Audio, hp_Audio


def applyFiltersSynthesis(lp_Audio, bp1_Audio, bp2_Audio, hp_Audio, bLp, bBp1, bBp2, bHp):
    origType = lp_Audio.dtype
    lp_Audio = signal.lfilter(bLp, 1, lp_Audio)
    bp1_Audio = signal.lfilter(bBp1, 1, bp1_Audio)
    bp2_Audio = signal.lfilter(bBp2, 1, bp2_Audio)
    hp_Audio = signal.lfilter(bHp, 1, hp_Audio)
    np.clip(lp_Audio, -128, 127, out=lp_Audio)
    np.clip(bp1_Audio, -128, 127, out=bp1_Audio)
    np.clip(bp2_Audio, -128, 127, out=bp2_Audio)
    np.clip(hp_Audio, -128, 127, out=hp_Audio)

    lp_Audio = lp_Audio.astype(origType)
    bp1_Audio = bp1_Audio.astype(origType)
    bp2_Audio = bp2_Audio.astype(origType)
    hp_Audio = hp_Audio.astype(origType)

    return lp_Audio, bp1_Audio, bp2_Audio, hp_Audio


def plotFilterbank(bLp,bBp1,bBp2,bHp):
    plt.plot (bLp)
    plt.plot (bBp1)
    plt.plot (bBp2)
    plt.plot (bHp)
    plt.title('Filter Impulse Response')
    plt.xlabel('Time in Samples')
    plt.legend(('low pass', 'bandpass 1', 'bandpass 2','highpass'))
    plt.show()
    w,H=signal.freqz(bLp)
    plt.plot(w,20*np.log10(abs(H)+1e-6))
    w,H=signal.freqz(bBp1)
    plt.plot(w,20*np.log10(abs(H)+1e-6))
    w,H=signal.freqz(bBp2)
    plt.plot(w,20*np.log10(abs(H)+1e-6))
    w,H=signal.freqz(bHp)
    plt.plot(w,20*np.log10(abs(H)+1e-6))
    plt.title('Filter Magnitude Frequency Response')
    plt.legend(('low pass', 'bandpass 1', 'bandpass 2','highpass'))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('dB')
    plt.show()
