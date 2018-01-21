from spectral_rolloff import spectral_rolloff
from spectral_centroid import spectral_centroid
from chroma_frequencies import chroma
from zero_crossing_rate import zero_crossing_rate
import sys
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
sys.path.append('C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\pymir')
from pymir import AudioFile
import numpy as np


def feature_calculator(file_path):
    """
    This function returns the vector containing 13 mfcc values, 12 chroma values and single values for spectral centroid, rolloff and zcr
    :param file_path: path for the wavFile of music
    :return: returns the feature vector
    """
    wavData = AudioFile.open(file_path)
    fixedFrames = wavData.frames(160)
    centroid_res = spectral_centroid(fixedFrames)
    rolloff_res = spectral_rolloff(fixedFrames)
    chroma_res = chroma(fixedFrames)
    zcr_res = zero_crossing_rate(fixedFrames)
    sample_rate, X = scipy.io.wavfile.read(file_path)
    ceps, mspec, spec = mfcc(X)
    num = len(ceps)
    mfcc_res = np.array(np.mean(np.array(ceps[num * 3 / 10: num * 7 / 10]), axis=0))
    return np.append(np.append(np.append(np.append(zcr_res, chroma_res), rolloff_res), centroid_res), mfcc_res)

