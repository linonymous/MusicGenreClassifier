from spectral_rolloff import spectral_rolloff
from mfcc import mfcc
from spectral_centroid import spectral_centroid
from chroma_frequencies import chroma
from zero_crossing_rate import zero_crossing_rate
import sys
sys.path.append('/home/mahesh/Mahesh/MusicGenreClassifier/Data Transformation/pymir')
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
    mfcc_res = mfcc(fixedFrames)
    centroid_res = spectral_centroid(fixedFrames)
    rolloff_res = spectral_rolloff(fixedFrames)
    chroma_res = chroma(fixedFrames)
    zcr_res = zero_crossing_rate(fixedFrames)
    return np.append(np.append(np.append(np.append(zcr_res, chroma_res), rolloff_res), centroid_res), mfcc_res)

