import numpy as np


def spectral_centroid(fixed_frames):
    """
        This function calculates mean spectral centroid amongst all frames
    :param fixed_frames: this is the numpy array of fixed_frames
    :return: return mean value of spectral centroid
    """
    n = len(fixed_frames)
    result = []
    for index in range(0, n):
        result.append(fixed_frames[index].spectrum().centroid())
    return np.mean(np.array(result), axis=0)
