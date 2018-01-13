import numpy as np


def spectral_rolloff(fixed_frames):
    """
        This function calculates spectral roll off of all frames and returns the mean value
    :param fixed_frames: numpy array of fixed frames
    :return: returns the mean value of spectral roll off
    """
    n = len(fixed_frames)
    result = []
    for index in range(0, n):
        result.append(fixed_frames[index].spectrum().rolloff())
    return np.mean(np.array(result), axis=0)