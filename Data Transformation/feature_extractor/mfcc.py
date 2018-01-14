import numpy as np


def mfcc(fixed_frames):
    """
        This function calculates mfcc using pymir library
    :param fixed_frames: numpy array of frames
    :return: 13 * 1 dimensional array of mean values of numpy
    """
    n = len(fixed_frames)
    result = []
    for index in range(0, n):
        result.append(fixed_frames[index].spectrum().mfcc2(13))
    return np.mean(np.array(result)[:], axis=0)