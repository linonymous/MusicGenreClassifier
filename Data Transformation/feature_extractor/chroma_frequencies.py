import numpy as np


def chroma(fixed_frames):
    """
        This function calculates mean of the chroma frequencies across fixed_frames
    :param fixed_frames: numpy array of frames
    :return: numpy array of mean chroma frequencies across frames
    """
    n = len(fixed_frames)
    result = []
    for index in range(0, n):
        result.append(fixed_frames[index].spectrum().chroma())
    return np.mean(np.array(result)[:], axis=0)