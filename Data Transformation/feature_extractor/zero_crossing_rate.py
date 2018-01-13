import numpy as np


def zero_crossing_rate(fixed_frames):
    """
        This calculates mean zero crossing rate amongst all frames
    :param fixed_frames: the numpy array of fixed frames
    :return: return mean value of zero crossing rate
    """
    n = len(fixed_frames)
    result = []
    for index in range(0, n):
        result.append(fixed_frames[index].zcr())
    return np.mean(np.array(result), axis=0)