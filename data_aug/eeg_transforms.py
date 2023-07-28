import numpy as np
import torch
from scipy.interpolate import interp1d

class EEG_Noise(object):
    """ Add noise to the EEG sensors reading

    Args:
        p: transformation possibility
        var: variance of data
    """

    def __init__(self, var, p):
        self.p = p
        self.mu = 0
        self.var = var

    def __call__(self, sensor, pos=None):
        if np.random.random() < self.p:
            noise = np.random.normal(self.mu, self.var, (sensor.shape[0], sensor.shape[1]))
            sensor += noise
        return sensor

class EEG_cutout(object):
    """ cutout the EEG sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sensor):
        if np.random.random() < self.p:
            low = int(np.random.uniform(low=0, high=sensor.shape[1]-2))
            high = int(np.random.uniform(low=low+1, high=sensor.shape[1]-1))
            sensor[low:high, :] = 0
        return sensor
    
class EEG_delay(object):
    """ Add delay to the EEG sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sensor):
        if np.random.random() < self.p:
            start = int(np.random.uniform(low=0, high=40))  # based on the paper SACL
            sensor[start:, :] = sensor[:(sensor.shape[0] - start), :]
            sensor[:start, :] = 0
        return sensor
    
class EEG_dropout(object):
    """ Add dropout to the EEG sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sensor):
        pos = np.random.rand(sensor.shape[2])
        idx = pos < self.p
        sensor[:, :, idx] = 0
        return sensor


class EEG_Scale(object):
    """ Rescale the EEG sensors reading

    Args:
        scale: the scaling range for all axis
        p: transformation possibility
    """

    def __init__(self, scale, p):
        self.scale = scale
        self.p = p

    def __call__(self, sensor):
        if np.random.random() < self.p:
            K = np.diag(np.random.uniform(low=self.scale[0], high=self.scale[1], size=sensor.shape[-1]))
            sensor = np.dot(sensor, K)
        return sensor


class EEG_Negated(object):
    """ Negate the EEG sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sensor):
        if np.random.random() < self.p:
            R = np.eye(sensor.shape[-1])
            for i in range(sensor.shape[-1]):
                if np.random.random() < 0.5:
                    R[i, i] = -1
            sensor = np.dot(sensor, R)
        return sensor

    

class EEGToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sensor):
        return torch.from_numpy(sensor.copy().astype('float32'))