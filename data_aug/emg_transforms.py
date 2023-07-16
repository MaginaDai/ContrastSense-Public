import numpy as np
import torch
from scipy.interpolate import interp1d

class EMGNoise(object):
    """ Add noise to the EMG sensors reading

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
            noise = np.random.normal(self.mu, self.var, (sensor.shape[0], sensor.shape[1], sensor.shape[2]))
            sensor += noise
        return sensor
    
class EMGScale(object):
    """ Rescale the EMG sensors reading

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

    
class EMGNegated(object):
    """ Negate the EMG sensors reading

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


class EMGHorizontalFlip(object):
    """ Flip the EMG sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sensor):
        if np.random.random() < self.p:
            sensor = sensor[:, ::-1, :]
        return sensor

class EMGTimeWarp(object):
    """ locally prolong the EMG sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p, pos=None):
        self.p = p
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if np.random.random() < self.p:
            length = sensor.shape[1]
            low = int(np.random.uniform(low=0, high=sensor.shape[1]-10))
            high = int(np.random.uniform(low=low+2, high=sensor.shape[1]-5))
            new_high = int(np.random.uniform(low=high+3, high=sensor.shape[1]-1))

            new_length = new_high - low
            pos = np.arange(low, high)
            pos_new = np.linspace(low, high-1, new_length)

            f_sensor = interp1d(pos, sensor[:, pos, :], axis=1, kind='linear')

            sensor_new = f_sensor(pos_new)

            sensor_scale = np.concatenate((sensor[:, :low, :], sensor_new, sensor[:, high:, :]), axis=1)

            sensor = sensor_scale[:, :length, :]

        return sensor


class EMGToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sensor):
        return torch.from_numpy(sensor.copy().astype('float32'))