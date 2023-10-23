import numpy as np

def phase(z: complex):
    "Return a phase of complex input mapped to [0, 2*pi) interval"

    phi = np.angle(z)
    # convert phase angles from [pi, -pi) which is the output of numpy.angle to [0, 2*pi)
    phi[phi<0] = 2*np.pi + phi[phi<0]
    return phi

def sawtooth(x, log_base=None):
    # setting numpy to ignore divide by zero and invalid input errors locally
    with np.errstate(divide='ignore', invalid='ignore'):
        if log_base is not None:
            x = np.log(x)
            x = x / np.log(log_base) # converting log from natural to log_base
        return np.ceil(x) - x
