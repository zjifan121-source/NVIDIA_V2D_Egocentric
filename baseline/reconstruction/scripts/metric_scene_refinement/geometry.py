"""Camera-space derivatives shared by the refinement modules."""
import numpy as np

def skew(q):
    o = np.zeros((len(q), 3, 3))
    o[:, 0, 1] = -q[:, 2]
    o[:, 0, 2] = q[:, 1]
    o[:, 1, 0] = q[:, 2]
    o[:, 1, 2] = -q[:, 0]
    o[:, 2, 0] = -q[:, 1]
    o[:, 2, 1] = q[:, 0]
    return o

def projection(q, K):
    z = q[:, 2]
    J = np.zeros((len(q), 2, 3))
    J[:, 0, 0] = K[0, 0] / z
    J[:, 1, 1] = K[1, 1] / z
    J[:, 0, 2] = -K[0, 0] * q[:, 0] / z ** 2
    J[:, 1, 2] = -K[1, 1] * q[:, 1] / z ** 2
    uv = (q @ K.T)[:, :2] / z[:, None]
    return (uv, J)
