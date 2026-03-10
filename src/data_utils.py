"""Synthetic data generation for testing OASIS deconvolution.

Generates fluorescence traces from homogeneous Poisson process spike trains
convolved with AR(1) or AR(2) calcium dynamics plus Gaussian noise.
"""

import numpy as np


def gen_data(g=[.95], sn=.2, T=1000, framerate=30, firerate=.5, b=10, N=1, seed=0):
    """Generate synthetic calcium imaging data from a homogeneous Poisson process.

    Args:
        g: list of floats
            Parameter(s) of the AR(p) process that models the fluorescence impulse response.
            Length 1 for AR(1), length 2 for AR(2).
        sn: float
            Noise standard deviation.
        T: int
            Duration (number of time bins).
        framerate: float
            Frame rate in Hz.
        firerate: float
            Neural firing rate in Hz.
        b: float
            Baseline fluorescence.
        N: int
            Number of generated traces.
        seed: int
            Seed of random number generator.

    Returns:
        Y: np.ndarray, shape (N, T)
            Noisy fluorescence data.
        trueC: np.ndarray, shape (N, T)
            Ground truth calcium traces (without noise).
        trueS: np.ndarray, shape (N, T)
            Ground truth spike trains.
    """
    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueC = trueS.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            trueC[:, i] += g[0] * trueC[:, i - 1] + g[1] * trueC[:, i - 2]
        else:
            trueC[:, i] += g[0] * trueC[:, i - 1]
    Y = b + trueC + sn * np.random.randn(N, T)
    return Y, trueC, trueS
