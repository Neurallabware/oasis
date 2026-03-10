"""Parameter estimation for OASIS deconvolution.

Functions for estimating noise standard deviation and AR model coefficients
from fluorescence traces.
"""

import numpy as np
import scipy.signal
import scipy.linalg


def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5],
                        method='logmexp', lags=5, fudge_factor=1.):
    """Estimate noise standard deviation and AR coefficients.

    Args:
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities.
        p: int
            Order of AR system.
        sn: float, optional
            Noise standard deviation. Estimated if not provided.
        g: array-like, optional
            AR coefficients. Estimated if not provided.
        range_ff: list of two floats
            Frequency range (x Nyquist rate) for averaging noise PSD.
        method: str
            Method of averaging: 'mean', 'median', or 'logmexp' (default).
        lags: int
            Number of additional lags for autocovariance computation.
        fudge_factor: float
            Shrinkage factor to reduce bias (0 < fudge_factor <= 1).

    Returns:
        g: np.ndarray
            Estimated AR coefficients.
        sn: float
            Estimated noise standard deviation.
    """
    if sn is None:
        sn = GetSn(fluor, range_ff, method)

    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn


def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.):
    """Estimate AR model parameters through the autocovariance function.

    Args:
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities.
        p: int
            Order of AR system.
        sn: float, optional
            Noise standard deviation. Estimated if not provided.
        lags: int
            Number of additional lags for autocovariance computation.
        fudge_factor: float
            Shrinkage factor to reduce bias (0 < fudge_factor <= 1).

    Returns:
        g: np.ndarray
            Estimated coefficients of the AR process.
    """
    if sn is None:
        sn = GetSn(fluor)

    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(c=np.ravel(xc[lags + np.arange(lags)]),
                              r=np.ravel(xc[lags + np.arange(p)])) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[lags + 1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    # Static seed for reproducibility while maintaining some variability
    np.random.seed(45)
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """Estimate noise power through the power spectral density.

    Estimates noise standard deviation over the range of large frequencies.

    Args:
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities.
        range_ff: list of two floats
            Frequency range (x Nyquist rate) for averaging.
        method: str
            Method of averaging: 'mean', 'median', or 'logmexp' (default).

    Returns:
        sn: float
            Noise standard deviation.
    """
    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]

    if method == 'mean':
        return np.sqrt(np.mean(Pxx_ind / 2))
    elif method == 'median':
        return np.sqrt(np.median(Pxx_ind / 2))
    elif method == 'logmexp':
        return np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    else:
        raise ValueError(f'Invalid method: {method}. Use mean, median, or logmexp.')


def axcov(data, maxlag=5):
    """Compute the autocovariance of data at lag = -maxlag:0:maxlag.

    Args:
        data: np.ndarray
            Array containing fluorescence data.
        maxlag: int
            Number of lags to use in autocovariance calculation.

    Returns:
        axcov: np.ndarray
            Autocovariances computed from -maxlag:0:maxlag.
    """
    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov / T)


def nextpow2(value):
    """Find exponent such that 2^exponent >= abs(value).

    Args:
        value: int

    Returns:
        exponent: int
    """
    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent
