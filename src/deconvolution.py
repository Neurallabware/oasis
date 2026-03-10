"""OASIS deconvolution: extract neural activity from fluorescence traces.

Standalone implementation of constrained_foopsi using the OASIS algorithm
for sparse nonneg deconvolution of calcium imaging data.

References:
    Friedrich J and Paninski L, NIPS 2016
    Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    Pnevmatikakis et al. 2016. Neuron
"""

import numpy as np
from math import log, sqrt, exp

from .estimation import estimate_parameters, estimate_time_constant


def constrained_foopsi(fluor, bl=None, c1=None, g=None, sn=None, p=None,
                       method_deconvolution='oasis', bas_nonneg=True,
                       noise_range=[.25, .5], noise_method='logmexp', lags=5,
                       fudge_factor=1., verbosity=False, optimize_g=0,
                       s_min=None, **kwargs):
    """Infer the most likely discretized spike train underlying a fluorescence trace.

    Uses noise constrained deconvolution via the OASIS algorithm.

    Args:
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.
        bl: float, optional
            Fluorescence baseline value. Estimated from data if not given.
        c1: float, optional
            Value of calcium at time 0.
        g: list or float, optional
            Parameters of the AR process. Estimated from data if not given.
        sn: float, optional
            Standard deviation of the noise distribution. Estimated if not given.
        p: int
            Order of the autoregression model (1 or 2).
        method_deconvolution: str, optional
            Only 'oasis' is supported in this standalone module.
        bas_nonneg: bool
            Baseline strictly non-negative.
        noise_range: list of two floats
            Frequency range for averaging noise PSD.
        noise_method: str
            Method of averaging noise PSD.
        lags: int
            Number of lags for estimating time constants.
        fudge_factor: float
            Fudge factor for reducing time constant bias.
        verbosity: bool
            Display optimization details.
        optimize_g: int, optional
            Number of large, isolated events to consider for optimizing g.
            If 0 (default) the provided or estimated g is not further optimized.
        s_min: float, optional
            Minimal non-zero activity within each bin (minimal 'spike size').
            For negative values: threshold = abs(s_min) * sn * sqrt(1-g).
            If None (default): standard L1 penalty.
            If 0: threshold determined automatically such that RSS <= sn^2 T.

    Returns:
        c: np.ndarray
            The inferred denoised fluorescence signal at each time-bin.
        bl: float
            Estimated baseline.
        c1: float
            Estimated initial calcium value.
        g: np.ndarray
            Estimated AR coefficients.
        sn: float
            Estimated noise standard deviation.
        sp: np.ndarray
            Discretized deconvolved neural activity (spikes).
        lam: float
            Regularization parameter.

    Raises:
        ValueError: If p is not specified or if p not in {0, 1, 2}.
    """
    if method_deconvolution != 'oasis':
        raise ValueError(
            f"Only method_deconvolution='oasis' is supported in this standalone module. "
            f"Got '{method_deconvolution}'."
        )

    if p is None:
        raise ValueError("You must specify the value of p")

    if g is None or sn is None:
        g, sn = estimate_parameters(fluor, p=p, sn=sn, g=g, range_ff=noise_range,
                                    method=noise_method, lags=lags, fudge_factor=fudge_factor)
    lam = None
    if p == 0:
        c1 = 0
        g = np.array(0)
        bl = 0
        c = np.maximum(fluor, 0)
        sp = c.copy()

    elif p == 1:
        from .oasis import constrained_oasisAR1
        penalty = 1 if s_min is None else 0
        if bl is None:
            c, sp, bl, g, lam = constrained_oasisAR1(
                fluor.astype(np.float32), g[0], sn, optimize_b=True, b_nonneg=bas_nonneg,
                optimize_g=optimize_g, penalty=penalty, s_min=0 if s_min is None else s_min)
        else:
            c, sp, _, g, lam = constrained_oasisAR1(
                (fluor - bl).astype(np.float32), g[0], sn, optimize_b=False, penalty=penalty,
                s_min=0 if s_min is None else s_min)

        c1 = c[0]
        # remove initial calcium to align with the other foopsi methods
        c -= c1 * g**np.arange(len(fluor))
        g = np.ravel(g)

    elif p == 2:
        penalty = 1 if s_min is None else 0
        if bl is None:
            c, sp, bl, g, lam = constrained_oasisAR2(
                fluor.astype(np.float32), g, sn, optimize_b=True, b_nonneg=bas_nonneg,
                optimize_g=optimize_g, penalty=penalty, s_min=s_min)
        else:
            c, sp, _, g, lam = constrained_oasisAR2(
                (fluor - bl).astype(np.float32), g, sn, optimize_b=False,
                penalty=penalty, s_min=s_min)
        c1 = c[0]
        d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
        c -= c1 * d**np.arange(len(fluor))
        g = np.ravel(g)

    else:
        raise ValueError('OASIS is currently only implemented for p=0, p=1 and p=2')

    return c, bl, c1, g, sn, sp, lam


def _nnls(KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
    """Solve non-negative least squares problem.

    ``argmin_s || Ks - y ||_2`` for ``s>=0``

    Args:
        KK: np.ndarray, shape (n, n)
            Dot-product of design matrix K transposed and K, K'K.
        Ky: np.ndarray, shape (n,)
            Dot-product of design matrix K transposed and target vector y, K'y.
        s: np.ndarray, optional
            Initialization of deconvolved neural activity.
        mask: np.ndarray of bool, optional
            Mask to restrict potential spike times considered.
        tol: float
            Tolerance parameter.
        max_iter: int, optional
            Maximum number of iterations.

    Returns:
        s: np.ndarray
            Discretized deconvolved neural activity (spikes).
    """
    if mask is None:
        mask = np.ones(len(KK), dtype=bool)
    else:
        KK = KK[mask][:, mask]
        Ky = Ky[mask]
    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
        P = np.zeros(len(KK), dtype=bool)
    else:
        s = s[mask]
        P = s > 0
        l = Ky - KK[:, P].dot(s[P])
    i = 0
    if max_iter is None:
        max_iter = len(KK)
    for i in range(max_iter):
        w = np.argmax(l)
        P[w] = True

        try:
            mu = np.linalg.solve(KK[P][:, P], Ky[P])
        except np.linalg.LinAlgError:
            mu = np.linalg.solve(KK[P][:, P] + tol * np.eye(P.sum()), Ky[P])
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.solve(KK[P][:, P], Ky[P])
            except np.linalg.LinAlgError:
                mu = np.linalg.solve(KK[P][:, P] + tol * np.eye(P.sum()), Ky[P])
        s[P] = mu.copy()
        l = Ky - KK[:, P].dot(s[P])
        if max(l) < tol:
            break
    tmp = np.zeros(len(mask))
    tmp[mask] = s
    return tmp


def onnls(y, g, lam=0, shift=100, window=None, mask=None, tol=1e-9, max_iter=None):
    """Infer spike train using Online Non-Negative Least Squares.

    Solves the sparse non-negative deconvolution problem
    ``argmin_s 1/2|Ks-y|^2 + lam |s|_1`` for ``s>=0``

    Args:
        y: np.ndarray, shape (T,)
            Fluorescence intensities.
        g: np.ndarray
            AR process parameters.
        lam: float
            Sparsity penalty parameter lambda.
        shift: int
            Number of frames to shift window between NNLS runs.
        window: int, optional
            Window size.
        mask: np.ndarray of bool, optional
            Mask to restrict potential spike times.
        tol: float
            Tolerance parameter.
        max_iter: int, optional
            Maximum number of iterations.

    Returns:
        c: np.ndarray
            Inferred denoised fluorescence signal.
        s: np.ndarray
            Discretized deconvolved neural activity (spikes).
    """
    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    if window is None:
        w = max(200, len(g) if len(g) > 2 else
                int(-5 / log(g[0] if len(g) == 1 else
                             (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2)))
    else:
        w = window
    w = min(T, w)
    shift = min(w, shift)
    K = np.zeros((w, w))

    if len(g) == 1:  # kernel for AR(1)
        _y = y - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        h = np.exp(log(g[0]) * np.arange(w))
        for i in range(w):
            K[i:, i] = h[:w - i]

    elif len(g) == 2:  # kernel for AR(2)
        _y = y - lam * (1 - g[0] - g[1])
        _y[-2] = y[-2] - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
        r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
        if d == r:
            h = np.exp(log(d) * np.arange(1, w + 1)) * np.arange(1, w + 1)
        else:
            h = (np.exp(log(d) * np.arange(1, w + 1)) -
                 np.exp(log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]

    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        a = np.linalg.inv(K).sum(0)
        _y = y - lam * a[0]
        _y[-w:] = y[-w:] - lam * a

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, max(1, T - w), shift):
        s[i:i + w] = _nnls(KK, K.T.dot(_y[i:i + w]), s[i:i + w], mask=mask[i:i + w],
                           tol=tol, max_iter=max_iter)[:w]
        # subtract contribution of spikes already committed to
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])
    s[i + shift:] = _nnls(KK[-(T - i - shift):, -(T - i - shift):],
                          K[:T - i - shift, :T - i -
                              shift].T.dot(_y[i + shift:]),
                          s[i + shift:], mask=mask[i + shift:])
    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s


def constrained_oasisAR2(y, g, sn, optimize_b=True, b_nonneg=True, optimize_g=0, decimate=5,
                         shift=100, window=None, tol=1e-9, max_iter=1, penalty=1, s_min=0):
    """Infer spike train underlying an AR(2) fluorescence trace.

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t - g1*c_{t-1} - g2*c_{t-2} >= 0

    Args:
        y: np.ndarray
            Fluorescence intensities (with baseline already subtracted).
        g: tuple of two floats
            Parameters of the AR(2) process.
        sn: float
            Standard deviation of the noise distribution.
        optimize_b: bool
            Optimize baseline if True.
        b_nonneg: bool
            Enforce strictly non-negative baseline.
        optimize_g: int
            Number of large, isolated events for optimizing g.
        decimate: int
            Decimation factor for faster hyper-parameter estimation.
        shift: int
            Frame shift between NNLS window runs.
        window: int, optional
            Window size.
        tol: float
            Tolerance parameter.
        max_iter: int
            Maximum number of iterations.
        penalty: int
            Sparsity penalty. 1: min |s|_1, 0: min |s|_0.
        s_min: float
            Minimal non-zero activity within each bin.

    Returns:
        c: np.ndarray
            Inferred denoised fluorescence signal.
        s: np.ndarray
            Discretized deconvolved neural activity (spikes).
        b: float
            Fluorescence baseline value.
        g: tuple
            AR(2) process parameters.
        lam: float
            Sparsity penalty parameter lambda.
    """
    T = len(y)
    d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
    r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
    if window is None:
        window = int(min(T, max(200, -5 / log(d))))

    if not optimize_g:
        g11 = (np.exp(log(d) * np.arange(1, T + 1)) * np.arange(1, T + 1)) if d == r else \
            (np.exp(log(d) * np.arange(1, T + 1)) -
             np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
        g12 = np.append(0, g[1] * g11[:-1])
        g11g11 = np.cumsum(g11 * g11)
        g11g12 = np.cumsum(g11 * g12)
        Sg11 = np.cumsum(g11)
        f_lam = 1 - g[0] - g[1]
    elif decimate == 0:
        decimate = 1
    thresh = sn * sn * T

    # get initial estimate of b and lam on downsampled data using AR1 model
    if decimate > 0:
        from .oasis import oasisAR1, constrained_oasisAR1
        _, s, b, aa, lam = constrained_oasisAR1(
            y[:len(y) // decimate * decimate].reshape(-1, decimate).mean(1),
            d**decimate, sn / sqrt(decimate),
            optimize_b=optimize_b, b_nonneg=b_nonneg, optimize_g=optimize_g)
        if optimize_g:
            from scipy.optimize import minimize
            d = aa**(1. / decimate)
            if decimate > 1:
                s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
            r = estimate_time_constant(s, 1, fudge_factor=.98)[0]
            g[0] = d + r
            g[1] = -d * r
            g11 = (np.exp(log(d) * np.arange(1, T + 1)) -
                   np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
            g12 = np.append(0, g[1] * g11[:-1])
            g11g11 = np.cumsum(g11 * g11)
            g11g12 = np.cumsum(g11 * g12)
            Sg11 = np.cumsum(g11)
            f_lam = 1 - g[0] - g[1]
        elif decimate > 1:
            s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
        lam *= (1 - d**decimate) / f_lam

        # this window size seems necessary and sufficient
        possible_spikes = [x + np.arange(-2, 3)
                           for x in np.where(s > s.max() / 10.)[0]]
        ff = np.array(possible_spikes, dtype=int).ravel()
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        mask = np.zeros(T, dtype=bool)
        mask[ff] = True
    else:
        b = np.percentile(y, 15) if optimize_b else 0
        lam = 2 * sn * np.linalg.norm(g11)
        mask = None
    if b_nonneg:
        b = max(b, 0)

    # run ONNLS
    c, s = onnls(y - b, g, lam=lam, mask=mask,
                 shift=shift, window=window, tol=tol)

    if not optimize_b:  # don't optimize b, just the dual variable lambda
        for _ in range(max_iter - 1):
            res = y - c
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4 * thresh:
                break

            # calc shift dlam
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * \
                np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):
                l = ls[i + 1] - f - 1

                if i == len(ls) - 2:  # last pool
                    tmp[f] = (1. / f_lam if l == 0 else
                              (Sg11[l] + g[1] / f_lam * g11[l - 1]
                               + (g[0] + g[1]) / f_lam * g11[l]
                               - g11g12[l] * tmp[f - 1]) / g11g11[l])
                elif i == len(ls) - 3 and ls[-2] == T - 1:
                    tmp[f] = (Sg11[l] + g[1] / f_lam * g11[l]
                              - g11g12[l] * tmp[f - 1]) / g11g11[l]
                else:
                    tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                l += 1
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]

            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except ValueError:
                db = -bb / aa

            # perform shift
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)
            db = np.mean(y - c) - b
            b += db
            lam -= db / f_lam

    else:  # optimize b
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g[0] - g[1])
        g_converged = False
        for _ in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4 * thresh:
                break
            # calc shift db
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * \
                np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l - 1] - g11g12[l - 1]
                          * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except ValueError:
                db = -bb / aa

            # perform shift
            if b_nonneg:
                db = max(db, -b)
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)

            # update b and lam
            db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            lam -= db / f_lam

            # update g and b
            if optimize_g and (not g_converged):
                from scipy.optimize import minimize as scipy_minimize

                def getRSS(y, opt):
                    b, ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d, r = exp(ld), exp(lr)
                    g1, g2 = d + r, -d * r
                    tmp = b + onnls(y - b, [g1, g2], lam,
                                    mask=(s > 1e-2 * s.max()))[0] - y
                    return tmp.dot(tmp)

                result = scipy_minimize(lambda x: getRSS(y, x), (b, log(d), log(r)),
                                bounds=((0 if b_nonneg else None, None),
                                        (None, -1e-4), (None, -1e-3)), method='L-BFGS-B',
                                options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - log(d)) < 1e-3:
                    g_converged = True
                b, ld, lr = result['x']
                d, r = exp(ld), exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y - b, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)

                # update b and lam
                db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                lam -= db

    if penalty == 0:  # get (locally optimal) L0 solution
        def c4smin(y, s, s_min):
            ls = np.append(np.where(s > s_min)[0], T)
            tmp = np.zeros_like(s)
            l = ls[0]
            tmp[:l] = max(0, np.exp(log(d) * np.arange(l)).dot(y[:l]) * (1 - d * d)
                          / (1 - d**(2 * l))) * np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):
                l = ls[i + 1] - f
                tmp[f] = (g11[:l].dot(y[f:f + l]) - g11g12[l - 1]
                          * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            return tmp

        if s_min == 0:
            spikesizes = np.sort(s[s > 1e-6])
            l = 0
            u = len(spikesizes) - 1
            i = u // 2
            if u >= 0:
                while True:
                    s_min = spikesizes[i]
                    tmp = c4smin(y - b, s, s_min)
                    res = y - b - tmp
                    RSS = res.dot(res)
                    if RSS < thresh:
                        res0 = tmp
                        if i == u:
                            break
                        l = i
                        i = (l + u + 1) // 2
                    else:
                        if i == u or i == 0:
                            break
                        u = i
                        i = (l + u) // 2
            if i > 0:
                c = res0
                s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])
        else:
            if s_min < 0:
                s_min = -s_min * sn * np.sqrt(1 - d)
            for factor in (.7, .8, .9, 1):
                c = c4smin(y - b, s, factor * s_min)
                s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])
        s[s < np.finfo(np.float32).eps] = 0

    return c, s, b, g, lam
