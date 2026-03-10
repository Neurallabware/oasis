"""Tests for parameter estimation functions."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.estimation import GetSn, estimate_time_constant, estimate_parameters, axcov, nextpow2
from src.data_utils import gen_data


class TestGetSn:
    """Test noise estimation."""

    def test_known_noise_level(self):
        """Test noise estimation on synthetic data with known noise."""
        np.random.seed(42)
        sn_true = 0.3
        # Pure noise (no signal) should give accurate estimate
        y = sn_true * np.random.randn(10000)
        sn_est = GetSn(y)
        assert abs(sn_est - sn_true) < 0.05, f"Estimated sn={sn_est:.3f}, expected ~{sn_true}"

    def test_methods(self):
        """Test all three PSD averaging methods."""
        np.random.seed(42)
        y = 0.2 * np.random.randn(5000)
        for method in ['mean', 'median', 'logmexp']:
            sn = GetSn(y, method=method)
            assert 0.1 < sn < 0.4, f"Method {method}: sn={sn:.3f} out of range"

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            GetSn(y, method='invalid')


class TestEstimateTimeConstant:
    """Test AR coefficient estimation."""

    def test_ar1_estimation(self):
        """Test AR(1) coefficient estimation."""
        g_true = np.array([0.95])
        y, _, _ = gen_data(g=g_true, sn=0.1, T=5000, seed=0)
        y = y[0]
        g_est = estimate_time_constant(y, p=1)
        assert abs(g_est[0] - g_true[0]) < 0.1, \
            f"Estimated g={g_est[0]:.3f}, expected ~{g_true[0]}"

    def test_ar2_estimation(self):
        """Test AR(2) coefficient estimation."""
        g_true = np.array([1.7, -0.71])
        y, _, _ = gen_data(g=g_true, sn=0.1, T=5000, seed=0)
        y = y[0]
        g_est = estimate_time_constant(y, p=2)
        assert len(g_est) == 2


class TestHelpers:
    """Test helper functions."""

    def test_nextpow2(self):
        assert nextpow2(1) == 0
        assert nextpow2(2) == 1
        assert nextpow2(3) == 2
        assert nextpow2(4) == 2
        assert nextpow2(5) == 3
        assert nextpow2(1024) == 10
        assert nextpow2(1025) == 11

    def test_axcov_symmetry(self):
        """Test that autocovariance is symmetric."""
        np.random.seed(0)
        data = np.random.randn(1000)
        xc = axcov(data, maxlag=10)
        # axcov should be symmetric around center
        assert len(xc) == 21
        # xc[0..9] should equal xc[20..11] (reversed)
        np.testing.assert_allclose(xc[:10], xc[20:10:-1], atol=1e-10)

    def test_estimate_parameters_passthrough(self):
        """Test that provided g and sn are passed through."""
        y = np.random.randn(100)
        g_in = np.array([0.9])
        sn_in = 0.5
        g_out, sn_out = estimate_parameters(y, g=g_in, sn=sn_in)
        np.testing.assert_array_equal(g_out, g_in)
        assert sn_out == sn_in
