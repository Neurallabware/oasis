"""Tests for OASIS AR(1) deconvolution."""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import gen_data
from src.deconvolution import constrained_foopsi


class TestOASISAR1:
    """Test AR(1) deconvolution with OASIS."""

    def test_high_snr(self):
        """Test AR(1) deconvolution with high SNR data."""
        g = np.array([.95])
        sn = 0.2
        y, c_true, s_true = gen_data(g=g, sn=sn, seed=0)
        y, c_true, s_true = y[0], c_true[0], s_true[0]

        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(
            y, g=g, sn=sn, p=1)

        # Check calcium trace correlation
        corr_c = np.corrcoef(c, c_true)[0, 1]
        assert corr_c > 0.99, f"Calcium correlation {corr_c:.4f} < 0.99"

        # Check spike train correlation
        corr_s = np.corrcoef(sp, s_true)[0, 1]
        assert corr_s > 0.97, f"Spike correlation {corr_s:.4f} < 0.97"

    def test_low_snr(self):
        """Test AR(1) deconvolution with low SNR data."""
        g = np.array([.95])
        sn = 0.5
        y, c_true, s_true = gen_data(g=g, sn=sn, seed=0)
        y, c_true, s_true = y[0], c_true[0], s_true[0]

        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(
            y, g=g, sn=sn, p=1)

        corr_c = np.corrcoef(c, c_true)[0, 1]
        assert corr_c > 0.9, f"Calcium correlation {corr_c:.4f} < 0.9"

        corr_s = np.corrcoef(sp, s_true)[0, 1]
        assert corr_s > 0.7, f"Spike correlation {corr_s:.4f} < 0.7"

    def test_auto_estimate_parameters(self):
        """Test that parameters are auto-estimated when not provided."""
        g = np.array([.95])
        y, _, _ = gen_data(g=g, sn=0.2, seed=0)
        y = y[0]

        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(y, p=1)

        assert len(g_est) >= 1
        assert sn_est > 0
        assert bl is not None

    def test_p0_passthrough(self):
        """Test p=0 case returns thresholded fluorescence."""
        y = np.array([1.0, -0.5, 2.0, 0.3, -1.0])
        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(y, p=0)

        npt.assert_array_equal(c, np.maximum(y, 0))
        assert bl == 0
        assert c1 == 0

    def test_raises_without_p(self):
        """Test that ValueError is raised if p is not specified."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="must specify"):
            constrained_foopsi(y)

    def test_rejects_non_oasis_method(self):
        """Test that non-oasis methods are rejected."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="Only.*oasis"):
            constrained_foopsi(y, p=1, method_deconvolution='cvxpy')
