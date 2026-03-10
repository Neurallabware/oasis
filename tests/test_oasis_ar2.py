"""Tests for OASIS AR(2) deconvolution."""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import gen_data
from src.deconvolution import constrained_foopsi


class TestOASISAR2:
    """Test AR(2) deconvolution with OASIS."""

    def test_high_snr(self):
        """Test AR(2) deconvolution with high SNR data."""
        g = np.array([1.7, -.71])
        sn = 0.2
        y, c_true, s_true = gen_data(g=g, sn=sn, seed=0)
        y, c_true, s_true = y[0], c_true[0], s_true[0]

        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(
            y, g=g, sn=sn, p=2)

        corr_c = np.corrcoef(c, c_true)[0, 1]
        assert corr_c > 0.99, f"Calcium correlation {corr_c:.4f} < 0.99"

        corr_s = np.corrcoef(sp, s_true)[0, 1]
        assert corr_s > 0.97, f"Spike correlation {corr_s:.4f} < 0.97"

    def test_low_snr(self):
        """Test AR(2) deconvolution with low SNR data."""
        g = np.array([1.7, -.71])
        sn = 0.5
        y, c_true, s_true = gen_data(g=g, sn=sn, seed=0)
        y, c_true, s_true = y[0], c_true[0], s_true[0]

        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(
            y, g=g, sn=sn, p=2)

        corr_c = np.corrcoef(c, c_true)[0, 1]
        assert corr_c > 0.9, f"Calcium correlation {corr_c:.4f} < 0.9"

        corr_s = np.corrcoef(sp, s_true)[0, 1]
        assert corr_s > 0.7, f"Spike correlation {corr_s:.4f} < 0.7"

    def test_auto_estimate_parameters(self):
        """Test AR(2) with auto-estimated parameters."""
        g = np.array([1.7, -.71])
        y, _, _ = gen_data(g=g, sn=0.2, seed=0)
        y = y[0]

        c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(y, p=2)

        assert len(g_est) >= 2
        assert sn_est > 0

    def test_invalid_p_raises(self):
        """Test that p=3 raises ValueError."""
        y = np.random.randn(100)
        g = np.array([0.9])
        with pytest.raises(ValueError, match="only implemented for p=0, p=1 and p=2"):
            constrained_foopsi(y, g=g, sn=0.2, p=3)
