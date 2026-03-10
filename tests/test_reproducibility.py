"""Reproducibility tests: verify standalone OASIS matches CaImAn's output exactly."""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import gen_data
from src.deconvolution import constrained_foopsi


# Try importing CaImAn for cross-validation; skip tests if unavailable
try:
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi as caiman_foopsi
    HAS_CAIMAN = True
except ImportError:
    HAS_CAIMAN = False


@pytest.mark.skipif(not HAS_CAIMAN, reason="CaImAn not installed")
class TestReproducibilityVsCaImAn:
    """Cross-validate standalone OASIS output against CaImAn."""

    def test_ar1_high_snr_matches_caiman(self):
        """AR(1) high SNR: exact numerical match with CaImAn."""
        g = np.array([.95])
        sn = 0.2
        y, _, _ = gen_data(g=g, sn=sn, seed=0)
        y = y[0]

        # Standalone
        c_s, bl_s, c1_s, g_s, sn_s, sp_s, lam_s = constrained_foopsi(
            y, g=g, sn=sn, p=1)

        # CaImAn
        c_c, bl_c, c1_c, g_c, sn_c, sp_c, lam_c = caiman_foopsi(
            y, g=g, sn=sn, p=1, method_deconvolution='oasis')

        npt.assert_allclose(c_s, c_c, atol=1e-6,
                            err_msg="Calcium traces differ")
        npt.assert_allclose(sp_s, sp_c, atol=1e-6,
                            err_msg="Spike trains differ")
        npt.assert_allclose(bl_s, bl_c, atol=1e-6,
                            err_msg="Baselines differ")

    def test_ar1_low_snr_matches_caiman(self):
        """AR(1) low SNR: exact numerical match with CaImAn."""
        g = np.array([.95])
        sn = 0.5
        y, _, _ = gen_data(g=g, sn=sn, seed=0)
        y = y[0]

        c_s, bl_s, c1_s, g_s, sn_s, sp_s, lam_s = constrained_foopsi(
            y, g=g, sn=sn, p=1)
        c_c, bl_c, c1_c, g_c, sn_c, sp_c, lam_c = caiman_foopsi(
            y, g=g, sn=sn, p=1, method_deconvolution='oasis')

        npt.assert_allclose(c_s, c_c, atol=1e-6)
        npt.assert_allclose(sp_s, sp_c, atol=1e-6)

    def test_ar2_high_snr_matches_caiman(self):
        """AR(2) high SNR: exact numerical match with CaImAn."""
        g = np.array([1.7, -.71])
        sn = 0.2
        y, _, _ = gen_data(g=g, sn=sn, seed=0)
        y = y[0]

        c_s, bl_s, c1_s, g_s, sn_s, sp_s, lam_s = constrained_foopsi(
            y, g=g, sn=sn, p=2)
        c_c, bl_c, c1_c, g_c, sn_c, sp_c, lam_c = caiman_foopsi(
            y, g=g, sn=sn, p=2, method_deconvolution='oasis')

        npt.assert_allclose(c_s, c_c, atol=1e-5,
                            err_msg="AR(2) calcium traces differ")
        npt.assert_allclose(sp_s, sp_c, atol=1e-5,
                            err_msg="AR(2) spike trains differ")

    def test_auto_params_match_caiman(self):
        """Test that auto-estimated parameters match CaImAn."""
        g = np.array([.95])
        y, _, _ = gen_data(g=g, sn=0.2, seed=42)
        y = y[0]

        c_s, bl_s, c1_s, g_s, sn_s, sp_s, lam_s = constrained_foopsi(y, p=1)
        c_c, bl_c, c1_c, g_c, sn_c, sp_c, lam_c = caiman_foopsi(
            y, p=1, method_deconvolution='oasis')

        npt.assert_allclose(sn_s, sn_c, atol=1e-6,
                            err_msg="Noise estimates differ")
        npt.assert_allclose(g_s, g_c, atol=1e-6,
                            err_msg="AR coefficient estimates differ")
        npt.assert_allclose(c_s, c_c, atol=1e-6,
                            err_msg="Calcium traces differ with auto params")


class TestSelfConsistency:
    """Self-consistency tests (no CaImAn dependency)."""

    def test_deterministic_ar1(self):
        """Same input produces identical output across runs."""
        g = np.array([.95])
        sn = 0.2
        y, _, _ = gen_data(g=g, sn=sn, seed=0)
        y = y[0]

        r1 = constrained_foopsi(y, g=g, sn=sn, p=1)
        r2 = constrained_foopsi(y, g=g, sn=sn, p=1)

        npt.assert_array_equal(r1[0], r2[0], err_msg="Non-deterministic calcium")
        npt.assert_array_equal(r1[5], r2[5], err_msg="Non-deterministic spikes")

    def test_deterministic_ar2(self):
        """Same input produces identical output across runs (AR2)."""
        g = np.array([1.7, -.71])
        sn = 0.2
        y, _, _ = gen_data(g=g, sn=sn, seed=0)
        y = y[0]

        r1 = constrained_foopsi(y, g=g, sn=sn, p=2)
        r2 = constrained_foopsi(y, g=g, sn=sn, p=2)

        npt.assert_array_equal(r1[0], r2[0])
        npt.assert_array_equal(r1[5], r2[5])

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        T = 500
        y = np.random.randn(T).astype(np.float64) + 10
        c, bl, c1, g, sn, sp, lam = constrained_foopsi(y, p=1)

        assert c.shape == (T,), f"c shape {c.shape} != ({T},)"
        assert sp.shape == (T,), f"sp shape {sp.shape} != ({T},)"
        assert np.isscalar(bl) or bl.ndim == 0
        assert np.isscalar(sn) or sn.ndim == 0

    def test_spikes_nonnegative(self):
        """Test that inferred spikes are non-negative."""
        g = np.array([.95])
        y, _, _ = gen_data(g=g, sn=0.3, seed=0)
        y = y[0]

        _, _, _, _, _, sp, _ = constrained_foopsi(y, g=g, sn=0.3, p=1)
        # Float32 Cython output may have tiny negative values (~-3e-8)
        assert np.all(sp >= -1e-6), f"Negative spikes detected: min={sp.min()}"
