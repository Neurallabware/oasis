#!/usr/bin/env python
"""Hydra-configured OASIS deconvolution entry point.

Usage:
    python scripts/run_oasis.py                          # default config
    python scripts/run_oasis.py model.p=2                # AR(2) model
    python scripts/run_oasis.py data.format=npy data.path=trace.npy
    python scripts/run_oasis.py data.synthetic.sn=0.5    # higher noise
"""

import sys
import os

# Add project root to path so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


def load_trace(cfg):
    """Load fluorescence trace from file or generate synthetic data."""
    fmt = cfg.data.format

    if fmt == "synthetic":
        from src.data_utils import gen_data
        syn = cfg.data.synthetic
        g_params = OmegaConf.to_container(cfg.model.g) if cfg.model.g is not None else [0.95]
        if cfg.model.p == 2 and len(g_params) < 2:
            g_params = [1.7, -0.71]
        Y, trueC, trueS = gen_data(
            g=g_params, sn=syn.sn, T=syn.T,
            framerate=cfg.data.framerate, firerate=syn.firerate,
            b=syn.baseline, N=syn.N, seed=syn.seed,
        )
        return Y[0], trueC[0], trueS[0]

    elif fmt == "npy":
        y = np.load(cfg.data.path)
        return y, None, None

    elif fmt == "npz":
        data = np.load(cfg.data.path)
        y = data[cfg.data.key]
        return y, None, None

    elif fmt == "mat":
        from scipy.io import loadmat
        data = loadmat(cfg.data.path)
        y = np.squeeze(data[cfg.data.key])
        return y, None, None

    elif fmt == "pkl":
        import pickle
        with open(cfg.data.path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            y = np.asarray(data[cfg.data.key])
        else:
            y = np.asarray(data)
        return y, None, None

    else:
        raise ValueError(f"Unknown data format: {fmt}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load data
    y, trueC, trueS = load_trace(cfg)
    print(f"Trace loaded: T={len(y)}")

    # Build model kwargs
    model_cfg = cfg.model
    g = OmegaConf.to_container(model_cfg.g) if model_cfg.g is not None else None
    if g is not None:
        g = np.array(g)

    from src.deconvolution import constrained_foopsi

    c, bl, c1, g_est, sn_est, sp, lam = constrained_foopsi(
        y,
        bl=model_cfg.bl,
        c1=model_cfg.c1,
        g=g,
        sn=model_cfg.sn,
        p=model_cfg.p,
        bas_nonneg=model_cfg.bas_nonneg,
        noise_range=OmegaConf.to_container(model_cfg.noise_range),
        noise_method=model_cfg.noise_method,
        lags=model_cfg.lags,
        fudge_factor=model_cfg.fudge_factor,
        optimize_g=model_cfg.optimize_g,
        s_min=model_cfg.s_min,
    )

    print(f"Deconvolution complete:")
    print(f"  Baseline: {bl:.4f}")
    print(f"  AR coefficients: {g_est}")
    print(f"  Noise std: {sn_est:.4f}")
    print(f"  Num spikes: {np.sum(sp > 0)}")

    # Save results
    if cfg.eval.save_results:
        out_path = os.path.join(cfg.output_dir, "results.npz")
        save_dict = dict(c=c, sp=sp, bl=bl, c1=c1, g=g_est, sn=sn_est, lam=lam, y=y)
        if trueC is not None:
            save_dict['trueC'] = trueC
            save_dict['trueS'] = trueS
        np.savez(out_path, **save_dict)
        print(f"Results saved to {out_path}")

    # Metrics
    if cfg.eval.compute_metrics and trueC is not None:
        corr_c = np.corrcoef(c, trueC)[0, 1]
        corr_s = np.corrcoef(sp, trueS.astype(float))[0, 1]
        print(f"  Calcium correlation: {corr_c:.4f}")
        print(f"  Spike correlation:   {corr_s:.4f}")

    # Plot
    if cfg.eval.plot:
        from src.visualization import plot_deconvolution
        save_path = cfg.eval.save_plot
        if save_path is None:
            save_path = os.path.join(cfg.output_dir, "deconvolution.png")
        plot_deconvolution(
            y, c, sp, bl=bl,
            framerate=cfg.data.framerate,
            title=f"OASIS AR({model_cfg.p}) Deconvolution",
            save_path=save_path, show=False,
        )
        print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
