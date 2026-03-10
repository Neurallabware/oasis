# OASIS

Online Active Set method for spike inference from calcium fluorescence traces.

This repository provides a standalone OASIS implementation (Cython + Python), configuration-driven experiments with Hydra, plotting utilities, tests, and a demo notebook. The code is extracted and adapted for focused benchmarking and reproducible deconvolution workflows.

## Highlights

- Fast deconvolution with OASIS-style active-set optimization.
- AR(1) and AR(2) support via `constrained_foopsi` interface.
- Automatic or user-specified noise and AR coefficient estimation.
- Synthetic data generation utilities for controlled experiments.
- Notebook and script workflows for quick experimentation and reproducible runs.

## Repository Layout

```text
.
├── configs/
│   ├── eval.yaml               # Main Hydra config entry
│   ├── data/trace.yaml         # Data source and synthetic-data params
│   └── model/oasis.yaml        # OASIS model hyperparameters
├── notebooks/
│   └── demo_oasis.ipynb        # End-to-end demonstration and comparisons
├── scripts/
│   └── run_oasis.py            # CLI/Hydra experiment entrypoint
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # Synthetic traces and helper generation
│   ├── deconvolution.py        # Main deconvolution API and wrappers
│   ├── estimation.py           # Noise and AR parameter estimation
│   ├── oasis.pyx               # Cython implementation
│   ├── oasis.cpp               # Generated C++ source from Cython
│   └── visualization.py        # Plotting and comparison utilities
├── tests/
│   ├── test_estimation.py
│   ├── test_oasis_ar1.py
│   ├── test_oasis_ar2.py
│   └── test_reproducibility.py
├── Makefile
├── requirements.txt
├── setup.py
└── setup_env.sh
```

## Requirements

- Python 3.10+ recommended (project also works in `conda` workflows).
- C/C++ build toolchain for compiling Cython extension.
- `pip`, `setuptools`, `cython`, and `numpy` (installed from `requirements.txt`).

## Installation

### Option A: `pip` / existing environment

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### Option B: helper script

```bash
bash setup_env.sh
```

### Option C: Makefile shortcuts

```bash
make build      # Compile extension module(s)
make test       # Run test suite
make notebook   # Launch Jupyter
```

## Quick Start

### 1) Run default experiment (synthetic trace)

```bash
python scripts/run_oasis.py
```

### 2) Override parameters from CLI (Hydra)

```bash
python scripts/run_oasis.py model.p=2 data.synthetic.sn=0.5
```

### 3) Run on external trace data

```bash
python scripts/run_oasis.py data.format=npy data.path=/path/to/trace.npy model.p=1
```

Outputs are written to timestamped directories under `outputs/`.

## Notebook Workflow

Primary notebook:

- `notebooks/demo_oasis.ipynb`

Before first notebook run in a fresh environment, compile extension once:

```bash
python setup.py build_ext --inplace
```

If notebook import fails with `No module named 'src.oasis'`, extension compilation is missing or stale.

## Programmatic Usage

```python
from src.deconvolution import constrained_foopsi
from src.data_utils import gen_data

# Synthetic trace generation
Y, trueC, trueS = gen_data(g=[0.95], sn=0.3, T=3000)
y = Y[0]

# Deconvolution
c, bl, c1, g, sn, sp, lam = constrained_foopsi(y, p=1)

print("baseline:", bl)
print("estimated g:", g)
print("inferred spikes:", (sp > 0).sum())
```

## Configuration Guide (Hydra)

Main configuration files:

- `configs/eval.yaml`
- `configs/model/oasis.yaml`
- `configs/data/trace.yaml`

Typical model knobs:

- `model.p`: AR order (`0`, `1`, `2`).
- `model.g`: AR coefficients (or auto-estimated when omitted).
- `model.sn`: noise level (or auto-estimated).
- `model.bl`: baseline value (or estimated).
- `model.s_min`: minimum spike threshold behavior.
- `model.optimize_g`: event count used for AR parameter refinement.

Hydra override examples:

```bash
python scripts/run_oasis.py model.p=1 model.s_min=0
python scripts/run_oasis.py model.p=2 model.optimize_g=10
python scripts/run_oasis.py data.synthetic.T=6000 data.synthetic.sn=0.2
```

## Testing

Run all tests:

```bash
pytest -q
```

Or via Makefile:

```bash
make test
```

Current tests validate:

- AR(1) deconvolution behavior.
- AR(2) deconvolution behavior.
- Estimation utilities.
- Reproducibility guarantees.

## Troubleshooting

- `ImportError: Cython extension not compiled`
	Re-run `python setup.py build_ext --inplace` from repo root.

- Build fails due to missing compiler
	Install `build-essential` (or platform equivalent), then retry.

- Notebook kernel mismatch
	Ensure Jupyter kernel points to the same Python environment where dependencies were installed.

- Unexpected parameter behavior
	Print resolved Hydra config and confirm overrides are applied as intended.

## Performance Notes

- Cython extension is the core performance path.
- AR(2) models can be more expressive but may require careful parameterization.
- Accurate `sn` and `g` estimation strongly affects spike reconstruction quality.

## References

- Friedrich J, Paninski L. Fast Active Set Methods for Online Spike Inference from Calcium Imaging. NeurIPS 2016.
- Friedrich J, Zhou P, Paninski L. Fast Online Deconvolution of Calcium Imaging Data. PLOS Computational Biology 2017.
- Pnevmatikakis EA et al. Simultaneous Denoising, Deconvolution, and Demixing of Calcium Imaging Data. Neuron 2016.

## Acknowledgments

Algorithmic foundations and original implementation ideas come from the CaImAn ecosystem and associated OASIS publications.
