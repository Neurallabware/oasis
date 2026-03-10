"""OASIS: Online Active Set method for Spike Inference from calcium imaging data.

Standalone implementation extracted from CaImAn.

References:
    Friedrich J and Paninski L, NIPS 2016
    Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
"""

from .deconvolution import constrained_foopsi
from .estimation import estimate_parameters, estimate_time_constant, GetSn

try:
    from .oasis import OASIS, oasisAR1, constrained_oasisAR1
except ImportError:
    raise ImportError(
        "Cython extension not compiled. Run 'python setup.py build_ext --inplace' "
        "or 'make build' from the project root."
    )

__all__ = [
    "OASIS",
    "oasisAR1",
    "constrained_oasisAR1",
    "constrained_foopsi",
    "estimate_parameters",
    "estimate_time_constant",
    "GetSn",
]
