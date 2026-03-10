"""Setup script for OASIS standalone deconvolution package."""

from setuptools import setup, Extension, find_packages
import numpy as np

try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [Extension(
            "src.oasis",
            sources=["src/oasis.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
        )],
        compiler_directives={'language_level': '3'},
    )
except ImportError:
    ext_modules = []

setup(
    name="oasis-deconv",
    version="1.0.0",
    description="OASIS: Online Active Set method for Spike Inference",
    author="Johannes Friedrich (original), extracted from CaImAn",
    packages=find_packages(),
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "Cython",
        "hydra-core>=1.3.2",
        "hydra-colorlog",
    ],
    extras_require={
        "notebook": ["jupyter", "ipywidgets"],
        "test": ["pytest", "pytest-cov"],
    },
)
