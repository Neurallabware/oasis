.PHONY: build test clean notebook run help

PYTHON ?= python3
VENV ?= venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build:  ## Build Cython extension in-place
	$(PYTHON) setup.py build_ext --inplace

test: build  ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: build  ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

run: build  ## Run OASIS with default Hydra config
	$(PYTHON) scripts/run_oasis.py

notebook: build  ## Launch Jupyter notebook
	jupyter notebook notebooks/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -f src/*.so src/*.c src/*.cpp
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

setup:  ## Set up virtual environment and install everything
	bash setup_env.sh $(VENV)
