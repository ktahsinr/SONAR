PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: help venv install install-local test synthetic report-synthetic smoke clean

help:
	@echo "Targets:"
	@echo "  venv             create a virtualenv in $(VENV)"
	@echo "  install          install pinned requirements (Kaggle-compatible, py3.10-3.12)"
	@echo "  install-local    install unpinned minimal set (newer local Pythons)"
	@echo "  test             run the pytest suite (synthetic data, no network)"
	@echo "  synthetic        generate the synthetic dataset into data_synthetic/"
	@echo "  report-synthetic run the data report on the synthetic dataset"
	@echo "  smoke            train + evaluate the tiny smoke config on CPU"
	@echo "  clean            remove results/, data_synthetic/, caches"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-local: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements-local.txt
	$(PIP) install -e .

test:
	$(PY) -m pytest tests -v

synthetic:
	$(PY) -m bd_usbench.scripts.make_synthetic --out data_synthetic

report-synthetic: synthetic
	$(PY) -m bd_usbench.scripts.data_report \
		--metadata data_synthetic/metadata.csv --image-dir data_synthetic/images || true

smoke: synthetic
	$(PY) -m bd_usbench.scripts.train --config bd_usbench/configs/smoke.yaml
	$(PY) -m bd_usbench.scripts.evaluate --track a --config bd_usbench/configs/smoke.yaml
	$(PY) -m bd_usbench.scripts.compare --results-dir results

clean:
	rm -rf results data_synthetic .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
