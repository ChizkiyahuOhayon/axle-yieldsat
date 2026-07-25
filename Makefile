# AXLE — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## editable install with dev extras (Python 3.10–3.12)
	$(PY) -m pip install -e ".[dev]"

test:  ## run the test suite (unit + clone-and-run smoke)
	pytest -q

demo:  ## build a synthetic cache and train AXLE on it — no dataset needed
	$(PY) scripts/make_synthetic_cache.py --out data/cache/Synthetic
	$(PY) -m axle.train data=synthetic model=transformer loss=axle protocol=cv10 \
	  protocol.n_splits=3 train.epochs=8

prepare:  ## build a country cache; pass NETCDF=... BOTH=... COUNTRY=... OUT=...
	$(PY) scripts/prepare.py --netcdf "$(NETCDF)" --both "$(BOTH)" \
	  --country "$(COUNTRY)" --out "$(OUT)"

train:  ## train one run; pass ARGS="data=germany model=lstm loss=axle protocol=loyo"
	$(PY) -m axle.train $(ARGS)

clean:  ## remove caches and run outputs
	rm -rf outputs/ data/cache/ .pytest_cache/ **/__pycache__/
