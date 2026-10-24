PY := .venv/bin/python
PYTEST := .venv/bin/pytest
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: check test-harness test-modem test-rtl modem e001 e002 e003 e004 e005

check: test-harness test-modem test-rtl

test-harness:
	$(PYTEST) -q tests

modem:
	cd modem && zig build -Doptimize=ReleaseFast

test-modem:
	cd modem && zig build test --summary all

test-rtl:
	rtl/run.sh reg_passthrough clean
	rtl/run.sh reg_passthrough
	rtl/run.sh reg_passthrough SIM=icarus clean
	rtl/run.sh reg_passthrough SIM=icarus

# Experiments. Each writes results/<name>.json (committed) and a PNG (not).
# From e002 on, PROFILE=quick writes results/<name>.quick.json (not committed)
# for ranking hypotheses; the reference profile is the one that gets committed.
PROFILE ?= reference

e001:
	$(PY) -I experiments/e001_awgn_baseline.py
e002:
	$(PY) -I experiments/e002_tdl_doppler.py --profile $(PROFILE)
e003:
	$(PY) -I experiments/e003_pilot_density.py --profile $(PROFILE)
e004:
	$(PY) -I experiments/e004_cp_delay_spread.py --profile $(PROFILE)
e005:
	$(PY) -I experiments/e005_pa_backoff.py --profile $(PROFILE)
