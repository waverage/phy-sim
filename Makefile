PY := .venv/bin/python
PYTEST := .venv/bin/pytest
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: check test-harness test-modem test-rtl modem e001 e002 e003 e004 e005 e006 e007 e008 e009 e010 e011

check: test-harness test-modem test-rtl

# the harness tests load the release build of the modem through ctypes
test-harness: modem
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
e006:
	$(PY) -I experiments/e006_channel_set.py --profile $(PROFILE)
e007:
	$(PY) -I experiments/e007_phase_noise.py --profile $(PROFILE)
e008:
	$(PY) -I experiments/e008_iq_adc.py --profile $(PROFILE)
e009:
	$(PY) -I experiments/e009_wifi_burst.py --profile $(PROFILE)

e010:
	$(PY) -I experiments/e010_phase_tracking.py --profile $(PROFILE)

e011: modem
	$(PY) -I experiments/e011_zig_overlay.py --profile $(PROFILE)
