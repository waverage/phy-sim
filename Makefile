PY := .venv/bin/python
PYTEST := .venv/bin/pytest
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: check test-harness test-modem test-rtl modem

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
