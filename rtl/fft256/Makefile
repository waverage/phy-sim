# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh fft256            (Verilator)
#   ../run.sh fft256 SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/fft256.v $(CURDIR)/fft256_twiddle.v
TOPLEVEL = fft256
COCOTB_TEST_MODULES = test_fft256
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
