# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh tx            (Verilator)
#   ../run.sh tx SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
RTL := $(CURDIR)/..
VERILOG_SOURCES = $(CURDIR)/tx_frame.v $(RTL)/fft256/fft256.v $(RTL)/fft256/fft256_twiddle.v
TOPLEVEL = tx_frame
COCOTB_TEST_MODULES = test_tx_frame
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
