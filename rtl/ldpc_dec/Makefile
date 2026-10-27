# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh ldpc_dec            (Verilator)
#   ../run.sh ldpc_dec SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/ldpc_dec.v
TOPLEVEL = ldpc_dec
COCOTB_TEST_MODULES = test_ldpc_dec
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
