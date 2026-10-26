# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh stf_corr            (Verilator)
#   ../run.sh stf_corr SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/stf_corr.v $(CURDIR)/../cordic/cordic.v
TOPLEVEL = stf_detect
COCOTB_TEST_MODULES = test_stf_corr
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
