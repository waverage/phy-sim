# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh cordic            (Verilator)
#   ../run.sh cordic SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/cordic.v $(CURDIR)/cordic_tb.v
TOPLEVEL = cordic_tb
COCOTB_TEST_MODULES = test_cordic
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
