# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh nco_rot            (Verilator)
#   ../run.sh nco_rot SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/nco_rot.v $(CURDIR)/../cordic/cordic.v
TOPLEVEL = nco_rot
COCOTB_TEST_MODULES = test_nco_rot
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
