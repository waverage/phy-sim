# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh chan_emu            (Verilator)
#   ../run.sh chan_emu SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/chan_emu.v $(CURDIR)/chan_emu_rtab.v $(CURDIR)/../cordic/cordic.v
TOPLEVEL = chan_emu
COCOTB_TEST_MODULES = test_chan_emu
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
