# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh reg_passthrough            (Verilator)
#   ../run.sh reg_passthrough SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/reg_passthrough.v
TOPLEVEL = reg_passthrough
COCOTB_TEST_MODULES = test_reg_passthrough
include $(shell cocotb-config --makefiles)/Makefile.sim
