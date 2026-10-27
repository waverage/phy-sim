# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh sync            (Verilator)
#   ../run.sh sync SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
RTL := $(CURDIR)/..
VERILOG_SOURCES = $(CURDIR)/rx_sync.v $(RTL)/stf_corr/stf_corr.v $(RTL)/nco_rot/nco_rot.v $(RTL)/cordic/cordic.v \
                  $(RTL)/fft256/fft256.v $(RTL)/fft256/fft256_twiddle.v
TOPLEVEL = rx_sync
COCOTB_TEST_MODULES = test_rx_sync
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
