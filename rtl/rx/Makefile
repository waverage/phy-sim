# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh rx            (Verilator)
#   ../run.sh rx SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
RTL := $(CURDIR)/..
VERILOG_SOURCES = $(CURDIR)/rx_frame.v $(CURDIR)/rx_fe.v $(CURDIR)/rx_chest.v $(CURDIR)/rx_track.v \
                  $(CURDIR)/rx_demap.v $(RTL)/fft256/fft256.v $(RTL)/fft256/fft256_twiddle.v \
                  $(RTL)/cordic/cordic.v $(RTL)/ldpc_dec/ldpc_dec.v
TOPLEVEL = rx_frame
COCOTB_TEST_MODULES = test_rx_frame
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
