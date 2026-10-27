# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh loopback            (Verilator)
#   ../run.sh loopback SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
RTL := $(CURDIR)/..
VERILOG_SOURCES = $(CURDIR)/loopback_axi.v $(CURDIR)/loopback.v $(RTL)/chan_emu/chan_emu.v $(RTL)/chan_emu/chan_emu_rtab.v \
                  $(RTL)/rx/rx_frame.v $(RTL)/rx/rx_fe.v $(RTL)/rx/rx_chest.v $(RTL)/rx/rx_track.v \
                  $(RTL)/rx/rx_demap.v $(RTL)/fft256/fft256.v $(RTL)/fft256/fft256_twiddle.v \
                  $(RTL)/cordic/cordic.v $(RTL)/ldpc_dec/ldpc_dec.v
TOPLEVEL = loopback_axi
COCOTB_TEST_MODULES = test_loopback
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
