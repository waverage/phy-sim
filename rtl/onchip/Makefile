# cocotb makefile. Run through ../run.sh so the project venv is on PATH:
#   ../run.sh onchip            (Verilator)
#   ../run.sh onchip SIM=icarus
SIM ?= verilator
TOPLEVEL_LANG = verilog
RTL := $(CURDIR)/..
VERILOG_SOURCES = $(CURDIR)/onchip_axi.v $(CURDIR)/onchip.v $(RTL)/tx/tx_frame.v $(RTL)/sync/rx_sync.v \
                  $(RTL)/stf_corr/stf_corr.v $(RTL)/nco_rot/nco_rot.v $(RTL)/chan_emu/chan_emu.v \
                  $(RTL)/chan_emu/chan_emu_rtab.v $(RTL)/rx/rx_frame.v $(RTL)/rx/rx_fe.v $(RTL)/rx/rx_chest.v \
                  $(RTL)/rx/rx_track.v $(RTL)/rx/rx_demap.v $(RTL)/fft256/fft256.v $(RTL)/fft256/fft256_twiddle.v \
                  $(RTL)/cordic/cordic.v $(RTL)/ldpc_dec/ldpc_dec.v
TOPLEVEL = onchip_axi
COCOTB_TEST_MODULES = test_onchip
export PYTHONPATH := $(CURDIR)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
