PY := .venv/bin/python
PYTEST := .venv/bin/pytest
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: check test-harness test-modem test-rtl lint-rtl rtl-tables modem vectors e001 e002 e003 e004 e005 e006 e007 e008 e009 e010 e011 e012 e013 e014 e015 e016 e017 e018 fpga-src fpga-ooc fpga arm board viewer

check: test-harness test-modem test-rtl

# the harness tests load the release build of the modem through ctypes
test-harness: modem
	$(PYTEST) -q tests

modem:
	cd modem && zig build -Doptimize=ReleaseFast

test-modem:
	cd modem && zig build test --summary all

# Every RTL block's cocotb test under Verilator and Icarus (a failing test
# fails make through cocotb's results.xml), and a -Wall lint.
RTL_BLOCKS := reg_passthrough fft256 cordic stf_corr nco_rot ldpc_dec rx sync tx chan_emu loopback onchip
RX_SRC := rx/rx_frame.v rx/rx_fe.v rx/rx_chest.v rx/rx_track.v rx/rx_demap.v fft256/fft256.v fft256/fft256_twiddle.v cordic/cordic.v ldpc_dec/ldpc_dec.v

test-rtl: lint-rtl
	@for b in $(RTL_BLOCKS); do \
	  for sim in verilator icarus; do \
	    rtl/run.sh $$b SIM=$$sim clean >/dev/null && rtl/run.sh $$b SIM=$$sim || exit 1; \
	  done; \
	done

lint-rtl:
	cd rtl && verilator --lint-only -Wall --top-module fft256 fft256/fft256.v fft256/fft256_twiddle.v
	cd rtl && verilator --lint-only -Wall --top-module cordic_tb cordic/cordic.v cordic/cordic_tb.v
	cd rtl && verilator --lint-only -Wall --top-module stf_detect stf_corr/stf_corr.v cordic/cordic.v
	cd rtl && verilator --lint-only -Wall --top-module nco_rot nco_rot/nco_rot.v cordic/cordic.v
	cd rtl && verilator --lint-only -Wall --top-module ldpc_dec ldpc_dec/ldpc_dec.v
	cd rtl && verilator --lint-only -Wall --top-module chan_emu chan_emu/chan_emu.v chan_emu/chan_emu_rtab.v cordic/cordic.v
	cd rtl && verilator --lint-only -Wall --top-module rx_sync sync/rx_sync.v stf_corr/stf_corr.v nco_rot/nco_rot.v cordic/cordic.v fft256/fft256.v fft256/fft256_twiddle.v
	cd rtl && verilator --lint-only -Wall --top-module tx_frame tx/tx_frame.v fft256/fft256.v fft256/fft256_twiddle.v
	cd rtl && verilator --lint-only -Wall --top-module loopback_axi loopback/loopback_axi.v loopback/loopback.v chan_emu/chan_emu.v chan_emu/chan_emu_rtab.v $(RX_SRC)
	cd rtl && verilator --lint-only -Wall --top-module onchip_axi onchip/onchip_axi.v onchip/onchip.v tx/tx_frame.v sync/rx_sync.v stf_corr/stf_corr.v nco_rot/nco_rot.v chan_emu/chan_emu.v chan_emu/chan_emu_rtab.v $(RX_SRC)

# The RTL's constant tables (committed), generated from the twin
rtl-tables:
	$(PY) -I tools/gen_rtl.py

# Experiments. Each writes results/<name>.json (committed) and a PNG (not).
# From e002 on, PROFILE=quick writes results/<name>.quick.json (not committed)
# for ranking hypotheses; the reference profile is the one that gets committed.
PROFILE ?= reference

e001:
	$(PY) -I experiments/e001_awgn_baseline.py
e002:
	$(PY) -I experiments/e002_tdl_doppler.py --profile $(PROFILE)
e003:
	$(PY) -I experiments/e003_pilot_density.py --profile $(PROFILE)
e004:
	$(PY) -I experiments/e004_cp_delay_spread.py --profile $(PROFILE)
e005:
	$(PY) -I experiments/e005_pa_backoff.py --profile $(PROFILE)
e006:
	$(PY) -I experiments/e006_channel_set.py --profile $(PROFILE)
e007:
	$(PY) -I experiments/e007_phase_noise.py --profile $(PROFILE)
e008:
	$(PY) -I experiments/e008_iq_adc.py --profile $(PROFILE)
e009:
	$(PY) -I experiments/e009_wifi_burst.py --profile $(PROFILE)

e010:
	$(PY) -I experiments/e010_phase_tracking.py --profile $(PROFILE)

e011: modem
	$(PY) -I experiments/e011_zig_overlay.py --profile $(PROFILE)
e012: modem
	$(PY) -I experiments/e012_decoder_fixed.py --profile $(PROFILE)
e013: modem
	$(PY) -I experiments/e013_fixed_point.py --profile $(PROFILE)
e014: modem
	$(PY) -I experiments/e014_sync_design.py --profile $(PROFILE)
e015: modem
	$(PY) -I experiments/e015_sync_gate.py --profile $(PROFILE)
# e017 reads e016's traces (results/e016_traces/)
e016: modem
	$(PY) -I experiments/e016_phy_traces.py --profile $(PROFILE)
e017:
	$(PY) -I experiments/e017_video_link.py --profile $(PROFILE)
e018: modem
	$(PY) -I experiments/e018_board_emulator.py --profile $(PROFILE)

# The board (M8, Arty Z7-20). Vivado and the ARM toolchain are the Windows
# install (VIVADO, ARM_GCC); BUILD_DIR is a Windows-visible build folder.
# fpga-ooc: utilisation and timing of rtl/loopback alone; fpga: the Zynq
# design, bitstream and .xsa; arm: the bare-metal error counter; board:
# the e018 points on the board, checked against e018 (results/e019).
BUILD_DIR ?= /mnt/e/phy-sim-build
VIVADO ?= /mnt/e/amd/2026.1/Vivado/bin/vivado.bat
ARM_GCC ?= /mnt/e/amd/2026.1/gnu/aarch32/nt/gcc-arm-none-eabi/bin/arm-none-eabi-gcc.exe
fpga-src:
	rm -rf $(BUILD_DIR)/rtl $(BUILD_DIR)/fpga && mkdir -p $(BUILD_DIR) && \
	  tar -c --exclude=sim_build --exclude=results.xml --exclude=__pycache__ rtl fpga | tar -x -C $(BUILD_DIR)
fpga-ooc: fpga-src
	cd $(BUILD_DIR) && cmd.exe /c "$$(wslpath -w $(VIVADO))" -mode batch -nojournal -log ooc.log -source fpga/ooc.tcl -tclargs rtl loopback out
# FPGA_TOP=onchip_axi FPGA_OUT=out_onchip builds the M9 whole-radio design
FPGA_TOP ?= loopback_axi
FPGA_OUT ?= out
fpga: fpga-src
	cd $(BUILD_DIR) && cmd.exe /c "$$(wslpath -w $(VIVADO))" -mode batch -nojournal -log build_$(FPGA_TOP).log -source fpga/build.tcl -tclargs rtl $(FPGA_OUT) 100 $(FPGA_TOP)
arm:
	mkdir -p $(BUILD_DIR)/arm && cp fpga/arm/* $(BUILD_DIR)/arm/ && cd $(BUILD_DIR)/arm && \
	  $(ARM_GCC) -mcpu=cortex-a9 -marm -O2 -Wall -Wextra -ffreestanding -nostdlib -T link.ld start.S main.c -o loopback.elf
board: arm
	BUILD_DIR=$(BUILD_DIR) $(PY) -I tools/board.py all

# The simulator viewer: a local web page drawing one frame's every stage
# (tools/viewer.py); open http://localhost:$(VIEWER_PORT), from Windows too.
VIEWER_PORT ?= 8050
viewer: modem
	$(PY) -I tools/viewer.py --port $(VIEWER_PORT)

# Golden vectors for the RTL (committed); regenerate only for an intended
# change to a fixed-point block.
vectors: modem
	$(PY) -I tools/make_vectors.py
