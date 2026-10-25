# Handoff, 2026-10-25

Read this, then `README.md`. The plan with the full rationale, and the
numerology decision, are in the hardware repo:
`hardware/pcb/radio/phy-simulator-plan-2026-10-24.html` (branch
`radio-investigate-vova`).

## What this is

Simulation environment for Nestua's custom OFDM video PHY. Three layers:
L1 research harness on NVIDIA Sionna (Python), L2 product modem in Zig
(fixed-point, C ABI, bit-exact NumPy twins), L3 Verilog verified with cocotb.
Every hypothesis is a script in `experiments/` that writes JSON to `results/`,
and `make check` gates everything.

## State: M0 to M2 done, M3 next

| Milestone | Status | Evidence |
|---|---|---|
| M0 environment and skeleton | done, c31f79d | `make check`: pytest env + ctypes tests, `zig build test`, cocotb block under Verilator and Icarus |
| M1 AWGN baseline curves | done, bdc1d88 | `results/e001_awgn_baseline.json`, `tests/test_e001_reference.py` |
| M2 OFDM chain, fading, impairments, numerology decision | done 2026-10-25. Decision in the hardware plan ("M2 result", then the e006-e010 follow-ups) | `results/e002..e010*.json`, `tests/test_m2_ofdm.py`, `tests/test_e002_reference.py`, `tests/test_m2_impairments_reference.py`, `tests/test_e010_reference.py` |
| M3 Zig modem, floating point first | not started | |

Reference curves from e001 (10 MHz numerology, 5G LDPC, 20 BP iterations,
Es/N0 per resource element where FER crosses 1 %):

| MCS | Mbit/s | Es/N0 at 1 % FER | gap to Shannon |
|---|---|---|---|
| qpsk-1/2 | 5.92 | 1.41 dB | 1.4 dB |
| qpsk-3/4 | 8.88 | 4.34 dB | 1.7 dB |
| 16qam-1/2 | 11.85 | 6.53 dB | 1.8 dB |
| 16qam-3/4 | 17.77 | 10.32 dB | 1.9 dB |
| 64qam-2/3 | 23.69 | 14.26 dB | 2.5 dB |
| 64qam-3/4 | 26.65 | 15.73 dB | 2.3 dB |
| 64qam-5/6 | 29.61 | 17.29 dB | 2.4 dB |

## What M2 decided (details and numbers in the hardware plan)

- 256-point FFT at 10 MHz, CP 32, 14 symbols, block pilots on symbols 2
  and 11, frame bit interleaver. Fading (TDL-D) costs 4 to 5 dB over AWGN;
  LS estimation is about 3 dB behind LMMSE, so M3 wants a 2D (time and
  frequency) filtered estimator, not per-RE LS.
- No continual pilot subcarriers below about 1500 Hz Doppler (e010). The
  AD9363's PLL (0.37 deg rms at 2.4 GHz, about 0.63 at 5.8 GHz,
  `ad9361.pdf`) costs at most 0.2 dB untracked; e007's 5 dB was the
  unbounded Wiener model. Residual CFO after sync must be tracked, and a
  per-frame phase ramp fitted to the two block pilot symbols does it at no
  overhead up to about 300 Hz (1000 Hz costs 0.6 dB, from ICI).
- At 2000 Hz Doppler two pilot symbols floor 64-QAM. A third pilot
  symbol (1, 6, 11) fixes it as well as 16 continual pilot subcarriers do,
  at the same 8 % of REs and with nothing new in the receiver, so a
  high-Doppler frame format is three pilot symbols, chosen per link.
- PA input back-off about 6 dB for 64-QAM, IQ imbalance 0.5 dB / 2 deg
  costs 0.4 dB, the ADC is free down to 6 bits, the AGC must aim low, and a
  Wi-Fi burst at 10 dB SIR floors 64-QAM (handled by MCS fallback and
  retransmission, not by the PHY).

## Next: M3

Zig modem in floating point first, matching the Sionna chain bit for bit
where it can (mapper, interleaver, OFDM) and statistically where it cannot
(LDPC decoder, estimator). The receiver pieces M2 says it needs:
2D channel estimator from the block pilots, the linear CFO tracker
(`OfdmLink.derotate`, `phase_track="linear"`), LMMSE equalisation, max-log
or exact LLRs. Synchronisation (preamble, coarse CFO, timing) is still
genie in the simulator: worth an experiment before M3 freezes the frame.

## Setup (done on the Windows PC)

WSL2 Ubuntu 24.04, repo at `~/phy-sim` on the Linux filesystem (not
`/mnt/...`, which is ten times slower). Install the Windows NVIDIA driver
only; CUDA works inside WSL2.

```
sudo apt install build-essential git python3.12-venv iverilog poppler-utils
# Verilator >= 5.036 for cocotb 2: Ubuntu's 5.020 is too old, build from source
# Zig 0.17.0 official tarball into ~/.local/zig/0.17.0, symlinked into ~/.local/bin
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt
make check
```

## Known gotchas

- **Sionna only accepts `cpu` and `cuda:N`.** Pass `device=` at
  construction; `.to()` afterwards does not move the LDPC index tables.
- **WSL2 CUDA silently spills to system RAM.** A batch that needs more
  than the 8 GB of the RTX 3060 Ti does not fail, it runs 10x slower.
  Keep the peak under about 6 GB; batch 250 is safe for every chain.
- **The chains `torch.compile` the decoder, demapper and equalizer** by
  default (5 to 8x on the decoder, identical output). The first call per
  block and input shape costs about 20 s, more after WSL restarts with a
  cold cache; pass `compile=False` for tests and one-off runs.
- **Background jobs die with the WSL session** that started them (`nohup`
  is not enough when WSL shuts the VM down); run long experiments from a
  session that stays open.
- **Perfect CSI is the symbol-averaged channel** (`symbol_average_ofdm_channel`):
  Sionna's `time_to_ofdm_channel` samples one instant per symbol and loses
  to a genie LMMSE estimate under Doppler.
- **A common phase error does not average out.** It hits every codeword
  of a frame together, so the 1 % FER point sees its tails: a tracker
  whose rms error looks harmless can cost 1 dB (e010, per-symbol tracking
  on 8 pilots).
- **macOS:** GNU make 3.81 ignores PATH set in a makefile (cocotb runs
  through `rtl/run.sh`), Homebrew Zig 0.17.0 is broken on macOS 26 (use
  the tarball), and Sionna has no Metal support.

## Repo conventions

Results JSON is committed, PNGs and `*.quick.json` are not. One experiment
per script, numbered; `make eNNN PROFILE=quick|reference`.
