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

## State: M0 to M4 done, M5 next

| Milestone | Status | Evidence |
|---|---|---|
| M0 environment and skeleton | done, c31f79d | `make check`: pytest env + ctypes tests, `zig build test`, cocotb block under Verilator and Icarus |
| M1 AWGN baseline curves | done, bdc1d88 | `results/e001_awgn_baseline.json`, `tests/test_e001_reference.py` |
| M2 OFDM chain, fading, impairments, numerology decision | done 2026-10-25. Decision in the hardware plan ("M2 result", then the e006-e010 follow-ups) | `results/e002..e010*.json`, `tests/test_m2_ofdm.py`, `tests/test_e002_reference.py`, `tests/test_m2_impairments_reference.py`, `tests/test_e010_reference.py` |
| M3 Zig modem, floating point first | done 2026-10-25 | `modem/src/*.zig`, `harness/modem.py`, `results/e011_zig_overlay.json`, `tests/test_m3_modem.py`, `tests/test_e011_reference.py` |
| M4 fixed point and NumPy twins | done 2026-10-25 | `modem/src/fx*.zig`, `harness/twin.py`, `vectors/`, `results/e012_decoder_fixed.json`, `results/e013_fixed_point.json`, `tests/test_m4_twins.py`, `tests/test_m4_vectors.py`, `tests/test_e012_reference.py`, `tests/test_e013_reference.py` |
| M5 synchronisation under stress | not started | |

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

## What M3 built (README "The Zig modem")

The whole TX and RX chain in Zig, f32, behind a C ABI:
- **Bit-exact against Sionna:** 5G LDPC code construction and encoding
  (every MCS, plus BG2 codes of every k_b class) and the QAM mapper.
  The 64-QAM points needed `x * (1/norm)`, because numpy's complex/real
  division rounds that way.
- **Within f32 rounding:** the TX samples (7e-7 max) and the receiver's
  LLRs on the same received samples. That covers AWGN, TDL-D at 600 Hz,
  200 Hz CFO with the linear tracker, and p3 at 2000 Hz.
- **Statistically:** the LDPC decoder, the same flooding boxplus-phi
  algorithm with clipping at 20. e011 decodes every frame with both
  receivers. The FER curves agree within 0.02 dB at 1 % on all 15 curves, and
  the receivers disagree on 7 codewords in 697,000. That is the M3 gate,
  `tests/test_e011_reference.py`.
- **Throughput:** 1,600 to 12,000 frames/s on the 5900X with 24
  threads, depending on MCS. The decoder stops early on a zero syndrome
  (5 to 14 iterations on average around 1 % FER), so it keeps up with the
  GPU chain: 1.2 to 3x faster than Sionna's receiver in e011.
- **Still from Python:** the numerology and codeword plan, the pilot
  values, the interleaver permutation, and the estimator filter. The
  filter is GenieLmmseEstimator's W per noise level, applied as one
  complex matmul per frame at the data REs. A product receiver needs a
  fixed filter designed for worst-case statistics (and separable, for
  the FPGA); that is M4 work.

## What M4 decided and built (README "The fixed-point modem")

The product modem in integer arithmetic (`modem/src/fxmodem.zig`), a
float twin of the same algorithm, a bit-exact NumPy twin of every block
(`harness/twin.py`), and golden vectors (`vectors/`, `make vectors`).
Gate met: e013 decodes every frame with three receivers (genie f32,
product float, product fixed). The fixed-point loss is +0.07 dB at worst
on 17 curves (tolerance 0.3), and under +0.05 on most.

Decisions, each from a measurement:
- **Decoder (e012):** layered offset min-sum with 6-bit channel LLRs,
  6-bit messages, an 8-bit APP, 0.25 nat per LSB, offset 2 LSB and 20
  iterations with early stop. It is 0.12 to 0.23 dB *better* than Sionna's
  flooding sum-product, because layering converges faster. Max-log costs
  0.00 dB. A 7-bit APP floors at 1e-3, 5-bit LLRs lose 0.4 dB, and
  10 iterations lose 0.1 to 0.4 dB.
- **Estimator (`harness/design.py`):** separable and fixed, one tap set
  per 3 dB of SNR.
  - Frequency: 32 taps over an exponential 300 ns PDP to the CP.
  - Time: Jakes weights at 1000 Hz for two pilot symbols, 1500 Hz for
    three.
  - Both are constrained to unit DC gain. A plain Wiener filter designed
    for more Doppler than the channel has scales the estimate by 1.17 mid
    frame and costs 64-QAM 1 to 3 dB.
- **Phase tracker:** removes half the fitted ramp for two pilot symbols,
  and none for three. It cannot tell residual CFO from the line-of-sight
  Doppler (Sionna's TDL-D puts it at 0.7 fd). Removing all of it shifts
  the diffuse paths' spectrum beyond what the pilots interpolate: 64-QAM
  with 100 Hz CFO lost 3.3 dB at full gain and 0.1 dB at half.
- **LLR scale:** a / (N0 LSB) per frame, capped at 8 / (a P_agc).
  Uncapped, every LLR saturates at high SNR, and 64-QAM on TDL-C floored
  above 1 % FER.
- **Tables:** pilots from the 38.211 Gold sequence (c_init 0x4E5354),
  and the interleaver from a xorshift32 Fisher-Yates shuffle. The
  interleaver is a ROM in hardware: 14,328 x 14 bits for 64-QAM.
- **Widths:** 12-bit ADC and DAC, 16-bit FFT data with Q16 twiddles,
  16-bit channel estimates, Q15 frequency taps, Q14 time weights,
  16-bit CORDIC angles.

What the product algorithm costs against the genie (float - genie, e013):
- **Up to 16-QAM, and the tracked and three-pilot cases:** 0.3 dB or less.
- **AWGN:** 0.1 to 0.2 dB, because the genie knows the channel is static.
- **64-QAM in fading:** 0.6 to 1.0 dB, the most on TDL-C 300 ns with no
  line of sight. An estimator that chooses its design from the measured
  channel (pilot correlation for Doppler, delay spread) would recover part
  of it.

Throughput of the fixed receiver: 9,000 to 24,000 frames/s on the 5900X.

## Next: M5, synchronisation under stress

Synchronisation is still genie: the receiver takes `l_min` and starts with
the residual CFO small. M5 is the preamble, coarse CFO and timing,
tested at full CFO, Doppler and the lowest-MCS SNR (sync success above
99 %). Carry over:
- **The AFC will also lock to the line-of-sight Doppler,** just as the
  tracker does. Check the tracker gain with the real sync in the loop.
- **The adaptive estimator design** (Doppler class per frame) is the
  cheapest remaining dB on 64-QAM.

## Gotchas found in M4

- **Round half up has a bias** that adds up coherently in an FFT (14 LSB
  at DC, 8 at bin 1). The FFT rounds half to even; everything else rounds
  half up, where products have random fractions.
- **Min-sum LLR scaling is free but not irrelevant.** The scale must keep
  typical LLRs inside the 6-bit range, and the offset is in LSBs, so too
  small a scale (a cap of 2 or 4) breaks the float decoder too.
- **Fading curves are shallow at 1 % FER.** With 100 frame errors per point
  the 1 % crossing moves by about ±0.3 dB, even with common samples, so
  the receiver-vs-genie numbers are that noisy. The fixed-vs-float gate is
  not, because those two receivers fail on the same frames.
- **Sionna's TDL draws are not fully pinned by `torch.manual_seed`.** Tests
  that depend on the channel realisation need statistical margins.
- **The Bash tool's shell breaks on apostrophes, even inside a quoted
  heredoc.** Write files with the editor tools, or keep scripts free of
  them.

## Gotchas found in M3

- **Link libc into the Zig library** (`build.zig`). Without libc,
  std.Thread uses its own clone() threads, which hit `unreachable`
  inside a glibc process. In ReleaseFast that is silent UB: `@min(8, 64)`
  came out as 1 and the batch ran on one thread.
- **Zig 0.17 removed `**` array repetition.** Use `@splat`.
- **Python block-buffers through a pipe.** Use `python -u` when tailing
  an experiment log.

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
