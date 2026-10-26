# Handoff, 2026-10-26

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

## State: M0 to M7 done, M8 next

| Milestone | Status | Evidence |
|---|---|---|
| M0 environment and skeleton | done, c31f79d | `make check`: pytest env + ctypes tests, `zig build test`, cocotb block under Verilator and Icarus |
| M1 AWGN baseline curves | done, bdc1d88 | `results/e001_awgn_baseline.json`, `tests/test_e001_reference.py` |
| M2 OFDM chain, fading, impairments, numerology decision | done 2026-10-25. Decision in the hardware plan ("M2 result", then the e006-e010 follow-ups) | `results/e002..e010*.json`, `tests/test_m2_ofdm.py`, `tests/test_e002_reference.py`, `tests/test_m2_impairments_reference.py`, `tests/test_e010_reference.py` |
| M3 Zig modem, floating point first | done 2026-10-25 | `modem/src/*.zig`, `harness/modem.py`, `results/e011_zig_overlay.json`, `tests/test_m3_modem.py`, `tests/test_e011_reference.py` |
| M4 fixed point and NumPy twins | done 2026-10-25 | `modem/src/fx*.zig`, `harness/twin.py`, `vectors/`, `results/e012_decoder_fixed.json`, `results/e013_fixed_point.json`, `tests/test_m4_twins.py`, `tests/test_m4_vectors.py`, `tests/test_e012_reference.py`, `tests/test_e013_reference.py` |
| M5 synchronisation under stress | done 2026-10-25 | `modem/src/fxsync.zig`, `harness/sync.py`, `harness/synclink.py`, `vectors/sync_*.npz`, `results/e014_sync_design.json`, `results/e015_sync_gate.json`, `tests/test_m5_sync.py`, `tests/test_e014_reference.py`, `tests/test_e015_reference.py` |
| M6 link layer and video model | done 2026-10-26 | `harness/fading.py`, `harness/phytrace.py`, `harness/linklayer.py`, `results/e016_phy_traces.json`, `results/e016_traces/`, `results/e017_video_link.json`, `tests/test_m6_link.py`, `tests/test_e016_reference.py`, `tests/test_e017_reference.py` |
| M7 first RTL blocks | done 2026-10-26 | `rtl/fft256/`, `rtl/cordic/`, `rtl/stf_corr/`, `rtl/nco_rot/` (cocotb, Verilator and Icarus), `rtl/synth.sh`, `tools/gen_rtl.py`, `tests/test_m7_rtl.py` |
| M8 full chain in RTL and Arty loopback | not started | |

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

## What M5 decided and built (README "The synchroniser")

The frame now starts with a one-slot preamble, and the fixed-point
synchroniser (`modem/src/fxsync.zig`) finds it in a buffer of ADC samples
and hands the aligned, derotated frame to the M4 receiver. The NumPy twin
(`harness/twin.py`, `TwinSync`) is bit exact against it. The float
reference is `harness/sync.py`. `harness/synclink.py` places each frame at
a random time in the buffer, with the oscillators' full offset: CFO uniform
in +-116 kHz (20 ppm at 5.8 GHz), 20 ppm SCO and PLL noise. e014 chose the
design in floating point; e015 is the gate in fixed point.

- **Preamble:** 9 x 32 samples (one slot, 6.7 % of the frame), every 8th
  subcarrier, QPSK from the Gold sequence at seed 0x4E533C (lowest PAPR,
  4.3 dB), at the data symbols' power.
- **Detection and coarse stage:**
  - Lag-32 autocorrelation over 256 samples against the energy, threshold
    26/256, no division.
  - Coarse timing at the largest |P| after the crossing.
  - Coarse CFO from its CORDIC angle, range +-156 kHz.
- **Fine timing:** impulse response of the first pilot symbol, using the
  FFT, the conjugate pilot and the IFFT. The first path is the earliest tap
  within a CP before the peak that is above peak/8 and 10x the noise floor.
  A CIR with no peak (below 8x its mean) rejects the detection. Timing
  lands within +-2 samples, and the FFT window starts 4 samples early.
- **Medium CFO:** cyclic prefixes against their copies, range +-19.5 kHz.
  It leaves 200 to 450 Hz rms, inside the tracker's +-1.9 kHz.
- **Gate met (e015):** sync success at the qpsk-1/2 1 % point:

  | Channel | Success |
  |---|---|
  | AWGN | 99.56 % |
  | TDL-D 600 | 100 % |
  | TDL-C 300 | 99.92 % |
  | p3 at 2000 Hz | 99.85 % |

  No false alarm in 33 M noise samples. Cold sync costs qpsk-1/2 0.0 to
  0.14 dB (p3: 0.6 dB at 1 %, 0.1 at 10 %). The fixed chain is within
  0.05 dB of float, and the synchroniser runs at 55,000 to 84,000 frames/s
  on one thread per frame.
- **Frequency control is a loop over frames, not per frame.**
  - A per-frame estimate carries the channel's own Doppler. Removing it
    costs 64-QAM about 2 dB on TDL-D, and more on TDL-C (e014: FER 8 %
    against 1.7 %).
  - A loop that averages many frames ("warm") is free without line of
    sight (0.0 to 0.1 dB).
  - **Behind a warm loop the M4 tracker is off** (gain 0 is equal or
    better). Gain 0.5 stays for cold frames.
- **Line-of-sight Doppler:** no receiver can tell it from CFO. Removing it
  with the CFO (warm) costs 64-QAM on TDL-D 0.6 dB at 600 Hz and floors
  p3 at 2000 Hz. Sionna keeps the diffuse paths centred on the carrier, so
  shifting them by -0.6 fd leaves the time filter's band.
  - Subtracting the LoS Doppler predicted from telemetry (positions and
    velocities, "aided") recovers the genie everywhere.
  - For a drone at altitude all paths share roughly the LoS Doppler, and
    then warm alone is right. Sionna's model is the pessimistic case.

## What M6 decided and built (README "The link layer and video model")

- **Two steps.** e016 runs the M5 product chain (fixed TX, preamble,
  oscillators, fixed synchroniser with the warm loop, fixed RX, tracker off)
  on a minute of consecutive frames per channel, on a TDL that is
  continuous in time (`harness/fading.py`). It stores, per frame and MCS,
  the mean Es/N0 at which the frame decodes (bisection on one noise draw,
  0.2 dB). e017 runs the link layer and video model over those traces at
  any SNR trajectory, a minute of link in about half a second.
  - The threshold model matches direct decoding with fresh noise
    (FER within 0.001 to 0.005); at most 1.3 % of frames break
    monotonicity, all within the 0.2 dB search resolution.
  - At 50 Hz Doppler losses come in runs (P(loss | loss) 0.81, mean run 5
    frames); at 600 Hz they are nearly independent (0.06).
- **LLR scale cap raised from 8 to 16** (`FixedPoint.llr_cap`). The cap of
  8 floored the rate-3/4 codes on TDL-C in the product receiver (fixed and
  float alike; the genie receiver had no floor): qpsk-3/4 2.4 % FER at
  40 dB, 64qam-5/6 18 %. A reliable RE 20 dB below the frame's mean gain
  got an LLR of a few LSB and the min-sum offset erased it. At 16:
  qpsk-3/4 has no floor (1 % at 15.2 dB), 16qam-3/4 needs 21.1 dB instead of
  22.9, 64qam-5/6 at 40 dB drops to 7 %; TDL-D and AWGN unchanged.
  64qam-5/6 on TDL-C still never reaches 1 %.
- **The MAC model:** a 23-slot TDD superframe (9.94 ms, 22 downlink, one
  uplink slot for NACKs, the decode ACK and the SNR report), one link
  packet per PHY frame, 1080p60, a 30 ms link budget (a 40 ms
  glass-to-glass target), a freeze is a display gap over 50 ms.
- **nestua-zig's rules do not carry over as they are.** With a 30 ms budget
  they are never freeze-free on a fading channel at its 0.85 bitrate
  fraction; even with its 70 ms window they need 3 to 6 dB more than the
  custom rules for 5 to 8 Mbit/s. The custom rules (e017 `CUSTOM`):
  - Reed-Solomon 16+4 on every MCS (8+4 needs 3 dB more at 8 Mbit/s,
    4+4 is 3 to 5 dB worse above 3 Mbit/s), parity on every frame;
  - a 24-byte packet header;
  - retransmission asked every superframe until 3 ms before the deadline.
    nestua's stop rule (min(window/2, 20 ms) left) costs 1 to 4 dB, worst
    at 50 Hz; no retransmission costs 2 to 4 dB;
  - recovery frames are not duplicated, a lost re-anchor is asked for again
    after 20 ms, the encoder caps a frame at 60 % of what fits in the
    budget (no cap: +3 to 5 dB at 5 Mbit/s), bitrate 0.6 of goodput.
  - The latency budget is worth 1 to 3 dB: 20 ms costs up to 3 dB at
    50 Hz, 50 ms gains 1 to 2.
- **Freeze-free bitrate (custom rules, best rung):** Es/N0 for 1.5, 3, 5,
  8 Mbit/s: AWGN 1.5, 1.5, 4.5, 10.5 dB; TDL-D 600 Hz 4.5, 5.0, 8.0, 14.0;
  TDL-D 50 Hz 6.0, 7.0, 10.0, 16.0; TDL-C 8.5, 8.5, 13.5, 19.0.
- **MCS policy chosen: `SnrPolicy`, outer loop at 1 % pre-FEC packet loss,
  200 ms SNR filter, a 3 dB penalty on an MCS whose step up lost a block**
  (decaying 0.3 dB/s), up one MCS after 200 ms, down at once. On the
  flight (out to 4 km and back at 30 m/s, shadowing, banking dips, 253 s)
  it freezes 1.32 s/min, against 1.29 for fixed qpsk-1/2 (the floor: the
  dips at range defeat every MCS), at 5.0 Mbit/s against 2.3, with 40 MCS
  changes per minute. nestua-zig's ladder freezes 3.57 s/min at 3.1 Mbit/s;
  an SNR table without the outer loop freezes 6 s/min on TDL-C (its
  thresholds ignore the channel). Outer-loop targets of 5 % freeze 2.3 s/min.
- **Per-SNR freeze table** (the M6 gate; e017's summary and
  `results/e017_video_link.png`), freeze s/min / Mbit/s:

  | Es/N0 | 2 | 4 | 6 | 8 | 12 | 16 | 20 | 24 | 30 dB |
  |---|---|---|---|---|---|---|---|---|---|
  | AWGN | 0.2/2.3 | 0.2/2.3 | 0.4/3.6 | 0.5/4.9 | 0.0/7.4 | 0.4/11.1 | 0.0/12.5 | 0.0/12.5 | 0.0/12.5 |
  | TDL-D 600 Hz | 58/2.3 | 0.4/2.3 | 0.0/2.4 | 0.0/2.7 | 0.0/5.0 | 0.1/7.5 | 0.0/10.2 | 0.1/12.2 | 0.0/12.5 |
  | TDL-D 50 Hz | 49/2.3 | 5.6/2.3 | 0.1/2.3 | 0.2/2.6 | 0.3/4.9 | 0.2/7.5 | 0.2/10.2 | 0.0/12.5 | 0.0/12.5 |
  | TDL-C 300 ns | 60/2.3 | 60/2.3 | 27/2.3 | 0.3/2.3 | 0.0/2.4 | 0.1/4.9 | 0.0/6.1 | 0.0/8.2 | 0.0/10.8 |

  On AWGN the outer loop keeps 0.2 to 0.5 s/min at mid SNR: the waterfall
  is a cliff, so each probe one MCS up loses a block. A per-MCS backoff
  longer than the 3 dB penalty would fix it; fading channels do not need it.

## What M7 built (README "The RTL blocks")

- **Gate met: zero diff on every golden vector the blocks touch,** under
  Verilator and Icarus, with a clean `verilator -Wall` lint:
  - `fft256`: blocks.npz (4 FFTs, 4 IFFTs), all 112 OFDM symbols of the
    four frame vectors (rx to the y probe, after the twin's l_min rotation
    and the tracker's derotation), the six sync frames' pilot FFT and
    impulse response, and saturation and exact-half cases;
  - `cordic`: blocks.npz, all 65,536 rotation angles, 5,000 vectoring and
    3,500 angle30 inputs against the twin, the sync frames' coarse and
    cyclic-prefix angles;
  - `stf_corr` / `stf_detect`: the full correlator trace (about 5,600 d per
    frame), the detection, the peak, P and E there, the coarse angle and
    NCO increment of all six sync frames; noise, truncated and full-scale
    buffers against the twin;
  - `nco_rot`: the six aligned frames (4,032 samples each), the pilot
    window and random full-scale cases.
- **Yosys estimate for the XC7A35T** (`rtl/synth.sh`, synth_xilinx, no
  timing): fft256 1,182 LUT, 4 DSP48, 1 RAMB18; stf_detect 4,566 LUT,
  217 SRL, 34 DSP48 (cordic_vec is 2,700 LUT of it); nco_rot 1,674 LUT,
  4 DSP48. Together about a third of the LUTs and half the DSPs.
- **The FFT's first version used a register array** with two reads and two
  writes per clock: 21,000 LUTs, the whole part. One true-dual-port RAM,
  read in one clock and written in the next, costs 2560 clocks a
  transform, inside a 2880-clock symbol at 100 MHz.
- Not in M7: the CIR first-path search, the medium CFO stage's
  accumulation and the control sequencing of the synchroniser (the
  control processor computes the NCO's start phase -n0 inc); the
  estimator, demapper and LDPC decoder are M8.

## Next: M8, full chain in RTL and the Arty loopback

Equaliser, demapper, LDPC decoder, channel emulator, ARM error counter;
the gate is the FER on the board within 0.3 dB of L2. Before the board:
- **stf_detect's 34 DSPs** are the detection metric's wide squares
  (|P|^2, E^2 at 26 to 28 bits); one sample per 10 clocks lets one
  multiplier be shared.
- **Timing:** nothing is constrained yet. The FFT butterfly (four
  multipliers, a 40-bit add and the rounding in one clock) and the
  correlator's metric are the long paths; Vivado will tell.
Open L2 work, for M8 or alongside it:
- **The AFC loop and its Doppler aid** belong in the control processor
  (bandwidth, lock detection, tracker gain switch, telemetry path). The
  traces assume the warm loop is exact.
- **Noise estimation:** the LLR scale still takes the true N0, and the
  link model's SNR report is the true per-frame SNR plus an analytic
  pilot-estimate error.
- **The adaptive estimator** (design per frame) is still the cheapest dB
  on 64-QAM, and TDL-C 64qam-5/6 still floors.
- **The video model** is synthetic: frame sizes from nestua-zig's x265 and
  Hantro numbers, no slices or intra refresh. Bitrate 0.75 of goodput gives
  +25 % bitrate for +0.2 s/min of freeze on the flight; worth revisiting
  with a real encoder trace.

## Gotchas found in M7

- **Verilator makes width warnings fatal** under cocotb: size every
  intermediate explicitly (products at the accumulator's width).
- **Icarus starts registers at X, Verilator at 0:** a pipeline valid bit
  without reset passes under Verilator and fails under Icarus. Reset every
  valid chain and run both simulators.
- **A register array with random-access reads and writes becomes
  flip-flops and multiplexers** (21,000 LUTs for 256 x 32 bits). Write
  RAMs as one always block per port with a registered read, and delay
  lines as shift registers.
- **cocotb refuses an out-of-range value** for a narrow port (a 16-bit tag
  with 65,542 inputs): size test tags for the longest stream.
- **Yosys `-q` hides `stat`:** tee it to a file (`rtl/synth.sh`).

## Gotchas found in M6

- **Airtime, not SNR, decides most freezes.** At a 30 ms budget a frame
  must leave in about 10 ms; an IDR or a duplicated re-anchor that does not
  fit is lost, and the next recovery frame fails the same way (a freeze
  until the next periodic IDR). Cap frame sizes in the encoder.
- **nestua's 10 ms re-request interval is just over the 9.94 ms
  superframe,** so a lost request was repeated only every other
  superframe. Ask every superframe.
- **Compare link configurations by freeze-free bitrate at an SNR, not by
  the SNR one MCS needs:** more parity always lowers the latter.
- **e016 takes about 2 h** (5,000 decodes/s); e017 about 50 min on 24
  cores. Quick traces are 1.8 s long and wrap, fine for debugging only.
- **A Python `multiprocessing` pool and the Zig threads share the CPU:**
  run e016 and e017 one at a time.

## Gotchas found in M5

- **Steep AWGN waterfalls defeat a 0.5 dB grid:** the 1 % point snaps to a
  grid point, so sync modes look identical or 0.17 dB apart at random.
  Compare AWGN at 10 %, or on a finer grid.
- **64-QAM 3/4 on TDL-C is nearly floored even for the genie:** 1 % FER at
  29 to 31 dB and 0.4 % at 39 dB. Its 1 % point is noise, so e015 uses
  16-QAM 3/4 there.
- **Sionna's AWGN draw is not pinned by torch.manual_seed either:**
  regenerating `vectors/sync_*.npz` changes them. They are self-consistent;
  regenerate only for an intended change.

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
sudo apt install build-essential git python3.12-venv iverilog poppler-utils yosys
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
