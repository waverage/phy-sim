# Handoff, 2026-10-24

For continuing on the Windows PC (or any machine). Read this, then
`README.md`. The plan with the full rationale is in the hardware repo:
`hardware/pcb/radio/phy-simulator-plan-2026-10-24.html` (branch
`radio-investigate-vova`).

## What this is

Simulation environment for Nestua's custom OFDM video PHY. Three layers:
L1 research harness on NVIDIA Sionna (Python), L2 product modem in Zig
(fixed-point, C ABI, bit-exact NumPy twins), L3 Verilog verified with cocotb.
Every hypothesis is a script in `experiments/` that writes JSON to `results/`,
and `make check` gates everything.

## State: M0 and M1 done, M2 not started

| Milestone | Status | Evidence |
|---|---|---|
| M0 environment and skeleton | done, commit c31f79d | `make check` green: pytest env + ctypes tests, `zig build test`, cocotb block under Verilator and Icarus |
| M1 AWGN baseline curves | done, commit bdc1d88 | `results/e001_awgn_baseline.json`, 11 pytest gates pass |
| M2 OFDM chain, fading, impairments, numerology decision | harness, impairments, e002-e005 done and run at reference on the Windows PC (RTX 3060 Ti, 46 min for all four); numerology write-up into the hardware repo pending | `results/e002..e005*.json`, `tests/test_m2_ofdm.py` (13 gates), `tests/test_e002_reference.py` (5 gates) |

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

These are the numbers the gate in `tests/test_e001_reference.py` protects
(0.8 to 3.0 dB from capacity, monotonic, re-run drift under 0.2 dB).

## Where the repo is and how to move it

The repo lives at `~/phy-sim` on the Mac, not in `~/Documents/prg`, because
macOS revoked iTerm2's access to Documents mid-session. It has no remote yet.
To get it to the Windows PC:

```
# on the Mac, after creating an empty GitHub repo
cd ~/phy-sim && git remote add origin https://github.com/waverage/phy-sim.git && git push -u origin main
# later, to put it where the other projects are
rm -rf ~/Documents/prg/phy-sim && mv ~/phy-sim ~/Documents/prg/phy-sim
```

The half-written folder `~/Documents/prg/phy-sim` from before the lockout
contains nothing the real repo lacks.

## Setup on the Windows PC

Use WSL2 with Ubuntu 24.04. Sionna recommends Ubuntu, the cocotb and
Verilator flow is Linux-native, and CUDA works inside WSL2 if the PC has an
NVIDIA GPU (install the Windows NVIDIA driver only; do not install a Linux
driver inside WSL). Native Windows is possible for the Python part but not
for Verilator or cocotb.

```
sudo apt install build-essential git python3.12-venv verilator iverilog
# Zig 0.17.0 official tarball (not a distro package), e.g.
curl -LO https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz
tar xf zig-x86_64-linux-0.17.0.tar.xz && mkdir -p ~/.local/zig ~/.local/bin
mv zig-x86_64-linux-0.17.0 ~/.local/zig/0.17.0 && ln -sf ~/.local/zig/0.17.0/zig ~/.local/bin/zig
# project
git clone https://github.com/waverage/phy-sim.git && cd phy-sim
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# with an NVIDIA GPU, install the CUDA build of torch first, then sionna:
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
make check
```

`make check` must be green before anything else. The e001 drift test runs a
short QPSK sweep, about 20 s on a CPU.

Then pass `device="cuda:0"` to `AwgnLink` (and to future chains) to use the
GPU. The reference profile that took 6 minutes on the M1 Pro CPU should take
seconds.

## Known gotchas

- **macOS GNU make 3.81** ignores PATH set inside a makefile, so cocotb tests
  run through `rtl/run.sh <block> [SIM=icarus]`. Linux make 4.x does not have
  this problem but the wrapper works there too.
- **Homebrew Zig 0.17.0 is broken on macOS 26** ("error: Unexpected" on every
  command). Use the official tarball. Not relevant on Linux.
- **Sionna only accepts `cpu` and `cuda:N`** (`sionna/phy/config.py`,
  `available_devices`). Passing `device=` at construction is required; `.to()`
  afterwards does not move the LDPC index tables.
- **WSL2 CUDA silently spills to system RAM.** A batch that needs more
  than the 8 GB of the RTX 3060 Ti does not fail, it runs 10x slower
  (`torch.cuda.max_memory_allocated()` reports more than the card has).
  Keep the peak under about 6 GB; batch 250 is safe for every chain.
- **The chains `torch.compile` the decoder, demapper and equalizer** by
  default (5 to 8x on the decoder, bit-identical output). The first call
  per block and input shape costs about 20 s; pass `compile=False` for
  one-off runs where that is longer than the run.
- **Sionna's LDPC decoder is the whole cost on CPU**: 95 % of each point,
  about 34 codewords/s at n = 9552 with 20 iterations. Fewer iterations or
  min-sum shift the curve at the knee, so they are not a shortcut for the
  reference profile. Use the `quick` profile in `harness/sweep.py` to rank
  hypotheses (about 8 s per MCS) and the GPU for reference runs.

## Apple Silicon GPU (Metal) investigation, unfinished

Question was whether Sionna can use the M1 GPU. Findings so far:

- NVIDIA does not support it: the device list is hard-coded, the docs name
  Ubuntu only, and the Sionna GitHub has no issues mentioning Apple, Metal
  or MPS as of today. Only Sionna RT (ray tracing) got Metal support in v2.1.
- With a two-line patch (`type(sionna.phy.config).available_devices`
  returning `["cpu", "mps"]` and a `torch.Generator(device="mps")` inserted
  into `config._torch_rngs["mps"]`), the LDPC encoder and the QAM mapper ran
  on Metal. The AWGN block then needed the generator. The decoder, the
  slow part, was not reached before the work was stopped.
- Unknown: whether the decoder's gather/scatter kernels run on MPS and how
  fast. Known PyTorch MPS limits: no float64 (Sionna's "single" precision
  is fine), some ops fall back to CPU with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Verdict: an afternoon experiment, not a dependency. The CUDA PC is the
  supported fast path and the Zig fixed-point decoder (M4) makes the CPU
  fast anyway.

## Next steps, M2

1. `harness/chain.py`: add `OfdmLink` using Sionna `ResourceGrid`,
   `ResourceGridMapper`, `OFDMModulator/Demodulator`, `OFDMChannel` with
   `TDL`, `LSChannelEstimator`, `LMMSEEqualizer`. Keep `AwgnLink` as is.
2. Impairments as plain torch functions in `harness/impairments.py`: CFO
   (±20 ppm), sampling clock offset, phase noise (Wiener process with the
   AD9363 mask), Rapp PA (p = 2, 6 to 10 dB back-off), IQ imbalance
   (0.5 dB, 2°), 12-bit ADC with AGC error, a 20 MHz Wi-Fi burst as
   interference.
3. Experiments e002 (TDL-D with 600 Hz Doppler vs AWGN), e003 (pilot
   density), e004 (CP length vs delay spread), e005 (PA back-off vs EVM
   and spectral mask). Quick profile first, reference on the GPU.
4. Write the numerology decision into the hardware repo's radio plan.

## Repo conventions

Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
Results JSON is committed, PNGs are not. One experiment per script, numbered.
