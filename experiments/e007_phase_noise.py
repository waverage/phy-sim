"""e007: phase noise against the SNR it costs.

Hypothesis: a free-running LO at about -94 dBc/Hz at 100 kHz (placeholder
for the AD9363; TX and RX together) costs under 0.5 dB on 64qam-5/6 with
only the two pilot symbols to track the phase, and the receiver needs a
per-symbol common-phase-error tracker only about 10 dB worse than that.

Model: Wiener phase noise (harness.impairments.phase_noise), a Lorentzian
spectrum through the given level at 100 kHz, on an AWGN channel so nothing
else moves the phase; genie-LMMSE estimation interpolates the channel
between the pilot symbols but does not know about the phase noise. The
phase variance over one 403 us frame grows 4x per 6 dB: 6.3e-3 rad^2
(4.6 deg rms) at -94 dBc/Hz.
Run:  .venv/bin/python -I experiments/e007_phase_noise.py [--profile quick|reference]
"""
import functools
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import grid                                    # noqa: E402
from harness import impairments as imp                      # noqa: E402
from harness.sweep import default_device, profile_arg, write_result  # noqa: E402
from harness.waveform import NUM_10MHZ                      # noqa: E402

NAME = "e007_phase_noise"
LEVELS_DBC = (-100.0, -94.0, -88.0, -82.0, -76.0)          # at 100 kHz offset
MCS_NAMES = ("16qam-3/4", "64qam-5/6")


def main():
    profile = profile_arg()
    dev = default_device()
    t0 = time.time()
    base = dict(channel="awgn", estimator="lmmse")
    configs = {"none": base}
    for lvl in LEVELS_DBC:
        configs[f"pn{lvl:g}"] = {**base, "channel_impairments":
                                 functools.partial(imp.phase_noise, dbc_hz=lvl, offset_hz=100e3)}
    results = grid.run_grid(NUM_10MHZ, configs, MCS_NAMES, profile, dev, log=lambda s: print(s, flush=True))
    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"numerology": NUM_10MHZ.as_dict(), "profile": profile, "device": dev,
                              "levels_dbc_at_100khz": LEVELS_DBC, "seconds": round(time.time() - t0),
                              "configs": results})
    print(f"\nwrote {out}")
    grid.summary(results, "none")
    grid.plot(results, "none", name, "e007 Wiener phase noise, dBc/Hz at 100 kHz, AWGN")


if __name__ == "__main__":
    main()
