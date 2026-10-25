"""e009: a Wi-Fi packet landing on a frame.

Hypothesis: a frame survives a Wi-Fi burst over half its band at 10 dB SIR
for the low MCS but not for 64-QAM, and a burst covering the whole frame
breaks everything below about 5 dB SIR, so coexistence is a link-layer
matter (retransmission, MCS fallback) rather than a waveform one.

Model: harness.impairments.wifi_burst, Gaussian noise in the upper half of
our band (a 20 MHz Wi-Fi channel offset to overlap it), hitting every frame
(prob = 1) for 100 us (a short packet) or for the whole 403 us frame. SIR is
mean signal power over interferer power while it is on. AWGN channel, genie
LMMSE estimation, and the receiver does not know about the interferer: its
LLRs on the hit resource elements are overconfident, as in a receiver
without interference detection. With a duty cycle d of frames hit, the
link's FER is about (1 - d) FER_clean + d FER_hit.
Run:  .venv/bin/python -I experiments/e009_wifi_burst.py [--profile quick|reference]
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

NAME = "e009_wifi_burst"
BURSTS_US = (100.0, 403.2)
SIRS_DB = (20.0, 10.0, 5.0, 0.0)
BAND_HZ = (0.0, 5e6)
MCS_NAMES = ("qpsk-1/2", "16qam-1/2", "64qam-3/4")


def main():
    profile = profile_arg()
    dev = default_device()
    t0 = time.time()
    base = dict(channel="awgn", estimator="lmmse")
    configs = {"clean": base}
    for burst in BURSTS_US:
        for sir in SIRS_DB:
            configs[f"{burst:g}us-sir{sir:g}"] = {**base, "channel_impairments": functools.partial(
                imp.wifi_burst, sir_db=sir, band_hz=BAND_HZ, burst_s=burst * 1e-6, prob=1.0)}
    results = grid.run_grid(NUM_10MHZ, configs, MCS_NAMES, profile, dev, span_db=16.0,
                            log=lambda s: print(s, flush=True))
    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"numerology": NUM_10MHZ.as_dict(), "profile": profile, "device": dev,
                              "bursts_us": BURSTS_US, "sirs_db": SIRS_DB, "band_hz": BAND_HZ,
                              "seconds": round(time.time() - t0), "configs": results})
    print(f"\nwrote {out}")
    grid.summary(results, "clean")
    grid.plot(results, "clean", name, "e009 Wi-Fi burst on every frame, upper half of the band, AWGN")


if __name__ == "__main__":
    main()
