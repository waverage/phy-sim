"""e006: does the M2 numerology hold beyond TDL-D at 600 Hz?

Hypothesis: the decision (two block pilot symbols, 32-sample CP, genie-LMMSE
estimation) holds for no-line-of-sight multipath (TDL-C at 100 and 300 ns)
and for higher Doppler. The carrier enters the simulator only through the
Doppler shift, so 5.8 GHz is tested as Doppler: 600, 1200 and 2000 Hz are
31, 62 and 103 m/s at 5.8 GHz. With 14 symbols of 28.8 us between pilot
symbols 2 and 11, 2000 Hz is about a fifth of a Doppler cycle per frame,
where linear-in-time channel tracking starts to strain, and the subcarrier
spacing (39 kHz) is still 20 times the Doppler, so inter-carrier
interference stays small. Perfect-CSI rows separate the channel's own loss
from the estimator's.

A two-ray ground reflection is not modelled here: at drone geometry the
reflected path arrives within a few ns of the direct one, which over 10 MHz
is a flat gain, a link-budget term rather than a waveform question.
Run:  .venv/bin/python -I experiments/e006_channel_set.py [--profile quick|reference]
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import grid                                    # noqa: E402
from harness.sweep import default_device, profile_arg, write_result  # noqa: E402
from harness.waveform import NUM_10MHZ                      # noqa: E402

NAME = "e006_channel_set"
BASE = dict(channel="tdl", carrier_hz=5.8e9, estimator="lmmse")
CONFIGS = {
    "D100-600":          dict(tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0),
    "D100-1200":         dict(tdl_model="D", delay_spread_s=100e-9, doppler_hz=1200.0),
    "D100-2000":         dict(tdl_model="D", delay_spread_s=100e-9, doppler_hz=2000.0),
    "D100-2000-perfect": dict(tdl_model="D", delay_spread_s=100e-9, doppler_hz=2000.0, estimator="perfect"),
    "C100-600":          dict(tdl_model="C", delay_spread_s=100e-9, doppler_hz=600.0),
    "C300-600":          dict(tdl_model="C", delay_spread_s=300e-9, doppler_hz=600.0),
    "C300-600-perfect":  dict(tdl_model="C", delay_spread_s=300e-9, doppler_hz=600.0, estimator="perfect"),
}
MCS_NAMES = ("qpsk-1/2", "16qam-1/2", "64qam-3/4")
SPAN_DB = 22.0                     # no line of sight costs far more than TDL-D


def main():
    profile = profile_arg()
    dev = default_device()
    t0 = time.time()
    configs = {k: {**BASE, **v} for k, v in CONFIGS.items()}
    results = grid.run_grid(NUM_10MHZ, configs, MCS_NAMES, profile, dev, span_db=SPAN_DB,
                            log=lambda s: print(s, flush=True))
    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"numerology": NUM_10MHZ.as_dict(), "profile": profile, "device": dev,
                              "configs_kwargs": configs, "seconds": round(time.time() - t0),
                              "configs": results})
    print(f"\nwrote {out}")
    grid.summary(results, "D100-600")
    grid.plot(results, "D100-600", name, "e006 channel set, genie LMMSE unless 'perfect'")


if __name__ == "__main__":
    main()
