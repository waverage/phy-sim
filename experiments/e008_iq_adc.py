"""e008: IQ imbalance, ADC resolution and AGC error.

Hypothesis: the AD9363 after its own calibration (0.5 dB, 2 deg, about
29 dB image rejection) and its 12-bit ADC with a +/-3 dB AGC cost under
0.2 dB on 64qam-5/6, so the fixed-point modem (M4) can take 12-bit input
with margin, and 10 bits would do. Positive AGC error means the signal
arrives hotter than intended, which eats the headroom and clips.

Model: harness.impairments.iq_imbalance and adc after the noise, on an AWGN
channel, genie-LMMSE estimation. Each sweep varies one impairment with the
others ideal; the ADC rows use 12 dB headroom (full scale at 4x RMS).
Run:  .venv/bin/python -I experiments/e008_iq_adc.py [--profile quick|reference]
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

NAME = "e008_iq_adc"
IQ = ((0.25, 1.0), (0.5, 2.0), (1.0, 4.0), (2.0, 8.0))       # (gain dB, phase deg)
BITS = (6, 8, 10, 12)
AGC_DB = (-6.0, -3.0, 3.0, 6.0)                              # at 12 bits
MCS_NAMES = ("16qam-3/4", "64qam-5/6")


def main():
    profile = profile_arg()
    dev = default_device()
    t0 = time.time()
    base = dict(channel="awgn", estimator="lmmse")
    configs = {"ideal": base}
    for g, ph in IQ:
        configs[f"iq{g:g}dB-{ph:g}deg"] = {**base, "rx_impairments":
                                         functools.partial(imp.iq_imbalance, gain_db=g, phase_deg=ph)}
    for b in BITS:
        configs[f"adc{b}"] = {**base, "rx_impairments": functools.partial(imp.adc, bits=b, headroom_db=12.0)}
    for e in AGC_DB:
        configs[f"agc{e:+g}dB"] = {**base, "rx_impairments":
                                   functools.partial(imp.adc, bits=12, headroom_db=12.0, agc_error_db=e)}
    results = grid.run_grid(NUM_10MHZ, configs, MCS_NAMES, profile, dev, log=lambda s: print(s, flush=True))
    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"numerology": NUM_10MHZ.as_dict(), "profile": profile, "device": dev,
                              "iq": IQ, "bits": BITS, "agc_db": AGC_DB, "headroom_db": 12.0,
                              "seconds": round(time.time() - t0), "configs": results})
    print(f"\nwrote {out}")
    grid.summary(results, "ideal")
    grid.plot(results, "ideal", name, "e008 IQ imbalance, ADC bits, AGC error (12 bit), AWGN")


if __name__ == "__main__":
    main()
