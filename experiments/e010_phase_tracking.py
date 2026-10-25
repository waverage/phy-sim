"""e010: phase tracking, and whether we need continual pilot subcarriers.

Question from e007: block pilots on symbols 2 and 11 cannot follow the phase
in between. Do we need continual pilots (802.11 uses 4 of 52 subcarriers),
how many, and what do they buy at high Doppler?

Hypotheses:
- The AD9363's locked PLL is far kinder than e007's Wiener model. Its
  integrated phase noise is 0.37 deg rms at 2.4 GHz and about 0.63 deg rms
  at 5.8 GHz (AD9361 data sheet, same synthesiser, ad9361.pdf: table and
  Fig. 67), about -100 and -95 dBc/Hz in a 100 kHz loop; TX and RX together
  add 3 dB. That costs under 0.3 dB on 64qam-5/6 without any tracking.
- What does need tracking is residual carrier frequency offset after the
  preamble's CFO estimate (around 100 Hz rms at moderate SNR): 100 Hz turns
  the phase by 15 deg over a 403 us frame.
- Residual CFO is a phase ramp, and the two block pilot symbols measure its
  slope with 199 pilots each, 9 symbols apart: the "linear" tracker fixes it
  without any continual pilots, unambiguously up to 1.9 kHz.
- At 2000 Hz Doppler (103 m/s at 5.8 GHz) continual pilots fed into the
  LMMSE estimate fix the 64-QAM floor e006 found.

Trackers (OfdmLink phase_track): "linear" fits one phase ramp per frame to
all pilots; "symbol" measures every symbol's phase on its own pilots
(802.11 style). The phase error of either is common to the whole frame, so
it does not average out over codewords and the 1 % FER point sees its
tails. cpN = N continual pilot subcarriers spread over the 199 effective
ones; 8 cost 4.0 % of the data REs.
Run:  .venv/bin/python -I experiments/e010_phase_tracking.py [--profile quick|reference]
"""
import dataclasses
import functools
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import grid                                    # noqa: E402
from harness import impairments as imp                      # noqa: E402
from harness.sweep import default_device, profile_arg, write_result  # noqa: E402
from harness.waveform import NUM_10MHZ, with_continual_pilots  # noqa: E402

NAME = "e010_phase_tracking"
CP8, CP16 = (with_continual_pilots(NUM_10MHZ, n) for n in (8, 16))
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
# PLL in-band level, TX and RX together (+3 dB on one synthesiser)
PLL_24 = -97.0                    # 2.4 GHz, 0.37 deg rms per synthesiser
PLL_58 = -92.0                    # 5.8 GHz, 0.63 deg rms per synthesiser
LIN = dict(phase_track="linear")
CP8_LIN = dict(num=CP8, phase_track="linear")
CP8_SYM = dict(num=CP8, phase_track="symbol")
CP16_SYM = dict(num=CP16, phase_track="symbol")


def pll(level):
    return dict(channel_impairments=functools.partial(imp.pll_phase_noise, inband_dbc_hz=level))


def cfo(hz):
    return dict(channel_impairments=functools.partial(imp.cfo, offset_hz=hz))


def wiener(level):
    return dict(channel_impairments=functools.partial(imp.phase_noise, dbc_hz=level, offset_hz=100e3))


OSC_BASE = dict(channel="awgn", estimator="lmmse")
OSC_CONFIGS = {
    "none":               {},
    # what tracking costs when there is nothing to track
    "lin":                LIN,
    "cp8-lin":            CP8_LIN,
    "cp8-sym":            CP8_SYM,
    # the real oscillator
    "pll2.4":             pll(PLL_24),
    "pll5.8":             pll(PLL_58),
    "pll5.8+6dB":         pll(PLL_58 + 6),
    "pll5.8+6dB-lin":     {**pll(PLL_58 + 6), **LIN},
    # e007's pessimistic random walk
    "wiener-100":         wiener(-100.0),
    "wiener-100-lin":     {**wiener(-100.0), **LIN},
    "wiener-100-cp8-sym": {**wiener(-100.0), **CP8_SYM},
    "wiener-100-cp16-sym": {**wiener(-100.0), **CP16_SYM},
    # residual CFO
    "cfo30":              cfo(30.0),
    "cfo100":             cfo(100.0),
    "cfo100-lin":         {**cfo(100.0), **LIN},
    "cfo300-lin":         {**cfo(300.0), **LIN},
    "cfo1000-lin":        {**cfo(1000.0), **LIN},
    "cfo1000-cp8-lin":    {**cfo(1000.0), **CP8_LIN},
}
OSC_MCS = ("16qam-3/4", "64qam-5/6")

FAST_BASE = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, carrier_hz=5.8e9, estimator="lmmse")
ALL_IMP = dict(channel_impairments=imp.chain(functools.partial(imp.pll_phase_noise, inband_dbc_hz=PLL_58),
                                             functools.partial(imp.cfo, offset_hz=100.0)))
FAST_CONFIGS = {
    "D600":               dict(doppler_hz=600.0),
    "D2000":              dict(doppler_hz=2000.0),
    "D2000-cp8":          dict(doppler_hz=2000.0, num=CP8),
    "D2000-cp16":         dict(doppler_hz=2000.0, num=CP16),
    # the same 8 % of REs spent on a third block pilot symbol instead
    "D2000-p3":           dict(doppler_hz=2000.0, num=P3),
    # everything at once at 5.8 GHz: the M3 receiver's job
    "D600-pll-cfo100-lin":     dict(doppler_hz=600.0, **ALL_IMP, **LIN),
    "D600-pll-cfo100-cp8-lin": dict(doppler_hz=600.0, **ALL_IMP, **CP8_LIN),
}
FAST_MCS = ("16qam-1/2", "64qam-3/4")


def describe(configs):
    """JSON-friendly copy of the config kwargs (functions and numerologies as names)."""
    out = {}
    for k, v in configs.items():
        d = {}
        for a, b in v.items():
            if a == "num":
                d[a] = b.as_dict()
            elif callable(b):
                d[a] = repr(getattr(b, "keywords", b))
            else:
                d[a] = b
        out[k] = d
    return out


def main():
    profile = profile_arg()
    dev = default_device()
    t0 = time.time()
    log = lambda s: print(s, flush=True)                    # noqa: E731
    osc = {k: {**OSC_BASE, **v} for k, v in OSC_CONFIGS.items()}
    fast = {k: {**FAST_BASE, **v} for k, v in FAST_CONFIGS.items()}
    r_osc = grid.run_grid(NUM_10MHZ, osc, OSC_MCS, profile, dev, log=log)
    r_fast = grid.run_grid(NUM_10MHZ, fast, FAST_MCS, profile, dev, span_db=22.0, log=log)
    name = NAME if profile == "reference" else f"{NAME}.quick"
    data_res = {n.name: n.num_data_res for n in (NUM_10MHZ, CP8, CP16, P3)}
    out = write_result(name, {"numerology": NUM_10MHZ.as_dict(), "profile": profile, "device": dev,
                              "pll_inband_dbc_hz": {"2.4GHz": PLL_24, "5.8GHz": PLL_58},
                              "data_res": data_res, "seconds": round(time.time() - t0),
                              "oscillator": {"kwargs": describe(osc), "configs": r_osc},
                              "fast": {"kwargs": describe(fast), "configs": r_fast}})
    print(f"\nwrote {out}")
    print("data REs per frame: " + ", ".join(f"{k} {v}" for k, v in data_res.items()))
    grid.summary(r_osc, "none", "oscillator: AWGN, genie LMMSE")
    grid.summary(r_fast, "D600", "TDL-D 100 ns at 5.8 GHz, genie LMMSE")
    grid.plot(r_osc, "none", f"{name}_osc", "e010 phase noise and residual CFO, AWGN")
    grid.plot(r_fast, "D600", f"{name}_fast", "e010 TDL-D 100 ns, 5.8 GHz")


if __name__ == "__main__":
    main()
