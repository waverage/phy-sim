"""e001: AWGN baseline FER/BER curves for the 10 MHz numerology, all MCS.

Hypothesis under test: none yet. This is the reference every later waveform
change is compared against (gate: tests/test_e001_reference.py).
Run:  .venv/bin/python -I experiments/e001_awgn_baseline.py
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.chain import AwgnLink                      # noqa: E402
from harness.sweep import RESULTS_DIR, run_sweep, snr_at_fer, write_result   # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, esno_to_ebno_db, phy_rate_bps, plan_codewords  # noqa: E402

NAME = "e001_awgn_baseline"
# Es/N0 start per MCS (dB); the sweep climbs in 0.5 dB steps and stops once
# a point shows zero errors at the frame cap.
START = {"qpsk-1/2": -1.0, "qpsk-3/4": 2.0, "16qam-1/2": 5.0, "16qam-3/4": 8.0,
         "64qam-2/3": 11.5, "64qam-3/4": 13.0, "64qam-5/6": 15.0}
SETTINGS = dict(batch=250, min_frame_errors=100, max_frames=5000, seed=1)


def main():
    num = NUM_10MHZ
    t0 = time.time()
    curves = {}
    for mcs in MCS_TABLE:
        print(f"{mcs.name}: {plan_codewords(num, mcs)}  {phy_rate_bps(num, mcs)/1e6:.2f} Mbit/s", flush=True)
        link = AwgnLink(num, mcs)
        pts = run_sweep(link, np.arange(START[mcs.name], START[mcs.name] + 8, 0.5), **SETTINGS,
                        log=lambda s: print(s, flush=True))
        s10, s1 = snr_at_fer(pts, 0.10), snr_at_fer(pts, 0.01)
        curves[mcs.name] = {
            "mcs": mcs.as_dict(), "codewords": plan_codewords(num, mcs).__dict__,
            "phy_rate_bps": phy_rate_bps(num, mcs), "points": pts,
            "esno_db_at_fer10": s10, "esno_db_at_fer1": s1,
            "ebno_db_at_fer1": esno_to_ebno_db(s1, mcs) if s1 is not None else None,
        }
    out = write_result(NAME, {"numerology": num.as_dict(), "settings": SETTINGS,
                              "seconds": round(time.time() - t0), "curves": curves})
    print(f"\nwrote {out}")
    print(f"{'MCS':10s} {'Mbit/s':>7s} {'Es/N0@10%':>10s} {'Es/N0@1%':>9s} {'Eb/N0@1%':>9s}")
    for name, c in curves.items():
        f = lambda v: f"{v:9.2f}" if v is not None else "      n/a"
        print(f"{name:10s} {c['phy_rate_bps']/1e6:7.2f} {f(c['esno_db_at_fer10'])} {f(c['esno_db_at_fer1'])} {f(c['ebno_db_at_fer1'])}")
    plot(curves)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c in curves.items():
        xs = [p["esno_db"] for p in c["points"]]
        ys = [max(p["fer"], 1e-5) for p in c["points"]]
        ax.semilogy(xs, ys, marker="o", ms=3, label=name)
    ax.axhline(0.01, color="k", lw=0.5, ls="--")
    ax.set(xlabel="Es/N0 per resource element (dB)", ylabel="frame error rate",
           title="e001 AWGN baseline, 10 MHz numerology, 5G LDPC", ylim=(1e-4, 1))
    ax.grid(True, which="both", lw=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "e001_awgn_baseline.png", dpi=130)


if __name__ == "__main__":
    main()
