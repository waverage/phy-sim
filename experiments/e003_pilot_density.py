"""e003: block-pilot density at 600 Hz Doppler.

Hypothesis: two pilot symbols per 14-symbol frame (the baseline) are enough
at 600 Hz Doppler (coherence time about 0.7 ms against a 0.4 ms frame);
one costs more SNR than the 7 % of rate it saves, and three or four buy
less SNR than the rate they cost. Each layout changes the number of data REs,
so the comparison is rate against the Es/N0 needed for 1 % FER.
Run:  .venv/bin/python -I experiments/e003_pilot_density.py [--profile quick|reference]
"""
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.chain import OfdmLink                          # noqa: E402
from harness.sweep import (RESULTS_DIR, default_device, profile_arg, reference_esno,  # noqa: E402
                           run_curve, write_result)
from harness.waveform import MCS_TABLE, NUM_10MHZ, phy_rate_bps  # noqa: E402

NAME = "e003_pilot_density"
CHANNEL = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0, carrier_hz=2.4e9)
LAYOUTS = {"p1": (6,), "p2": (2, 11), "p3": (1, 6, 11), "p4": (0, 4, 9, 13)}
ESTIMATORS = ("ls", "lmmse")
MCS_NAMES = ("qpsk-1/2", "16qam-1/2", "64qam-3/4")
SPAN_DB = 14.0


def main():
    profile = profile_arg()
    dev = default_device()
    mcs_list = [m for m in MCS_TABLE if m.name in MCS_NAMES]
    t0 = time.time()
    results = {}
    for lay, pilots in LAYOUTS.items():
        num = dataclasses.replace(NUM_10MHZ, name=f"n10-{lay}", pilot_ofdm_symbols=pilots)
        for est in ESTIMATORS:
            key = f"{lay}-{est}"
            results[key] = {"pilots": pilots, "estimator": est, "numerology": num.as_dict(), "curves": {}}
            for mcs in mcs_list:
                print(f"{key} {mcs.name}  {phy_rate_bps(num, mcs)/1e6:.2f} Mbit/s", flush=True)
                link = OfdmLink(num, mcs, device=dev, estimator=est, **CHANNEL)
                c = run_curve(link, reference_esno(mcs.name) - 1.5, SPAN_DB, profile,
                              log=lambda s: print(s, flush=True))
                c["phy_rate_bps"] = phy_rate_bps(num, mcs)
                results[key]["curves"][mcs.name] = c
    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"profile": profile, "device": dev, "channel": CHANNEL,
                              "seconds": round(time.time() - t0), "configs": results})
    print(f"\nwrote {out}")
    summary(results)
    plot(results, name)


def summary(results):
    print(f"\n{'config':10s} " + " ".join(f"{m:>20s}" for m in MCS_NAMES))
    print(f"{'':10s} " + " ".join(f"{'Mbit/s  Es/N0@1%':>20s}" for _ in MCS_NAMES))
    for key, r in results.items():
        cells = []
        for m in MCS_NAMES:
            c = r["curves"][m]
            s1 = c["esno_db_at_fer1"]
            cells.append(f"{c['phy_rate_bps']/1e6:6.2f} {s1:9.2f}" if s1 is not None else f"{c['phy_rate_bps']/1e6:6.2f} {'n/a':>9s}")
        print(f"{key:10s} " + " ".join(f"{c:>20s}" for c in cells))


def plot(results, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, r) in enumerate(results.items()):
        pts = [(c["esno_db_at_fer1"], c["phy_rate_bps"] / 1e6) for c in r["curves"].values()
               if c["esno_db_at_fer1"] is not None]
        ax.plot(*zip(*pts), marker="o", color=f"C{list(LAYOUTS).index(r['numerology']['name'].split('-')[1])}",
                ls="-" if r["estimator"] == "lmmse" else "--", label=key)
    ax.set(xlabel="Es/N0 for 1 % FER (dB)", ylabel="PHY rate (Mbit/s)",
           title="e003 pilot symbols per frame, TDL-D 100 ns, 600 Hz (up and left is better)")
    ax.grid(True, lw=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=130)


if __name__ == "__main__":
    main()
