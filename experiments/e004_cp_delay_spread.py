"""e004: cyclic prefix length against delay spread.

Hypothesis: the baseline CP of 32 samples (3.2 us, 12.5 % overhead) is
longer than needed for TDL-D up to 300 ns RMS delay spread (most of the
energy is in the line-of-sight tap; the latest tap, at 12.5x the RMS
spread, is weak), so 16 samples would recover about 6 % of rate at no SNR
cost; at 1 us spread the CP starts to matter. Perfect CSI isolates inter-symbol interference from estimation.
Run:  .venv/bin/python -I experiments/e004_cp_delay_spread.py [--profile quick|reference]
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

NAME = "e004_cp_delay_spread"
CHANNEL = dict(channel="tdl", tdl_model="D", doppler_hz=600.0, carrier_hz=2.4e9, estimator="perfect")
CP_LENS = (8, 16, 32, 64)
DELAY_SPREADS_NS = (100, 300, 1000)
MCS_NAMES = ("16qam-1/2", "64qam-3/4")
SPAN_DB = 14.0


def main():
    profile = profile_arg()
    dev = default_device()
    mcs_list = [m for m in MCS_TABLE if m.name in MCS_NAMES]
    t0 = time.time()
    results = {}
    for ds in DELAY_SPREADS_NS:
        for cp in CP_LENS:
            num = dataclasses.replace(NUM_10MHZ, name=f"n10-cp{cp}", cp_len=cp)
            key = f"ds{ds}-cp{cp}"
            results[key] = {"delay_spread_ns": ds, "cp_len": cp, "cp_us": cp / num.bandwidth_hz * 1e6,
                            "numerology": num.as_dict(), "curves": {}}
            for mcs in mcs_list:
                print(f"{key} {mcs.name}  {phy_rate_bps(num, mcs)/1e6:.2f} Mbit/s", flush=True)
                link = OfdmLink(num, mcs, device=dev, delay_spread_s=ds * 1e-9, **CHANNEL)
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
    print(f"\nEs/N0 at 1 % FER (dB); 'n/a' means the curve never reached 1 % (error floor)")
    for m in MCS_NAMES:
        print(f"\n{m}")
        print(f"{'delay spread':>12s} " + " ".join(f"{'cp' + str(cp):>8s}" for cp in CP_LENS))
        for ds in DELAY_SPREADS_NS:
            cells = [results[f"ds{ds}-cp{cp}"]["curves"][m]["esno_db_at_fer1"] for cp in CP_LENS]
            print(f"{str(ds) + ' ns':>12s} " + " ".join(f"{c:8.2f}" if c is not None else f"{'n/a':>8s}" for c in cells))
        rates = [results[f"ds{DELAY_SPREADS_NS[0]}-cp{cp}"]["curves"][m]["phy_rate_bps"] / 1e6 for cp in CP_LENS]
        print(f"{'Mbit/s':>12s} " + " ".join(f"{r:8.2f}" for r in rates))


def plot(results, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(MCS_NAMES), figsize=(12, 4.5))
    for ax, m in zip(axs, MCS_NAMES):
        for i, ds in enumerate(DELAY_SPREADS_NS):
            ys = [results[f"ds{ds}-cp{cp}"]["curves"][m]["esno_db_at_fer1"] for cp in CP_LENS]
            ax.plot([cp / 10 for cp in CP_LENS], [y if y is not None else float("nan") for y in ys],
                    marker="o", label=f"{ds} ns RMS")
        ax.set(xlabel="CP (us)", ylabel="Es/N0 for 1 % FER (dB)", title=m, xscale="log")
        ax.set_xticks([cp / 10 for cp in CP_LENS], [f"{cp / 10:g}" for cp in CP_LENS])
        ax.grid(True, lw=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("e004 CP length vs TDL-D delay spread (perfect CSI; missing point = error floor)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=130)


if __name__ == "__main__":
    main()
