"""e002: TDL-D fading with 600 Hz Doppler vs AWGN, through the OFDM chain.

Hypothesis: the 10 MHz numerology (39 kHz subcarriers, 403 us frame, block
pilots on symbols 2 and 11) holds up at 600 Hz Doppler: with perfect CSI the
fading costs little over AWGN (TDL-D is Rician, K about 13 dB), and the
remaining loss is channel estimation, bracketed by raw LS (floor) and genie
LMMSE (ceiling).
Run:  .venv/bin/python -I experiments/e002_tdl_doppler.py [--profile quick|reference]
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.chain import OfdmLink                          # noqa: E402
from harness.sweep import (RESULTS_DIR, default_device, profile_arg, reference_esno,  # noqa: E402
                           run_curve, write_result)
from harness.waveform import MCS_TABLE, NUM_10MHZ, phy_rate_bps  # noqa: E402

NAME = "e002_tdl_doppler"
CHANNEL = dict(tdl_model="D", delay_spread_s=100e-9, carrier_hz=2.4e9)
CONFIGS = {
    "awgn-perfect":   dict(channel="awgn", estimator="perfect"),
    "tdl0-perfect":   dict(channel="tdl", doppler_hz=0.0, estimator="perfect"),
    "tdl600-perfect": dict(channel="tdl", doppler_hz=600.0, estimator="perfect"),
    "tdl600-lmmse":   dict(channel="tdl", doppler_hz=600.0, estimator="lmmse"),
    "tdl600-ls":      dict(channel="tdl", doppler_hz=600.0, estimator="ls"),
}
SPAN_DB = 14.0


def main():
    profile = profile_arg()
    num, dev = NUM_10MHZ, default_device()
    t0 = time.time()
    results = {}
    for cfg, kw in CONFIGS.items():
        results[cfg] = {}
        for mcs in MCS_TABLE:
            print(f"{cfg} {mcs.name}", flush=True)
            link = OfdmLink(num, mcs, device=dev, **CHANNEL, **kw)
            start = reference_esno(mcs.name) - 1.5
            results[cfg][mcs.name] = run_curve(link, start, SPAN_DB, profile,
                                               log=lambda s: print(s, flush=True))
    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"numerology": num.as_dict(), "profile": profile, "device": dev,
                              "channel": CHANNEL, "configs_kwargs": CONFIGS,
                              "seconds": round(time.time() - t0), "configs": results})
    print(f"\nwrote {out}")
    summary(results, num)
    plot(results, name)


def loss_table(results):
    """Es/N0 at 1 % FER of each config minus awgn-perfect, per MCS."""
    base = results["awgn-perfect"]
    return {cfg: {m: (c["esno_db_at_fer1"] - base[m]["esno_db_at_fer1"])
                  if c["esno_db_at_fer1"] is not None and base[m]["esno_db_at_fer1"] is not None else None
                  for m, c in curves.items()} for cfg, curves in results.items()}


def summary(results, num):
    loss = loss_table(results)
    print(f"\nEs/N0 at 1 % FER (dB), and loss vs awgn-perfect")
    print(f"{'MCS':10s} {'Mbit/s':>7s} " + " ".join(f"{c:>16s}" for c in results))
    for mcs in MCS_TABLE:
        cells = []
        for cfg in results:
            s1 = results[cfg][mcs.name]["esno_db_at_fer1"]
            l = loss[cfg][mcs.name]
            cells.append(f"{s1:7.2f} ({l:+5.2f})" if s1 is not None and l is not None else f"{'n/a':>16s}")
        print(f"{mcs.name:10s} {phy_rate_bps(num, mcs)/1e6:7.2f} " + " ".join(f"{c:>16s}" for c in cells))


def plot(results, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for cfg, curves in results.items():
        for i, (m, c) in enumerate(curves.items()):
            xs = [p["esno_db"] for p in c["points"]]
            ys = [max(p["fer"], 1e-4) for p in c["points"]]
            axs[0].semilogy(xs, ys, color=f"C{list(results).index(cfg)}", lw=1,
                            label=cfg if i == 0 else None)
    axs[0].axhline(0.01, color="k", lw=0.5, ls="--")
    axs[0].set(xlabel="Es/N0 per RE (dB)", ylabel="FER", ylim=(1e-3, 1), title="FER, all MCS")
    axs[0].grid(True, which="both", lw=0.3)
    axs[0].legend(fontsize=8)
    loss = loss_table(results)
    names = [m.name for m in MCS_TABLE]
    for cfg in results:
        if cfg == "awgn-perfect":
            continue
        axs[1].plot(names, [loss[cfg][m] for m in names], marker="o", label=cfg)
    axs[1].set(ylabel="dB more than AWGN at 1 % FER", title="Loss vs AWGN, perfect CSI")
    axs[1].grid(True, lw=0.3)
    axs[1].legend(fontsize=8)
    axs[1].tick_params(axis="x", rotation=30)
    fig.suptitle("e002 TDL-D 100 ns, 600 Hz Doppler, 10 MHz numerology")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=130)


if __name__ == "__main__":
    main()
