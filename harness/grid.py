"""The shape most experiments share: a set of OfdmLink configurations times a
few MCS, each swept to give the Es/N0 at 1 % FER, compared with a baseline
configuration. Experiments define the configurations and the story; this
module runs, tabulates and plots them."""
from __future__ import annotations

import math

from .chain import OfdmLink
from .sweep import RESULTS_DIR, reference_esno, run_curve
from .waveform import MCS_TABLE, Numerology, phy_rate_bps


def run_grid(num: Numerology, configs: dict[str, dict], mcs_names, profile: str, device: str,
             span_db: float = 14.0, start_offset_db: float = -1.5, log=print) -> dict:
    """configs maps a name to OfdmLink keyword arguments. Each sweep starts
    start_offset_db below the e001 AWGN 10 % point of its MCS."""
    mcs_list = [m for m in MCS_TABLE if m.name in mcs_names]
    results = {}
    for cfg, kw in configs.items():
        results[cfg] = {}
        for mcs in mcs_list:
            log(f"{cfg} {mcs.name}")
            link = OfdmLink(num, mcs, device=device, **kw)
            c = run_curve(link, reference_esno(mcs.name) + start_offset_db, span_db, profile, log=log)
            c["phy_rate_bps"] = phy_rate_bps(num, mcs)
            results[cfg][mcs.name] = c
    return results


def loss_vs(results: dict, baseline: str) -> dict:
    """Es/N0 at 1 % FER of every config minus the baseline's, per MCS; None
    where either curve has an error floor above 1 %."""
    base = results[baseline]
    out = {}
    for cfg, curves in results.items():
        out[cfg] = {}
        for m, c in curves.items():
            a, b = c["esno_db_at_fer1"], base[m]["esno_db_at_fer1"]
            out[cfg][m] = a - b if a is not None and b is not None else None
    return out


def floor_fer(curve: dict) -> float:
    """Lowest FER the sweep reached, the floor when it never crossed 1 %."""
    return min(p["fer"] for p in curve["points"])


def summary(results: dict, baseline: str, title: str = "") -> str:
    """Table of Es/N0 at 1 % FER (loss vs baseline), or the FER floor."""
    loss = loss_vs(results, baseline)
    mcs_names = list(next(iter(results.values())))
    lines = [f"\nEs/N0 at 1 % FER in dB (loss vs {baseline}); 'floor x' = never reached 1 %, lowest FER x"]
    if title:
        lines.insert(0, f"\n{title}")
    lines.append(f"{'config':22s} " + " ".join(f"{m:>18s}" for m in mcs_names))
    for cfg, curves in results.items():
        cells = []
        for m in mcs_names:
            s1, l = curves[m]["esno_db_at_fer1"], loss[cfg][m]
            if s1 is None:
                cells.append(f"floor {floor_fer(curves[m]):.3f}")
            elif l is None:
                cells.append(f"{s1:7.2f}")
            else:
                cells.append(f"{s1:7.2f} ({l:+5.2f})")
        lines.append(f"{cfg:22s} " + " ".join(f"{c:>18s}" for c in cells))
    text = "\n".join(lines)
    print(text)
    return text


def plot(results: dict, baseline: str, name: str, title: str):
    """Loss vs baseline per config, one line per MCS; floors drawn at the top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loss = loss_vs(results, baseline)
    cfgs = list(results)
    mcs_names = list(results[cfgs[0]])
    finite = [v for c in loss.values() for v in c.values() if v is not None]
    top = (max(finite) if finite else 1.0) + 2.0
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(cfgs) + 3), 4.8))
    for i, m in enumerate(mcs_names):
        ys = [loss[c][m] if loss[c][m] is not None else math.nan for c in cfgs]
        ax.plot(cfgs, ys, marker="o", color=f"C{i}", label=m)
        floors = [j for j, c in enumerate(cfgs) if results[c][m]["esno_db_at_fer1"] is None]
        ax.plot([cfgs[j] for j in floors], [top] * len(floors), "x", color=f"C{i}", ms=9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(ylabel=f"dB more than {baseline} at 1 % FER", title=f"{title}  (x = error floor)")
    ax.grid(True, lw=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=130)
