"""e017: the link layer and the video model on the PHY traces (M6 gate).

Every run is harness/linklayer.py over an e016 trace: a 10 ms superframe
(22 downlink slots, 1 uplink), one link packet per PHY frame, 1080p60 at
the rung's bitrate, a frame lost when not decoded `latency_ms` after the
encoder. Metric: freeze seconds per minute (display gaps above 50 ms);
freeze-free means at most 0.1 s per minute.

A rung is an MCS and a video bitrate (a fraction of the rung's goodput).
The figure of merit of a set of rules is its envelope: per mean Es/N0, the
highest bitrate of a freeze-free rung, and from it the Es/N0 that carries
1.5, 3, 5 and 8 Mbit/s freeze-free.

1. Rules, at every rung, freeze per minute against the mean Es/N0 on each
   channel:
     nestua      nestua-zig's rules as they are, with its 70 ms window;
     nestua-30   the same with the 30 ms link budget of a 40 ms glass-to-glass
                 target (roadmap: encoder 10-15, decode and display 15);
     custom      the rules tuned for a radio that owns its airtime (CUSTOM).
2. Ablations of custom, one change at a time: the shift of the Es/N0 for
   each target bitrate.
3. MCS policies with the custom rules: at fixed Es/N0 (after a 3 s start)
   against the envelope, and on a flight (out to 4 km and back at 30 m/s,
   shadowing and banking dips) on every fading channel's trace. The policy
   is chosen on the flight.
Run:  .venv/bin/python -I experiments/e017_video_link.py [--profile quick|reference]
"""
import os
import pathlib
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import linklayer as L                         # noqa: E402
from harness.sweep import RESULTS_DIR, profile_arg, write_result   # noqa: E402

NAME = "e017_video_link"
CHANNELS = ("awgn", "tdl-d-600", "tdl-d-50", "tdl-c-300-600")
FADING = CHANNELS[1:]
FREEZE_OK = 0.1                     # s per minute
FRACTIONS = (0.45, 0.6, 0.75, 0.85)
TARGETS = (1.5e6, 3e6, 5e6, 8e6)

NESTUA = replace(L.LinkConfig(), latency_ms=70.0)
NESTUA_30 = L.LinkConfig()
CUSTOM = L.with_fec(replace(
    L.LinkConfig(),
    header_bytes=24,          # 4 B sequence and block ids, 2 B length and flags, 16 B AEAD tag, 2 B frame info
    small_frame_bytes=0,      # parity on every frame: a packet is 0.3 to 1.5 KB here, not Wi-Fi's 1.4 KB
    stop_scaled=False,        # ask until 3 ms before the deadline: the uplink comes every 10 ms
    stop_ms=3.0,
    rerequest_ms=5.0,         # ask again every superframe (nestua's 10 ms is just over one, so every other)
    resync_dup=1,             # no copies of recovery frames: airtime is the scarce thing
    frame_cap=0.6,            # the encoder caps a frame at 60 % of what fits in the budget
    ltr_cooldown_ms=20.0,     # a lost re-anchor is asked for again after two superframes
    bitrate_fraction=0.6),
    16, 4)                    # 20 % parity over 16 packets: the first reference run's best FEC (8+4: -3 dB at 8 Mbit/s)
RULES = {"nestua": NESTUA, "nestua-30": NESTUA_30, "custom": CUSTOM}
ABLATIONS = {
    "custom": CUSTOM,
    "header 87 B": replace(CUSTOM, header_bytes=L.NESTUA_HEADER_BYTES),
    "no parity <= 2200 B": replace(CUSTOM, small_frame_bytes=2200),
    "stop min(w/2, 20 ms)": replace(CUSTOM, stop_scaled=True, stop_ms=20.0),
    "re-ask after 10 ms": replace(CUSTOM, rerequest_ms=10.0),
    "max 10 entries -> 64": replace(CUSTOM, max_entries=64),
    "resync x2": replace(CUSTOM, resync_dup=2),
    "no frame cap": replace(CUSTOM, frame_cap=0.0),
    "cooldown 100 ms": replace(CUSTOM, ltr_cooldown_ms=100.0),
    "fec 4+4": L.with_fec(CUSTOM, 4, 4),
    "fec 8+2": L.with_fec(CUSTOM, 8, 2),
    "fec 8+3": L.with_fec(CUSTOM, 8, 3),
    "fec 8+4": L.with_fec(CUSTOM, 8, 4),
    "fec 16+2": L.with_fec(CUSTOM, 16, 2),
    "fec 16+6": L.with_fec(CUSTOM, 16, 6),
    "fec 8+4 on QPSK": L.with_fec(CUSTOM, 8, 4, mcs=("qpsk-1/2", "qpsk-3/4")),
    "fill parity 2": replace(CUSTOM, fill_parity=2),
    "no retransmission": replace(CUSTOM, retransmit=False),
    "latency 20 ms": replace(CUSTOM, latency_ms=20.0),
    "latency 50 ms": replace(CUSTOM, latency_ms=50.0),
    "IDR 9.4x (x265)": replace(CUSTOM, idr_factor=9.4, idr_recovery_factor=9.4),
    "no LTR": replace(CUSTOM, ltr_frames=0),
    "uplink -6 dB": replace(CUSTOM, ul_offset_db=-6.0),
}
ABL_FRACTIONS = (0.45, 0.6, 0.75)


def _policies():
    p = {}
    for tau in (50, 200):
        for tgt in (0.01, 0.02, 0.05):
            for pen in (0.0, 3.0):
                p[f"snr t{tgt:.0%} tau{tau}" + (" pen3" if pen else "")] = (
                    "snr", dict(margin_db=3.0, target=tgt, tau_ms=float(tau), penalty_db=pen))
    p["snr m6 no-olla"] = ("snr", dict(margin_db=6.0, olla=False))
    p["ladder"] = ("ladder", {})
    return p


# the outer-loop policies at 1 and 2 % also run at these bitrate fractions
POLICY_FRACTIONS = (0.75,)


def policy_set() -> dict:
    """{name: (rules, policy spec)}"""
    out = {pn: (CUSTOM, spec) for pn, spec in POLICIES.items()}
    for f in POLICY_FRACTIONS:
        for pn in ("snr t1% tau50", "snr t2% tau50"):
            out[f"{pn} f{f}"] = (replace(CUSTOM, bitrate_fraction=f), POLICIES[pn])
    out["ladder (nestua rules)"] = (NESTUA, ("ladder", {}))
    return out


POLICIES = _policies()
PROFILES = {
    "quick": dict(traces="e016_traces.quick", step_db=1.0, pool=8, policy_step=2),
    "reference": dict(traces="e016_traces", step_db=0.5, pool=os.cpu_count(), policy_step=2),
}
GRID = (-2.0, 40.0)
WARMUP_S = 3.0

_TRACES = {}


def trace(path, ch):
    key = (path, ch)
    if key not in _TRACES:
        d = np.load(RESULTS_DIR / path / f"{ch}.npz")
        thr = d["thr"].astype(np.float32)
        _TRACES[key] = (thr.reshape(thr.shape[0], -1), d["gain_db"].astype(np.float32).reshape(-1), float(d["frame_s"]))
    return _TRACES[key]


def policy_of(spec):
    kind, kw = spec
    if kind == "fixed":
        return L.FixedPolicy(kw)
    if kind == "snr":
        return L.SnrPolicy(**kw)
    return L.LadderPolicy(**kw)


def job(a):
    path, ch, cfg, pol, snr, warmup, seed = a
    thr, gain, slot = trace(path, ch)
    sec = thr.shape[1] * slot if np.isscalar(snr) else len(snr) * slot
    return L.LinkSim(thr, gain, slot, cfg, policy_of(pol), snr, seed=seed).run(sec + warmup, warmup)


def slim(r):
    keys = ("freeze_s_per_min", "freezes_per_min", "frames_lost_per_min", "mean_bitrate_bps", "latency_p50_ms",
            "latency_p99_ms", "pre_fec_loss", "retx_share", "ul_success", "mcs_changes", "longest_freeze_s")
    out = {k: round(r[k], 4) for k in keys}
    out["mcs"] = max(r["mcs_share"], key=r["mcs_share"].get)
    return out


def required(curve):
    """Lowest grid Es/N0 from which every higher point is freeze-free."""
    ok = None
    for s, r in sorted(curve.items(), reverse=True):
        if r["freeze_s_per_min"] <= FREEZE_OK:
            ok = s
        else:
            break
    return ok


def envelope(curves: dict, grid) -> dict:
    """curves {rung: {snr: slim}} -> {snr: (rung, bitrate)}: the best rung
    freeze-free from that Es/N0 upwards."""
    req = {k: required(c) for k, c in curves.items()}
    rate = {k: max(r["mean_bitrate_bps"] for r in c.values()) for k, c in curves.items()}
    out = {}
    for s in grid:
        best = None
        for k, q in req.items():
            if q is not None and q <= s and (best is None or rate[k] > best[1]):
                best = (k, rate[k])
        out[float(s)] = best
    return out


def esno_for(env: dict, target: float):
    for s in sorted(env):
        if env[s] is not None and env[s][1] >= target:
            return s
    return None


def plot(res: dict, name: str):
    """Freeze-free bitrate against the mean Es/N0, per channel and rules,
    and the chosen policy's mean bitrate at fixed Es/N0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(CHANNELS), figsize=(4.2 * len(CHANNELS), 4.0), sharey=True)
    pg = res["policy_grid_db"]
    for ax, ch in zip(axs, CHANNELS):
        for i, rn in enumerate(RULES):
            env = res["rules"][ch][rn]["envelope"]
            s = sorted(env)
            ax.step(s, [env[x][1] / 1e6 if env[x] else 0 for x in s], where="post", color=f"C{i}", label=f"{rn} (best rung)")
        pol = res["policies"][ch].get(CHOSEN)
        if pol:
            ok = [f <= FREEZE_OK for f in pol["freeze_s_per_min"]]
            ax.plot([x for x, o in zip(pg, ok) if o], [b / 1e6 for b, o in zip(pol["mean_bitrate_bps"], ok) if o],
                    "k.", ms=4, label=f"{CHOSEN} (freeze-free points)")
        ax.set(title=ch, xlabel="mean Es/N0 (dB)", xlim=(-2, 35))
        ax.grid(True, lw=0.3)
    axs[0].set_ylabel("video Mbit/s freeze-free (<= 0.1 s/min)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=130)


# the policy M6 chooses (on the flight; see the summary)
CHOSEN = "snr t1% tau200 pen3"


def main():
    profile = profile_arg()
    prof = PROFILES[profile]
    t_all = time.time()
    grid = np.arange(GRID[0], GRID[1] + 1e-9, prof["step_db"]).round(2)
    path = prof["traces"]
    res = {"rules": {}, "ablations": {}, "policies": {}, "flight": {}}

    def sweep(pool, items, grid_):
        """items {key: (ch, cfg, policy spec)} -> {key: {snr: slim}}"""
        jobs, keys = [], []
        for k, (ch, cfg, spec) in items.items():
            for s in grid_:
                jobs.append((path, ch, cfg, spec, float(s), 0.0 if spec[0] == "fixed" else WARMUP_S, 1))
                keys.append((k, float(s)))
        out = pool.map(job, jobs, chunksize=4)
        curves = {}
        for (k, s), r in zip(keys, out):
            curves.setdefault(k, {})[s] = slim(r)
        return curves

    with Pool(prof["pool"]) as pool:
        # ---- 1. rules, every rung
        t0 = time.time()
        items = {(ch, rn, m, f): (ch, replace(cfg, bitrate_fraction=f), ("fixed", m))
                 for ch in CHANNELS for rn, cfg in RULES.items() for m in L.MCS_NAMES for f in FRACTIONS}
        curves = sweep(pool, items, grid)
        print(f"rules: {len(items)} rungs, {time.time() - t0:.0f} s", flush=True)
        for ch in CHANNELS:
            res["rules"][ch] = {}
            for rn, cfg in RULES.items():
                rungs = {f"{m} {f}": curves[(ch, rn, m, f)] for m in L.MCS_NAMES for f in FRACTIONS}
                env = envelope(rungs, grid)
                as_is = {f"{m} {cfg.bitrate_fraction}": rungs[f"{m} {cfg.bitrate_fraction}"] for m in L.MCS_NAMES}
                res["rules"][ch][rn] = {
                    "rungs": {k: {"required_esno_db": required(c),
                                  "bitrate_bps": round(max(r["mean_bitrate_bps"] for r in c.values())),
                                  "freeze_s_per_min": [round(c[float(s)]["freeze_s_per_min"], 3) for s in grid]}
                              for k, c in rungs.items()},
                    "envelope": env,
                    "esno_db_for": {f"{t / 1e6:g}": esno_for(env, t) for t in TARGETS},
                    "esno_db_for_as_is": {f"{t / 1e6:g}": esno_for(envelope(as_is, grid), t) for t in TARGETS}}
        # ---- 2. ablations of custom
        t0 = time.time()
        agrid = grid[::2] if profile == "reference" else grid
        items = {(ch, an, m, f): (ch, replace(cfg, bitrate_fraction=f), ("fixed", m))
                 for ch in FADING for an, cfg in ABLATIONS.items() for m in L.MCS_NAMES for f in ABL_FRACTIONS}
        curves = sweep(pool, items, agrid)
        print(f"ablations: {len(items)} rungs, {time.time() - t0:.0f} s", flush=True)
        for ch in FADING:
            res["ablations"][ch] = {}
            for an in ABLATIONS:
                env = envelope({f"{m} {f}": curves[(ch, an, m, f)] for m in L.MCS_NAMES for f in ABL_FRACTIONS}, agrid)
                res["ablations"][ch][an] = {"esno_db_for": {f"{t / 1e6:g}": esno_for(env, t) for t in TARGETS}}
            base = res["ablations"][ch]["custom"]["esno_db_for"]
            for an, x in res["ablations"][ch].items():
                x["delta_db"] = {t: (None if base[t] is None or v is None else round(v - base[t], 2))
                                 for t, v in x["esno_db_for"].items()}
        # ---- 3a. policies at fixed Es/N0
        t0 = time.time()
        pgrid = grid[::prof["policy_step"]]
        items = {(ch, pn): (ch, cfg, spec) for ch in CHANNELS for pn, (cfg, spec) in policy_set().items()}
        curves = sweep(pool, items, pgrid)
        print(f"policies: {len(items)} curves, {time.time() - t0:.0f} s", flush=True)
        for (ch, pn), c in curves.items():
            env = res["rules"][ch]["custom"]["envelope"]
            # freeze and bitrate against the custom envelope's best rung at that Es/N0
            res["policies"].setdefault(ch, {})[pn] = {
                **{k: [c[float(s)][k] for s in pgrid] for k in ("freeze_s_per_min", "mean_bitrate_bps", "mcs_changes",
                                                                 "latency_p99_ms")},
                "mcs": [c[float(s)]["mcs"] for s in pgrid],
                "freeze_s_per_min_mean": round(float(np.mean([r["freeze_s_per_min"] for s, r in c.items()
                                                              if env.get(s) is not None])), 4),
                "bitrate_over_envelope": round(float(np.mean([r["mean_bitrate_bps"] / env[s][1] for s, r in c.items()
                                                              if env.get(s) is not None])), 4)}
        # ---- 3b. the flight, on every fading channel
        t0 = time.time()
        slot = trace(path, CHANNELS[0])[2]
        snr_f, dist = L.flight_snr(slot, snr_1km_db=20.0, seed=5)
        specs = policy_set()
        for m in L.MCS_NAMES:
            specs[f"fixed {m}"] = (CUSTOM, ("fixed", m))
        jobs, keys = [], []
        for ch in FADING:
            for pn, (cfg, spec) in specs.items():
                jobs.append((path, ch, cfg, spec, snr_f, 0.0, 2))
                keys.append((ch, pn))
        out = pool.map(job, jobs, chunksize=1)
        print(f"flight: {len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
        for (ch, pn), r in zip(keys, out):
            res["flight"].setdefault(ch, {})[pn] = {**slim(r), "mcs_share": r["mcs_share"]}
    minutes = len(snr_f) * slot / 60
    res["flight_profile"] = {"snr_1km_db": 20.0, "d_min_m": 200.0, "d_max_m": 4000.0, "speed_mps": 30.0,
                             "seconds": round(minutes * 60, 1), "esno_db_min": round(float(snr_f.min()), 2),
                             "esno_db_max": round(float(snr_f.max()), 2)}
    res["grid_db"] = [float(s) for s in grid]
    res["policy_grid_db"] = [float(s) for s in pgrid]
    # the flight's score per policy: freeze summed over the fading channels, and the mean bitrate
    score = {}
    for pn in res["flight"][FADING[0]]:
        rs = [res["flight"][ch][pn] for ch in FADING]
        score[pn] = {"freeze_s_per_min": round(sum(r["freeze_s_per_min"] for r in rs) / len(rs), 4),
                     "mean_bitrate_bps": round(sum(r["mean_bitrate_bps"] for r in rs) / len(rs)),
                     "mcs_changes_per_min": round(sum(r["mcs_changes"] for r in rs) / len(rs) / minutes, 2)}
    res["flight_score"] = score

    f = lambda v: "-" if v is None else f"{v:5.1f}"        # noqa: E731
    lines = [f"Es/N0 (dB) that carries {', '.join(f'{t / 1e6:g}' for t in TARGETS)} Mbit/s freeze-free"
             f" (<= {FREEZE_OK} s/min): best rung, and [as is]"]
    for ch in CHANNELS:
        for rn in RULES:
            x = res["rules"][ch][rn]
            lines.append(f"{ch:14s} {rn:10s} " + " ".join(f(v) for v in x["esno_db_for"].values()) + "   ["
                         + " ".join(f(v) for v in x["esno_db_for_as_is"].values()) + "]")
    lines.append("\nAblations of custom: Es/N0 for each bitrate (shift)")
    for ch in FADING:
        for an, x in res["ablations"][ch].items():
            lines.append(f"{ch:14s} {an:22s} " + "  ".join(
                f"{f(v)} ({'-' if x['delta_db'][t] is None else format(x['delta_db'][t], '+.1f')})"
                for t, v in x["esno_db_for"].items()))
    lines.append("\nPolicies at fixed Es/N0: mean freeze s/min and bitrate over the envelope, per channel")
    for pn in res["policies"][CHANNELS[0]]:
        lines.append(f"{pn:28s} " + "  ".join(
            f"{res['policies'][ch][pn]['freeze_s_per_min_mean']:6.2f} {res['policies'][ch][pn]['bitrate_over_envelope']:5.2f}"
            for ch in CHANNELS))
    pg = [float(s) for s in pgrid]
    show = [s for s in pg if s >= 0 and s <= 32 and abs(s / 2 - round(s / 2)) < 1e-9]
    lines.append(f"\n{CHOSEN} with the custom rules: freeze s/min / Mbit/s per mean Es/N0")
    lines.append(f"{'Es/N0':14s} " + " ".join(f"{s:>9.0f}" for s in show))
    for ch in CHANNELS:
        p = res["policies"][ch][CHOSEN]
        cells = []
        for s in show:
            i = pg.index(s)
            cells.append(f"{p['freeze_s_per_min'][i]:4.1f}/{p['mean_bitrate_bps'][i] / 1e6:4.1f}")
        lines.append(f"{ch:14s} " + " ".join(f"{c:>9s}" for c in cells))
    lines.append(f"\nFlight ({res['flight_profile']['seconds']} s, Es/N0 {res['flight_profile']['esno_db_min']} to "
                 f"{res['flight_profile']['esno_db_max']} dB): freeze s/min per channel, mean of channels, Mbit/s, MCS changes/min")
    for pn, sc in sorted(score.items(), key=lambda kv: kv[1]["freeze_s_per_min"]):
        lines.append(f"{pn:28s} " + " ".join(f"{res['flight'][ch][pn]['freeze_s_per_min']:6.2f}" for ch in FADING)
                     + f"  | {sc['freeze_s_per_min']:6.2f} {sc['mean_bitrate_bps'] / 1e6:6.2f} {sc['mcs_changes_per_min']:6.1f}")
    text = "\n".join(lines)
    print(text)
    res["chosen_policy"] = CHOSEN
    out_name = NAME + (".quick" if profile == "quick" else "")
    plot(res, out_name)
    write_result(out_name, {
        "profile": profile, "profile_params": prof, "freeze_ok_s_per_min": FREEZE_OK, "fractions": FRACTIONS,
        "targets_bps": TARGETS, "warmup_s": WARMUP_S,
        "configs": {k: v.as_dict() for k, v in {**RULES, **{f"ablation {a}": c for a, c in ABLATIONS.items()}}.items()},
        "policies": {k: [v[0], v[1]] for k, v in POLICIES.items()},
        "results": res, "summary": text, "seconds": round(time.time() - t_all)})


if __name__ == "__main__":
    main()
