"""e016: PHY traces for the link layer (M6).

Per channel, a minute of consecutive frames on a fading channel that is
continuous in time (harness/fading.py), and for every frame and MCS the
mean Es/N0 at which that frame decodes (harness/phytrace.py): the product
transmitter, preamble, oscillators (+-116 kHz, 20 ppm, PLL noise), the
fixed-point synchroniser with the warm frequency loop, and the fixed-point
receiver with the tracker off. e017 runs the link layer and the video
model on these traces.

Channels: AWGN; TDL-D 100 ns at 600 Hz (31 m/s at 5.8 GHz) and at 50 Hz
(2.6 m/s, slow flight or hover: fades last tens of frames); TDL-C 300 ns at
600 Hz (low altitude, no line of sight).

Checks written to the JSON:
  - the trace's FER against frames decoded directly at fixed Es/N0 with a
    fresh noise draw (the threshold model's error), and the fraction of
    frames whose outcome at a fixed Es/N0 disagrees with its threshold on
    the same noise (decoding is not exactly monotone in the noise scale);
  - the 1 % FER points against e015's (same receiver, independent frames);
  - loss bursts at each MCS's 1 % point: P(loss | previous frame lost),
    the mean run length, the share of losses in runs of 8 or more frames.

The traces go to results/e016_traces/<channel>.npz (thresholds in dB,
float16, frames in time order with independent links one after another).
Run:  .venv/bin/python -I experiments/e016_phy_traces.py [--profile quick|reference]
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.fxmodem import FixedPoint                     # noqa: E402
from harness.phytrace import AWGN_1PCT_DB, TraceConfig, TraceLink     # noqa: E402
from harness.sweep import RESULTS_DIR, default_device, profile_arg, write_result  # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords   # noqa: E402

NAME = "e016_phy_traces"
CHANNELS = {
    "awgn": TraceConfig(model="awgn", seed=10),
    "tdl-d-600": TraceConfig(model="D", delay_spread_s=100e-9, doppler_hz=600.0, seed=11),
    "tdl-d-50": TraceConfig(model="D", delay_spread_s=100e-9, doppler_hz=50.0, seed=12),
    "tdl-c-300-600": TraceConfig(model="C", delay_spread_s=300e-9, doppler_hz=600.0, seed=13),
}
# AWGN has no fading to sample: a quarter of the frames
FRAMES_SCALE = {"awgn": 0.25}
PROFILES = {
    # links side by side x consecutive frames per link; rows per block = links x block
    "quick": dict(links=8, frames=512, block=128, check_frames=256),
    "reference": dict(links=16, frames=8192, block=64, check_frames=1024),
}
# e015's matching curves (fixed receiver; qpsk: cold sync, others: warm loop, tracker off)
E015 = {"tdl-d-600": {"qpsk-1/2": "sync_fixed", "16qam-1/2": "warm_tg0", "64qam-3/4": "warm_tg0"},
        "tdl-c-300-600": {"qpsk-1/2": "sync_fixed", "16qam-1/2": "warm_tg0", "16qam-3/4": "warm_tg0"},
        "awgn": {"qpsk-1/2": "sync_fixed", "16qam-1/2": "warm_tg0", "64qam-3/4": "warm_tg0"}}
GRID = np.arange(-10, 46, 0.05)


def fer_curve(thr: np.ndarray) -> np.ndarray:
    """FER on GRID from per-frame thresholds."""
    t = np.sort(np.where(np.isfinite(thr), thr, np.inf))
    return 1 - np.searchsorted(t, GRID, side="right") / len(t)


def snr_at(fer: np.ndarray, target: float):
    i = np.nonzero(fer <= target)[0]
    return None if not len(i) else round(float(GRID[i[0]]), 2)


def bursts(lost: np.ndarray) -> dict:
    """lost [links, frames] bool."""
    a, b = lost[:, :-1], lost[:, 1:]
    p_cond = float((a & b).sum() / max(a.sum(), 1))
    runs = []
    for row in lost:
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        runs.append(e - s)
    runs = np.concatenate(runs) if runs else np.zeros(0)
    n = runs.sum()
    return {"loss_rate": float(lost.mean()), "p_loss_after_loss": round(p_cond, 4),
            "mean_run": round(float(runs.mean()), 3) if len(runs) else None,
            "share_in_runs_ge8": round(float(runs[runs >= 8].sum() / n), 4) if n else None}


def main():
    profile = profile_arg()
    prof = PROFILES[profile]
    device = default_device()
    t_all = time.time()
    out_dir = RESULTS_DIR / ("e016_traces" + (".quick" if profile == "quick" else ""))
    out_dir.mkdir(exist_ok=True)
    e015 = json.loads((RESULTS_DIR / "e015_sync_gate.json").read_text())["results"]
    results = {}
    for ch, base in CHANNELS.items():
        t0 = time.time()
        prof = dict(PROFILES[profile])
        prof["frames"] = int(prof["frames"] * FRAMES_SCALE.get(ch, 1.0)) // prof["block"] * prof["block"]
        cfg = TraceConfig(**{**base.as_dict(), "num_links": prof["links"]})
        tl = TraceLink(NUM_10MHZ, MCS_TABLE, cfg=cfg, device=device)
        names = tl.mcs
        thr = {m: np.zeros((prof["links"], prof["frames"]), np.float32) for m in names}
        gain = np.zeros((prof["links"], prof["frames"]), np.float32)
        # the check: the first check_frames of every link, at each MCS's AWGN 1 % point + channel margin
        check_pts = {m: [AWGN_1PCT_DB[m] + d for d in ((0.0, 1.0, 2.0) if cfg.model == "awgn" else (2.0, 4.0, 7.0))]
                     for m in names}
        chk = {m: {"same": [], "fresh": [], "thr": []} for m in names}
        for f0 in range(0, prof["frames"], prof["block"]):
            r = tl.block(f0, prof["block"], check_db=check_pts if f0 < prof["check_frames"] else None)
            for m in names:
                thr[m][:, f0:f0 + prof["block"]] = r["thr"][m]
                if "check" in r:
                    chk[m]["same"].append(r["check"][m]["same"])
                    chk[m]["fresh"].append(r["check"][m]["fresh"])
                    chk[m]["thr"].append(r["thr"][m])
            gain[:, f0:f0 + prof["block"]] = r["gain_db"]
            done = f0 + prof["block"]
            el = time.time() - t0
            print(f"{ch}: {done}/{prof['frames']} frames per link, {el:.0f} s, "
                  f"{tl.timing['decodes'] / max(tl.timing['decode'], 1e-9):.0f} decodes/s", flush=True)
        res = {"config": cfg.as_dict(), "frame_s": tl.frame_seconds, "frames": int(prof["links"] * prof["frames"]),
               "doppler_centroid_hz": tl.centroid * tl.fs, "mcs": {}}
        for m in names:
            t = thr[m]
            fer = fer_curve(t.reshape(-1))
            pts = {"esno_db_at_fer10": snr_at(fer, 0.10), "esno_db_at_fer1": snr_at(fer, 0.01),
                   "esno_db_at_fer01": snr_at(fer, 0.001), "never": float(np.mean(~np.isfinite(t))),
                   "payload_bytes": plan_codewords(NUM_10MHZ, next(x for x in MCS_TABLE if x.name == m)).info_bits_per_frame // 8}
            s1 = pts["esno_db_at_fer1"]
            if s1 is not None:
                pts["bursts_at_fer1"] = bursts(t > s1)
            c = chk[m]
            same, fresh, ct = (np.concatenate(c["same"], 2), np.concatenate(c["fresh"], 2), np.concatenate(c["thr"], 1))
            pts["check"] = [{"esno_db": s, "fer_trace": float((ct > s).mean()), "fer_fresh": float(1 - fresh[i].mean()),
                             "fer_same": float(1 - same[i].mean()), "disagree": float(((ct <= s) != same[i]).mean()),
                             "frames": int(ct.size)} for i, s in enumerate(check_pts[m])]
            if ch in E015 and m in E015[ch]:
                ref = e015[ch][m]["curves"][E015[ch][m]]["esno_db_at_fer1"]
                pts["e015_esno_db_at_fer1"] = ref
            res["mcs"][m] = pts
            print(f"  {m:10s} 10% {pts['esno_db_at_fer10']}  1% {s1}  0.1% {pts['esno_db_at_fer01']}"
                  f"  e015 {pts.get('e015_esno_db_at_fer1')}  bursts {pts.get('bursts_at_fer1')}", flush=True)
            for k in pts["check"]:
                print(f"    check {k['esno_db']:5.2f} dB: trace {k['fer_trace']:.4f} fresh {k['fer_fresh']:.4f}"
                      f" same {k['fer_same']:.4f} disagree {k['disagree']:.4f}", flush=True)
        res["seconds"] = round(time.time() - t0)
        res["decodes_per_s"] = round(tl.timing["decodes"] / max(tl.timing["decode"], 1e-9))
        results[ch] = res
        q = lambda x: np.where(np.isfinite(x), np.round(x / 0.05) * 0.05, np.inf).astype(np.float16)   # noqa: E731
        np.savez_compressed(out_dir / f"{ch}.npz", thr=np.stack([q(thr[m]) for m in names]),
                            gain_db=gain.astype(np.float16), mcs=np.array(names), frame_s=tl.frame_seconds,
                            config=json.dumps(cfg.as_dict()))
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "device": device, "profile_params": PROFILES[profile], "frames_scale": FRAMES_SCALE,
        "numerology": NUM_10MHZ.as_dict(), "fixed_point": FixedPoint().as_dict(),
        "results": results, "seconds": round(time.time() - t_all)})


if __name__ == "__main__":
    main()
