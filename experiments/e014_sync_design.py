"""e014: the synchroniser's design in floating point (M5).

Every frame now has a one-slot preamble (harness/sync.py) and arrives at
a random time in a sample buffer with the full oscillator offset: CFO
uniform in +-116 kHz (20 ppm at 5.8 GHz), the same crystal's 20 ppm
sampling clock offset and the 5.8 GHz PLL's phase noise, over AWGN, TDL-D
100 ns at 600 Hz, TDL-C 300 ns at 600 Hz and TDL-D at 2000 Hz with three
pilot symbols. The receiver is the M4 product algorithm in floating point.

A. Acquisition at the lowest MCS (qpsk-1/2) around its 1 % FER point:
   missed frames, timing and CFO errors, and the FER after the cold
   synchroniser against the genie (true start and carrier offset).
B. False alarms on noise alone.
C. Frequency control at higher MCS: per frame (cold) against a loop that
   averages many frames (warm), the loop with the line-of-sight Doppler
   predicted and subtracted (aided), and the genies, each with the M4
   tracker at gain 0 and 0.5.
Run:  .venv/bin/python -I experiments/e014_sync_design.py [--profile quick|reference]
"""
import dataclasses
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.design import EstimatorDesign                  # noqa: E402
from harness.fxmodem import FixedPoint                      # noqa: E402
from harness.sweep import default_device, profile_arg, snr_at_fer, write_result   # noqa: E402
from harness.sync import FloatSync, FxSync, SyncParams, papr_db, preamble   # noqa: E402
from harness.synclink import SyncLink                       # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

NAME = "e014_sync_design"
MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
TDL600 = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0)
DESIGN = EstimatorDesign()
DESIGN_P3 = dataclasses.replace(DESIGN, doppler_hz=1500.0, track_gain=0.0)
CHANNELS = {
    # name: (numerology, link kwargs, design, qpsk-1/2 grid start, (MCS, Es/N0) for part C)
    "awgn": (NUM_10MHZ, dict(channel="awgn"), DESIGN, -0.5, [("16qam-1/2", 6.5), ("64qam-3/4", 16.0)]),
    "tdl-d-600": (NUM_10MHZ, TDL600, DESIGN, 3.5, [("16qam-1/2", 10.5), ("64qam-3/4", 21.5)]),
    "tdl-c-300-600": (NUM_10MHZ, dict(TDL600, tdl_model="C", delay_spread_s=300e-9), DESIGN, 6.5,
                      [("16qam-1/2", 14.5), ("64qam-3/4", 29.0)]),
    "tdl-d-2000-p3": (P3, dict(TDL600, doppler_hz=2000.0), DESIGN_P3, 2.5, [("16qam-1/2", 10.0), ("64qam-3/4", 21.5)]),
}
PROFILES = {"quick": dict(batches_a=4, points_a=4, batches_c=4, noise_buffers=500),
            "reference": dict(batches_a=40, points_a=5, batches_c=20, noise_buffers=4000)}
BATCH = 250
RX = {"tg0": {"track_gain": 0.0}, "tg05": {"track_gain": 0.5}}


def accumulate(link, esno_db, batches):
    tot, sync = {}, {k: [] for k in ("dt", "df_hz", "detected", "coarse_df_hz", "metric", "cir_ratio")}
    for _ in range(batches):
        out, st = link.run(BATCH, esno_db, return_sync=True)
        s = st["float"]
        for k, v in out.items():
            tot[k] = tot.get(k, 0) + v
        for k in sync:
            sync[k].append(s[k])
    return tot, {k: np.concatenate(v) for k, v in sync.items()}


def sync_stats(s: dict, backoff: int) -> dict:
    det = s["detected"]
    dt, df, cf = s["dt"][det], s["df_hz"][det], s["coarse_df_hz"][det]
    ok = det & (np.abs(s["dt"]) <= backoff - 1) & (np.abs(s["df_hz"]) <= 1000)
    return {
        "frames": int(det.size), "missed": int((~det).sum()), "success_rate": float(ok.mean()),
        "timing_error_hist": {int(k): int(v) for k, v in zip(*np.unique(dt, return_counts=True))},
        "cfo_error_rms_hz": float(np.sqrt((df ** 2).mean())), "cfo_error_p999_hz": float(np.percentile(np.abs(df), 99.9)),
        "cfo_error_max_hz": float(np.abs(df).max()), "coarse_cfo_error_rms_hz": float(np.sqrt((cf ** 2).mean())),
        "coarse_cfo_error_max_hz": float(np.abs(cf).max()),
        "metric_p1": float(np.percentile(s["metric"], 1)), "cir_ratio_p1": float(np.percentile(s["cir_ratio"], 1)),
    }


def false_alarms(num, n_buffers, sp: SyncParams, length=8192, seed=7):
    """Noise-only buffers at the AGC's level: how many cross the detection
    threshold, and how many of those also pass the CIR check, in floating
    point and through the 12-bit ADC into the fixed-point synchroniser."""
    rng = np.random.default_rng(seed)
    sync = FloatSync(num, sp)
    fxs = FxSync(num, sp)
    sigma = np.sqrt(FixedPoint().agc_power / 2)
    hi = 2 ** (FixedPoint().adc_bits - 1)
    trig = passed = trig_fx = passed_fx = 0
    mmax = []
    for _ in range(n_buffers // 100):
        r = (rng.standard_normal((100, length)) + 1j * rng.standard_normal((100, length))) / np.sqrt(2)
        res = sync(r)
        trig += int(sync.detect(r)[0].sum())
        passed += int(res.detected.sum())
        # the metric's largest value anywhere in the buffer
        c = r[:, 32:] * r[:, :-32].conj()
        e = np.abs(r[:, 32:]) ** 2 + np.abs(r[:, :-32]) ** 2
        w = sp.window
        Ps = np.cumsum(c, -1)[:, w:] - np.cumsum(c, -1)[:, :-w]
        Es = np.cumsum(e, -1)[:, w:] - np.cumsum(e, -1)[:, :-w]
        mmax.append(float((4 * np.abs(Ps) ** 2 / Es ** 2).max()))
        q = np.stack([r.real, r.imag], -1) * sigma * np.sqrt(2)
        o = fxs.raw(np.clip(np.floor(q + 0.5), -hi, hi - 1).astype(np.int16))
        trig_fx += int(((o["detected"] & 1) != 0).sum())
        passed_fx += int((o["detected"] == 3).sum())
    return {"buffers": n_buffers, "samples": n_buffers * length, "triggered": trig, "passed_cir": passed,
            "fixed_triggered": trig_fx, "fixed_passed_cir": passed_fx, "metric_max": float(np.max(mmax)),
            "threshold": sp.threshold}


def main():
    profile = profile_arg()
    prof = PROFILES[profile]
    device = default_device()
    sp = SyncParams()
    t0 = time.time()
    pre = preamble(NUM_10MHZ)
    out = {"preamble": {"length": len(pre), "papr_db": papr_db(pre[:32])}, "acquisition": {}, "frequency": {}}
    print(f"preamble {len(pre)} samples, PAPR {out['preamble']['papr_db']:.2f} dB", flush=True)

    for ch, (num, kw, design, start, _) in CHANNELS.items():
        link = SyncLink(num, MCS["qpsk-1/2"], design=design, sync=sp, device=device, receivers={"float": {}},
                        syncs=("sync", "genie"), **kw)
        pts = []
        for i in range(prof["points_a"]):
            es = start + i
            np.random.seed(i); torch.manual_seed(i)
            tot, s = accumulate(link, es, prof["batches_a"])
            st = sync_stats(s, sp.backoff)
            pt = {"esno_db": es, "fer_sync": tot["frame_errors"] / tot["frames"],
                  "fer_genie": tot["genie_float_frame_errors"] / tot["frames"], **st}
            pts.append(pt)
            print(f"A {ch} {es:5.1f} dB  FER sync {pt['fer_sync']:.4f} genie {pt['fer_genie']:.4f}  missed {st['missed']}"
                  f"  success {st['success_rate']:.4f}  cfo rms {st['cfo_error_rms_hz']:.0f} max {st['cfo_error_max_hz']:.0f}"
                  f"  coarse max {st['coarse_cfo_error_max_hz']:.0f}  dt {st['timing_error_hist']}", flush=True)
        fs = [{"esno_db": p["esno_db"], "fer": p["fer_sync"]} for p in pts]
        fg = [{"esno_db": p["esno_db"], "fer": p["fer_genie"]} for p in pts]
        out["acquisition"][ch] = {"doppler_centroid_hz": link.doppler_centroid_hz, "points": pts,
                                  "esno_db_at_fer1_sync": snr_at_fer(fs, 0.01), "esno_db_at_fer1_genie": snr_at_fer(fg, 0.01)}

    out["false_alarms"] = false_alarms(NUM_10MHZ, prof["noise_buffers"], sp)
    print("B false alarms", out["false_alarms"], flush=True)

    syncs = ("sync", "warm", "aided", "genie", "genie_afc")
    for ch, (num, kw, design, _, pts) in CHANNELS.items():
        out["frequency"][ch] = {}
        for mname, es in pts:
            link = SyncLink(num, MCS[mname], design=design, sync=sp, device=device, receivers=RX, syncs=syncs, **kw)
            np.random.seed(11); torch.manual_seed(11)
            tot, _ = accumulate(link, es, prof["batches_c"])
            fr = tot["frames"]
            fer = {"sync_tg0": tot["frame_errors"] / fr}
            fer.update({k[:-len("_frame_errors")]: v / fr for k, v in tot.items()
                        if k.endswith("_frame_errors") and k != "frame_errors"})
            out["frequency"][ch][mname] = {"esno_db": es, "frames": fr, "fer": fer}
            print(f"C {ch} {mname} {es} dB  " + "  ".join(f"{k} {v:.4f}" for k, v in fer.items()), flush=True)

    out["seconds"] = round(time.time() - t0)
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "device": device, "sync_params": sp.as_dict(), "cfo_hz": 116e3, "batch": BATCH,
        "channels": {c: {"numerology": n.as_dict(), "design": d.as_dict(),
                         "kwargs": {k: (v if isinstance(v, (int, float, str)) else repr(v)) for k, v in kw.items()}}
                     for c, (n, kw, d, _, _) in CHANNELS.items()},
        **out})


if __name__ == "__main__":
    main()
