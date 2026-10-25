"""e015: synchronisation under stress, in fixed point (the M5 gate).

Every frame is the fixed-point transmitter's, behind its preamble, at a
random time in a buffer, with the oscillators' full offset (CFO uniform in
+-116 kHz, the same crystal's 20 ppm sampling offset, 5.8 GHz PLL noise),
through Sionna's channel, the AGC and the 12-bit ADC. The fixed-point
synchroniser (modem/src/fxsync.zig) finds it and hands the aligned frame
to the fixed-point receiver (M4).

1. Acquisition at the lowest MCS, qpsk-1/2, over its 1 % FER point on each
   channel: sync success (detected, timing within the backoff, CFO within
   1 kHz of the offset plus the Doppler centroid) and the FER after cold
   synchronisation (this frame's estimates only) against the genie (true
   start and offset), plus the same chain in floating point.
   Gate: success above 99 % at the operating point.
2. Frequency control at 16-QAM 1/2 and 64-QAM 3/4 (16-QAM 3/4 on TDL-C):
   cold, warm (a loop that
   has averaged many frames), aided (warm less the line-of-sight Doppler
   centroid, from telemetry) and the genie, with the M4 phase tracker at
   gain 0 and 0.5.
Run:  .venv/bin/python -I experiments/e015_sync_gate.py [--profile quick|reference]
"""
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.design import EstimatorDesign                  # noqa: E402
from harness.sweep import (default_device, profile_arg, run_sweep, snr_at_fer, snr_grid,  # noqa: E402
                           sweep_kwargs, write_result)
from harness.sync import SyncParams                         # noqa: E402
from harness.synclink import SyncLink                       # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

NAME = "e015_sync_gate"
MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
TDL600 = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0)
DESIGN = EstimatorDesign()
DESIGN_P3 = dataclasses.replace(DESIGN, doppler_hz=1500.0, track_gain=0.0)
CHANNELS = {
    # name: (numerology, link kwargs, design, {MCS: (grid start, span)})
    "awgn": (NUM_10MHZ, dict(channel="awgn"), DESIGN,
             {"qpsk-1/2": (-0.5, 4.5), "16qam-1/2": (5.0, 5.0), "64qam-3/4": (14.5, 5.0)}),
    "tdl-d-600": (NUM_10MHZ, TDL600, DESIGN,
                  {"qpsk-1/2": (3.5, 4.5), "16qam-1/2": (9.5, 5.0), "64qam-3/4": (19.5, 6.0)}),
    # 64-QAM 3/4 on TDL-C is nearly floored even with the genie (e013: 1 % at
    # 29 to 31 dB, 0.4 % at 39 dB), so its 1 % point says nothing; 16-QAM 3/4
    "tdl-c-300-600": (NUM_10MHZ, dict(TDL600, tdl_model="C", delay_spread_s=300e-9), DESIGN,
                      {"qpsk-1/2": (7.5, 4.5), "16qam-1/2": (13.0, 5.0), "16qam-3/4": (17.0, 7.0)}),
    "tdl-d-2000-p3": (P3, dict(TDL600, doppler_hz=2000.0), DESIGN_P3,
                      {"qpsk-1/2": (2.5, 4.5), "16qam-1/2": (8.5, 5.0), "64qam-3/4": (20.0, 6.0)}),
}
ACQ_SYNCS = ("sync", "genie")
FREQ_SYNCS = ("warm", "sync", "aided", "genie")
FREQ_RX = {"tg0": {"fixed": True, "track_gain": 0.0}, "tg05": {"fixed": True, "track_gain": 0.5}}


def curve(points, key):
    pts = [{"esno_db": p["esno_db"], "fer": p[key] / p["frames"]} for p in points]
    return {"esno_db_at_fer10": snr_at_fer(pts, 0.10), "esno_db_at_fer1": snr_at_fer(pts, 0.01)}


def main():
    profile = profile_arg()
    device = default_device()
    sp = SyncParams()
    t0 = time.time()
    results = {}
    for ch, (num, kw, design, grids) in CHANNELS.items():
        results[ch] = {}
        for mname, (start, span) in grids.items():
            acq = mname == "qpsk-1/2"
            rx = ({"fixed": {"fixed": True}, "float": {}} if acq else FREQ_RX)
            syncs = ACQ_SYNCS if acq else FREQ_SYNCS
            link = SyncLink(num, MCS[mname], design=design, sync=sp, device=device, receivers=rx, syncs=syncs, **kw)
            print(f"{ch} {mname}", flush=True)
            pts = run_sweep(link, snr_grid(start, span, profile), **sweep_kwargs(profile),
                            log=lambda s: print(s, flush=True))
            first = (syncs[0], next(iter(rx)))
            keys = {f"{s}_{r}": ("frame_errors" if (s, r) == first else f"{s}_{r}_frame_errors")
                    for s in syncs for r in rx}
            curves = {name: curve(pts, k) for name, k in keys.items()}
            r = {"doppler_centroid_hz": link.doppler_centroid_hz, "points": pts, "curves": curves}
            for p in pts:
                for tag in ("fixed", "float"):
                    if f"sync_{tag}_failed" in p:
                        p[f"sync_{tag}_success"] = 1 - p[f"sync_{tag}_failed"] / p["frames"]
            if acq:
                g = curves["genie_fixed"]["esno_db_at_fer1"]
                # the operating point: the last grid point at or below the genie's 1 % crossing
                below = [p for p in pts if g is not None and p["esno_db"] <= g + 1e-9]
                op = below[-1] if below else pts[0]
                r.update(operating_point_db=op["esno_db"], success_at_operating_point=op["sync_fixed_success"],
                         float_success_at_operating_point=op["sync_float_success"],
                         missed_at_operating_point=op["sync_fixed_missed"],
                         sync_fixed_frames_per_s=sum(p["frames"] for p in pts) / sum(p["sync_fixed_s"] for p in pts))
            for key, out in (("esno_db_at_fer1", "loss_db"), ("esno_db_at_fer10", "loss_db_at_fer10")):
                ref = curves["genie_fixed" if acq else "genie_tg0"][key]
                r[out] = {n: (None if c[key] is None or ref is None else c[key] - ref) for n, c in curves.items()}
            results[ch][mname] = r
            print("  1 % FER: " + "  ".join(f"{n} {c['esno_db_at_fer1'] if c['esno_db_at_fer1'] is None else round(c['esno_db_at_fer1'], 2)}"
                                           for n, c in curves.items()), flush=True)
            if acq:
                print(f"  operating point {r['operating_point_db']} dB: success {r['success_at_operating_point']:.4f}"
                      f" (float {r['float_success_at_operating_point']:.4f}), missed {r['missed_at_operating_point']},"
                      f" fixed sync {r['sync_fixed_frames_per_s']:.0f} frames/s", flush=True)

    lines = ["\nEs/N0 of the genie (true start and offset) at 1 % FER, and each synchronisation's loss against it"
             " in dB at 1 % / 10 % FER"]
    f = lambda v: "-" if v is None else f"{v:+.2f}"   # noqa: E731
    for ch, by in results.items():
        for m, r in by.items():
            ls = "  ".join(f"{n} {f(v)}/{f(r['loss_db_at_fer10'][n])}" for n, v in r["loss_db"].items()
                           if not n.startswith("genie"))
            extra = f"  success {r['success_at_operating_point']:.4f} at {r['operating_point_db']} dB" if "operating_point_db" in r else ""
            lines.append(f"{ch:15s} {m:10s} genie {r['curves']['genie_fixed' if 'genie_fixed' in r['curves'] else 'genie_tg0']['esno_db_at_fer1']}  {ls}{extra}")
    text = "\n".join(lines)
    print(text)
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "device": device, "sync_params": sp.as_dict(), "cfo_hz": 116e3, "sco_ppm": 20.0,
        "channels": {c: {"numerology": n.as_dict(), "design": d.as_dict(),
                         "kwargs": {k: (v if isinstance(v, (int, float, str)) else repr(v)) for k, v in kw.items()}}
                     for c, (n, kw, d, _) in CHANNELS.items()},
        "results": results, "summary": text, "seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
