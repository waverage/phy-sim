"""e013: the fixed-point product modem (M4) against floating point.

M4 gate: the fixed-point receiver loses less than 0.3 dB against the same
algorithm in floating point. Every frame is sent by the fixed-point
transmitter (12-bit DAC) through Sionna's channel, impairments and noise,
and decoded by three receivers from the same samples:
- fixed: the product receiver after an ideal per-frame AGC and a 12-bit
  ADC: 16-bit FFT, separable fixed estimator (harness/design.py), CORDIC
  phase tracker, division-free max-log demapper with 6-bit LLRs, 6/8-bit
  layered offset min-sum (e012);
- float: the same algorithm in floating point, unquantised input;
- genie: the M3 f32 receiver with the genie 2D LMMSE filter, the e011
  curves (now with the generated pilot and interleaver tables).
fixed - float is the fixed-point loss (the gate); float - genie is what the
product algorithm costs (fixed estimator, max-log, min-sum), reported.

Configurations are e011's, plus TDL-C at 300 ns, the worst channel of e006,
to show that the fixed estimator design generalises beyond TDL-D.
Run:  .venv/bin/python -I experiments/e013_fixed_point.py [--profile quick|reference]
"""
import dataclasses
import functools
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import impairments as imp                      # noqa: E402
from harness.design import EstimatorDesign                  # noqa: E402
from harness.fxmodem import FixedPoint, FxOfdmLink          # noqa: E402
from harness.sweep import (default_device, profile_arg, reference_esno, run_sweep, snr_at_fer,  # noqa: E402
                           snr_grid, sweep_kwargs, write_result)
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

NAME = "e013_fixed_point"
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
TDL600 = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0, carrier_hz=5.8e9)
ALL = [m.name for m in MCS_TABLE]
FADING = ["qpsk-1/2", "16qam-1/2", "64qam-3/4", "64qam-5/6"]
DESIGN = EstimatorDesign()
DESIGN_P3 = dataclasses.replace(DESIGN, doppler_hz=1500.0, track_gain=0.0)
CONFIGS = {
    # name: (numerology, OfdmLink kwargs, estimator design, MCS, start offset from e001's 10 % point, span)
    "awgn": (NUM_10MHZ, dict(channel="awgn"), DESIGN, ALL, -1.0, 8.0),
    "tdl-d-600": (NUM_10MHZ, TDL600, DESIGN, FADING, 1.5, 14.0),
    "tdl-d-600-pll-cfo100-lin": (NUM_10MHZ, dict(
        TDL600, phase_track="linear",
        channel_impairments=imp.chain(functools.partial(imp.pll_phase_noise, inband_dbc_hz=-92.0),
                                      functools.partial(imp.cfo, offset_hz=100.0))),
        DESIGN, ["16qam-1/2", "64qam-3/4"], 1.5, 14.0),
    "tdl-d-2000-p3": (P3, dict(TDL600, doppler_hz=2000.0), DESIGN_P3, ["16qam-1/2", "64qam-3/4"], 1.5, 14.0),
    "tdl-c-300-600": (NUM_10MHZ, dict(TDL600, tdl_model="C", delay_spread_s=300e-9), DESIGN,
                      ["16qam-1/2", "64qam-3/4"], 4.0, 20.0),
}
TOL_DB = 0.3
RECEIVERS = {"fixed": "", "float": "float_", "genie": "genie_"}


def summarise(points: list[dict], prefix: str) -> dict:
    pts = [{"esno_db": p["esno_db"], "fer": p[f"{prefix}frame_errors"] / p["frames"]} for p in points]
    return {"esno_db_at_fer10": snr_at_fer(pts, 0.10), "esno_db_at_fer1": snr_at_fer(pts, 0.01)}


def delta(a: dict, b: dict, key: str = "esno_db_at_fer1"):
    return None if a[key] is None or b[key] is None else a[key] - b[key]


def main():
    profile = profile_arg()
    device = default_device()
    fp = FixedPoint()
    t0 = time.time()
    results = {}
    for cfg, (num, kw, design, mcs_names, offset, span) in CONFIGS.items():
        results[cfg] = {}
        for mcs in [m for m in MCS_TABLE if m.name in mcs_names]:
            print(f"{cfg} {mcs.name}", flush=True)
            link = FxOfdmLink(num, mcs, design=design, fp=fp, device=device, **kw)
            pts = run_sweep(link, snr_grid(reference_esno(mcs.name) + offset, span, profile),
                            **sweep_kwargs(profile), log=lambda s: print(s, flush=True))
            curves = {r: summarise(pts, p) for r, p in RECEIVERS.items()}
            frames = sum(p["frames"] for p in pts)
            it, cw = link.fixed.decoder_stats()
            r = {
                "points": pts, **curves,
                "fixed_loss_db": delta(curves["fixed"], curves["float"]),
                "fixed_loss_db_at_fer10": delta(curves["fixed"], curves["float"], "esno_db_at_fer10"),
                "algorithm_loss_db": delta(curves["float"], curves["genie"]),
                "total_loss_db": delta(curves["fixed"], curves["genie"]),
                "disagree_codewords": sum(p["disagree_codewords"] for p in pts),
                "codewords": sum(p["codewords"] for p in pts),
                "fixed_iterations_per_codeword": it / max(cw, 1),
                **{f"{x}_rx_frames_per_s": frames / sum(p[f"{x}_rx_s"] for p in pts) for x in RECEIVERS},
            }
            results[cfg][mcs.name] = r
            f = lambda v: "-" if v is None else f"{v:+.2f}"   # noqa: E731
            print(f"  1 % FER: fixed {curves['fixed']['esno_db_at_fer1']}  float {curves['float']['esno_db_at_fer1']}"
                  f"  genie {curves['genie']['esno_db_at_fer1']}  fixed loss {f(r['fixed_loss_db'])}"
                  f"  algorithm {f(r['algorithm_loss_db'])}  {r['fixed_rx_frames_per_s']:.0f} fr/s", flush=True)

    lines = [f"\nEs/N0 at 1 % FER; fixed-point loss = fixed - float (gate {TOL_DB} dB), "
             "algorithm = float - genie",
             f"{'config':26s} {'MCS':10s} {'fixed':>7s} {'float':>7s} {'genie':>7s} {'fx loss':>8s} "
             f"{'algo':>6s} {'total':>6s} {'fr/s':>6s}"]
    for cfg, curves in results.items():
        for m, r in curves.items():
            v = lambda x: "    -  " if x is None else f"{x:7.2f}"   # noqa: E731
            d = lambda x: "    -  " if x is None else f"{x:+7.2f}"  # noqa: E731
            lines.append(f"{cfg:26s} {m:10s} {v(r['fixed']['esno_db_at_fer1'])} {v(r['float']['esno_db_at_fer1'])} "
                         f"{v(r['genie']['esno_db_at_fer1'])} {d(r['fixed_loss_db'])} "
                         f"{d(r['algorithm_loss_db'])[1:]} {d(r['total_loss_db'])[1:]} "
                         f"{r['fixed_rx_frames_per_s']:6.0f}")
    text = "\n".join(lines)
    print(text)
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "device": device, "tolerance_db": TOL_DB, "fixed_point": fp.as_dict(),
        "configs": {c: {"numerology": n.as_dict(), "design": d.as_dict(), "mcs": m,
                        "kwargs": {k: (v if isinstance(v, (int, float, str)) else repr(v)) for k, v in kw.items()}}
                    for c, (n, kw, d, m, _, _) in CONFIGS.items()},
        "results": results, "summary": text, "seconds": round(time.time() - t0),
    })


if __name__ == "__main__":
    main()
