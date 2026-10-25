"""e011: the Zig modem (M3, floating point) overlaid on the Sionna chain.

M3 gate: the Zig TX and RX chain gives the same FER curves as Sionna's
OfdmLink within 0.2 dB. Every frame here is transmitted by the Zig
modem, sent through Sionna's channel, impairments and noise, and decoded
by both receivers from the same samples (common random numbers), so the
difference between the two curves is the receivers alone, not Monte Carlo
noise. The transmitters are compared sample by sample in
tests/test_m3_modem.py (agree to f32 rounding).

Configurations cover what M2 decided the receiver must do:
- AWGN, every MCS: mapper, interleaver, OFDM, LMMSE estimate, LLRs, LDPC.
- TDL-D 100 ns at 5.8 GHz, 600 Hz Doppler: the 2D estimator under fading.
- The same plus the 5.8 GHz PLL and 100 Hz residual CFO with the linear
  phase tracker: the full M2 receiver.
- 2000 Hz Doppler with three pilot symbols (1, 6, 11): the fast-flight
  frame format.

The Zig decoder stops early on a zero syndrome (Sionna always runs 20
iterations); the report carries the average iterations and both
receivers' wall time.
Run:  .venv/bin/python -I experiments/e011_zig_overlay.py [--profile quick|reference]
"""
import dataclasses
import functools
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import impairments as imp                      # noqa: E402
from harness.modem import ZigOfdmLink                       # noqa: E402
from harness.sweep import (default_device, profile_arg, reference_esno, run_sweep, snr_at_fer,  # noqa: E402
                           snr_grid, sweep_kwargs, write_result)
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

NAME = "e011_zig_overlay"
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
TDL600 = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0, carrier_hz=5.8e9)
ALL = [m.name for m in MCS_TABLE]
FADING = ["qpsk-1/2", "16qam-1/2", "64qam-3/4", "64qam-5/6"]
CONFIGS = {
    # name: (numerology, OfdmLink kwargs, MCS, start offset from the e001 10 % point)
    "awgn": (NUM_10MHZ, dict(channel="awgn"), ALL, -1.0),
    "tdl-d-600": (NUM_10MHZ, TDL600, FADING, 1.5),
    "tdl-d-600-pll-cfo100-lin": (NUM_10MHZ, dict(
        TDL600, phase_track="linear",
        channel_impairments=imp.chain(functools.partial(imp.pll_phase_noise, inband_dbc_hz=-92.0),
                                      functools.partial(imp.cfo, offset_hz=100.0))),
        ["16qam-1/2", "64qam-3/4"], 1.5),
    "tdl-d-2000-p3": (P3, dict(TDL600, doppler_hz=2000.0), ["16qam-1/2", "64qam-3/4"], 1.5),
}
TOL_DB = 0.2


def summarise(points: list[dict], prefix: str = "") -> dict:
    pts = [{"esno_db": p["esno_db"], "fer": p[f"{prefix}frame_errors"] / p["frames"]} for p in points]
    return {"esno_db_at_fer10": snr_at_fer(pts, 0.10), "esno_db_at_fer1": snr_at_fer(pts, 0.01)}


def main():
    profile = profile_arg()
    device = default_device()
    t0 = time.time()
    results = {}
    for cfg, (num, kw, mcs_names, offset) in CONFIGS.items():
        results[cfg] = {}
        for mcs in [m for m in MCS_TABLE if m.name in mcs_names]:
            print(f"{cfg} {mcs.name}")
            link = ZigOfdmLink(num, mcs, compare=True, device=device, **kw)
            pts = run_sweep(link, snr_grid(reference_esno(mcs.name) + offset, 14.0, profile),
                            **sweep_kwargs(profile))
            zig, ref = summarise(pts), summarise(pts, "sionna_")
            frames = sum(p["frames"] for p in pts)
            it, cw = link.zig.decoder_stats()
            r = {
                "points": pts, "zig": zig, "sionna": ref,
                "delta_db_at_fer1": (zig["esno_db_at_fer1"] - ref["esno_db_at_fer1"]
                                     if zig["esno_db_at_fer1"] is not None and ref["esno_db_at_fer1"] is not None
                                     else None),
                "disagree_codewords": sum(p["disagree_codewords"] for p in pts),
                "codewords": sum(p["codewords"] for p in pts),
                "zig_iterations_per_codeword": it / max(cw, 1),
                "zig_rx_frames_per_s": frames / sum(p["zig_rx_s"] for p in pts),
                "sionna_rx_frames_per_s": frames / sum(p["sionna_rx_s"] for p in pts),
            }
            results[cfg][mcs.name] = r
            d = r["delta_db_at_fer1"]
            print(f"  1 % FER: zig {zig['esno_db_at_fer1']}  sionna {ref['esno_db_at_fer1']}  "
                  f"delta {d if d is None else round(d, 3)}  disagree {r['disagree_codewords']}/{r['codewords']}  "
                  f"zig {r['zig_rx_frames_per_s']:.0f} fr/s, {r['zig_iterations_per_codeword']:.1f} it/cw")

    lines = [f"\n{'config':28s} {'MCS':10s} {'zig':>7s} {'sionna':>7s} {'delta':>6s} {'disagree':>9s} {'fr/s':>6s}"]
    for cfg, curves in results.items():
        for m, r in curves.items():
            z, s, d = r["zig"]["esno_db_at_fer1"], r["sionna"]["esno_db_at_fer1"], r["delta_db_at_fer1"]
            f = lambda v: "   -   " if v is None else f"{v:7.2f}"   # noqa: E731
            lines.append(f"{cfg:28s} {m:10s} {f(z)} {f(s)} {'  -   ' if d is None else f'{d:+6.2f}'} "
                         f"{r['disagree_codewords'] / r['codewords']:9.4f} {r['zig_rx_frames_per_s']:6.0f}")
    text = "\n".join(lines)
    print(text)
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "device": device, "tolerance_db": TOL_DB,
        "configs": {c: {"numerology": n.as_dict(), "kwargs": {k: (v if isinstance(v, (int, float, str)) else repr(v))
                                                               for k, v in kw.items()}, "mcs": m}
                    for c, (n, kw, m, _) in CONFIGS.items()},
        "results": results, "summary": text, "seconds": round(time.time() - t0),
    })


if __name__ == "__main__":
    main()
