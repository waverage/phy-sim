"""e018: the board loopback's channel emulator against L2 (M8).

M8 gate: the FER measured on the board matches L2 within 0.3 dB. The board
runs the RTL receiver, which is bit exact with the Zig fixed-point
receiver (rtl/rx, cocotb), behind the RTL channel emulator, which is bit
exact with harness/chanemu.py; harness/loopback.EmuLink is therefore the
board's FER exactly, frame for frame (the board run checks that its counts
equal EmuLink's). What differs from L2 is the channel: counter-hashed
Box-Muller noise with a tabulated radius instead of Sionna's AWGN, an AGC
gain set from the expected signal and noise power instead of measured per
frame, and BOARD_FRAMES transmitted frames reused with fresh noise instead
of a new frame each time. This experiment measures that difference per MCS
under AWGN: L2 is FxOfdmLink's fixed receiver (the e013 "fixed" curve), the
board is EmuLink, on the same Es/N0 grid.
Run:  .venv/bin/python -I experiments/e018_board_emulator.py [--profile quick|reference]
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.fxmodem import FixedPoint, FxOfdmLink          # noqa: E402
from harness.loopback import BOARD_FRAMES, EmuLink          # noqa: E402
from harness.sweep import (default_device, profile_arg, reference_esno, run_sweep, snr_at_fer,  # noqa: E402
                           write_result)
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

NAME = "e018_board_emulator"
TOL_DB = 0.3
# A fine grid around e001's 1 % point: both curves must be resolved at 1 %,
# not interpolated across a 1 dB step of the waterfall.
PROFILES = {
    "quick":     dict(start=-0.75, span=1.75, step=0.5, batch=250, min_frame_errors=50, max_frames=2_000),
    "reference": dict(start=-0.75, span=1.75, step=0.25, batch=500, min_frame_errors=200, max_frames=20_000),
}


def curve(points):
    pts = [{"esno_db": p["esno_db"], "fer": p["frame_errors"] / p["frames"]} for p in points]
    return {"esno_db_at_fer10": snr_at_fer(pts, 0.10), "esno_db_at_fer1": snr_at_fer(pts, 0.01)}


def main():
    profile = profile_arg()
    device = default_device()
    fp = FixedPoint()
    t0 = time.time()
    results = {}
    pr = PROFILES[profile]
    kw = {k: pr[k] for k in ("batch", "min_frame_errors", "max_frames")}
    for mcs in MCS_TABLE:
        f1 = reference_esno(mcs.name, "esno_db_at_fer1")
        grid = np.round(np.arange(f1 + pr["start"], f1 + pr["start"] + pr["span"], pr["step"]), 3)
        print(mcs.name, flush=True)
        l2 = FxOfdmLink(NUM_10MHZ, mcs, fp=fp, device=device, genie=False, channel="awgn")
        pts_l2 = run_sweep(l2, grid, **kw, log=lambda s: print("  L2   " + s, flush=True))
        emu = EmuLink(NUM_10MHZ, mcs, fp=fp)
        pts_emu = run_sweep(emu, grid, **kw, log=lambda s: print("  emu  " + s, flush=True))
        c_l2, c_emu = curve(pts_l2), curve(pts_emu)
        d = (None if c_l2["esno_db_at_fer1"] is None or c_emu["esno_db_at_fer1"] is None
             else c_emu["esno_db_at_fer1"] - c_l2["esno_db_at_fer1"])
        results[mcs.name] = {"l2": {"points": pts_l2, **c_l2}, "board": {"points": pts_emu, **c_emu},
                             "board_minus_l2_db": d, "signal_power": emu.signal_power,
                             "settings": [emu.setting(float(s)) for s in grid]}
        print(f"  1 % FER: L2 {c_l2['esno_db_at_fer1']}  board {c_emu['esno_db_at_fer1']}  delta {d}", flush=True)

    lines = [f"\nEs/N0 at 1 % FER under AWGN, board (EmuLink) vs L2 (FxOfdmLink fixed); gate {TOL_DB} dB",
             f"{'MCS':10s} {'L2':>7s} {'board':>7s} {'delta':>7s}"]
    for m, r in results.items():
        v = lambda x: "    -  " if x is None else f"{x:7.2f}"    # noqa: E731
        dd = "    -  " if r["board_minus_l2_db"] is None else f"{r['board_minus_l2_db']:+7.2f}"
        lines.append(f"{m:10s} {v(r['l2']['esno_db_at_fer1'])} {v(r['board']['esno_db_at_fer1'])} {dd}")
    deltas = [r["board_minus_l2_db"] for r in results.values()]
    gate = all(x is not None and abs(x) <= TOL_DB for x in deltas)
    lines.append(f"gate: every |delta| <= {TOL_DB} dB: {'met' if gate else 'NOT met'}")
    text = "\n".join(lines)
    print(text)
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "profile_params": pr, "device": device, "tolerance_db": TOL_DB, "fixed_point": fp.as_dict(),
        "board_frames": BOARD_FRAMES, "numerology": NUM_10MHZ.as_dict(), "results": results, "gate_met": gate,
        "summary": text, "seconds": round(time.time() - t0),
    })


if __name__ == "__main__":
    main()
