"""The M9 board run of the whole radio on the chip (rtl/onchip): the
transmitter, the emulated air with a carrier offset, the synchroniser and
the receiver on the Arty Z7-20, against harness/onchip.OnChipLink.

  python -I tools/board_onchip.py job     build BOARD_WORK/job.bin (points below, the model's counts)
  python -I tools/board_onchip.py run     program the board and run the job (fpga/board.tcl)
  python -I tools/board_onchip.py check   compare, write results/e021_board_onchip.json

The bitstream comes from `make fpga FPGA_TOP=onchip_axi FPGA_OUT=out_onchip`
(BUILD_DIR/out_onchip), the ARM program from fpga/arm (the same job format
as tools/board.py; its result words are rtl/onchip's counters).
"""
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.loopback import REAL_TIME_PACE, frame_seconds     # noqa: E402
from harness.onchip import OC_COUNTERS, OnChipLink, cfo_inc_of  # noqa: E402
from harness.sweep import write_result                         # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ              # noqa: E402
from tools import board                                        # noqa: E402

BUILD = pathlib.Path(os.environ.get("BUILD_DIR", "/mnt/e/phy-sim-build"))
board.WORK = pathlib.Path(os.environ.get("BOARD_WORK", BUILD / "board_onchip"))
board.OUT = pathlib.Path(os.environ.get("BOARD_OUT", BUILD / "out_onchip"))
MCS = {m.name: m for m in MCS_TABLE}

# (MCS, Es/N0, carrier offset, frames): each MCS near its 1 % point (e001),
# the offsets across the synchroniser's +-116 kHz
POINTS = [("qpsk-1/2", 1.66, 20e3, 2000), ("qpsk-3/4", 4.59, -60e3, 2000), ("16qam-1/2", 6.78, 100e3, 2000),
          ("16qam-3/4", 10.57, -100e3, 2000), ("64qam-2/3", 14.51, 50e3, 2000), ("64qam-3/4", 15.98, -30e3, 2000),
          ("64qam-5/6", 17.55, 5e3, 2000)]


def build_job(pace: int = REAL_TIME_PACE) -> list[dict]:
    points, words = [], []
    for name, esno, cfo, nframes in POINTS:
        link = OnChipLink(NUM_10MHZ, MCS[name], cfo_inc=cfo_inc_of(cfo))
        w = link.board_words(esno, pace)
        ws = w["rx"] + w["tx"] + w["sync"] + w["ref"] + w["regs"]
        exp = link.run(nframes, esno)
        points.append({"mcs": name, "esno_db": esno, "cfo_hz": cfo, "nframes": nframes, "nwords": len(ws),
                       "pace": pace, "expected": exp})
        words.append(np.array(ws, np.uint32).reshape(-1))
        print(f"{name} {esno:6.2f} dB {cfo / 1e3:+5.0f} kHz: {nframes} frames, model {exp}", flush=True)
    head = 4 + 4 * len(points)
    hdr = [board.JOB_MAGIC, len(points), 0, 0]
    off = head
    for p, w in zip(points, words):
        hdr += [off, p["nwords"], p["nframes"], 0]
        off += len(w)
    board.WORK.mkdir(parents=True, exist_ok=True)
    blob = np.concatenate([np.array(hdr, np.uint32)] + words).astype("<u4")
    (board.WORK / "job.bin").write_bytes(blob.tobytes())
    (board.WORK / "job.json").write_text(json.dumps(points, indent=1))
    print(f"job: {len(points)} points, {blob.nbytes / 1e6:.1f} MB")
    return points


def check() -> bool:
    fclk_mhz = float((board.OUT / "fclk_mhz.txt").read_text())
    points = json.loads((board.WORK / "job.json").read_text())
    res = np.frombuffer((board.WORK / "results.bin").read_bytes(), "<u4").reshape(-1, 16)
    ok, rows = True, []
    for p, r in zip(points, res):
        got = {k: int(r[a - 16]) for k, a in OC_COUNTERS.items()}
        e = p["expected"]
        match = (int(r[15]) == board.RES_MAGIC and got["frames"] == e["synced"] and got["sync_overruns"] == 0
                 and got["frame_errors"] + got["missed"] == e["frame_errors"]
                 and all(got[k] == e[k] for k in ("missed", "spurious", "codeword_errors", "bit_errors", "iterations")))
        ok &= match
        clocks = got["clocks_lo"] + (got["clocks_hi"] << 32)
        rate = p["nframes"] / (clocks / (fclk_mhz * 1e6)) if clocks else None
        rows.append({**p, "board": got, "match": match, "board_frames_per_s": rate})
        print(f"{p['mcs']:10s} {p['esno_db']:6.2f} dB {p['cfo_hz'] / 1e3:+5.0f} kHz  board errors "
              f"{got['frame_errors'] + got['missed']:5d}/{p['nframes']} (missed {got['missed']})  model "
              f"{e['frame_errors']:5d}  {rate:7.0f} frames/s  {'match' if match else 'MISMATCH'}")
    air = 1 / (frame_seconds(NUM_10MHZ) * (4320 + 128) / 4032)       # with the preamble and the gap
    write_result("e021_board_onchip", {
        "points": rows, "all_match": ok, "fclk_mhz": fclk_mhz, "air_frames_per_s": air, "gate_met": ok,
        "summary": f"the whole radio on the chip, the air emulated with carrier offsets up to 100 kHz: counters "
                   f"{'equal to' if ok else 'DIFFERENT FROM'} the model at every point"})
    print("all points match" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("job", "all"):
        build_job()
    if cmd in ("run", "all"):
        board.run_board()
    if cmd in ("check", "all"):
        sys.exit(0 if check() else 1)
