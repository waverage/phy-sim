"""The board runs: the loopback on the Arty Z7-20 against the model.

e018 measured the board's FER in software (harness/loopback.EmuLink, bit
exact with the RTL) and found it within the gate of L2. This script runs
the same SNR points with the same seeds and frame counts on the board and
requires the board's counters to equal e018's exactly: then the board's
FER is e018's, and the gate holds on the board.

  python -I tools/board.py job [pace]   build build/board/job.bin from results/e018_board_emulator.json
  python -I tools/board.py run          program the board and run the job (xsdb, Vivado's hardware server)
  python -I tools/board.py check        compare build/board/results.bin, write the result

pace 0 (M8, e019): the emulator as fast as the receiver takes the samples.
pace 10 (M9, e020): one sample every 10 clocks, the radio's 10 MS/s at
100 MHz; then no frame may be dropped and no sample lost, so the receiver
keeps up with the air, and the counters must still equal e018's.

The bitstream and the hardware platform come from fpga/build.tcl
(BUILD_DIR/out/sys.bit, sys.xsa and fclk_mhz.txt), the ARM program from fpga/arm.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.loopback import LB_COUNTERS, EmuLink, board_words, frame_seconds     # noqa: E402
from harness.sweep import write_result                             # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ                  # noqa: E402

BUILD = pathlib.Path(os.environ.get("BUILD_DIR", "/mnt/e/phy-sim-build"))
WORK = pathlib.Path(os.environ.get("BOARD_WORK", BUILD / "board"))
OUT = pathlib.Path(os.environ.get("BOARD_OUT", BUILD / "out"))       # fpga/build.tcl output
XSDB = os.environ.get("XSDB", r"E:\amd\2026.1\Vitis\bin\xsdb.bat")
JOB_MAGIC = 0x424A504E
RES_MAGIC = 0x52534C54
MCS = {m.name: m for m in MCS_TABLE}
E018 = ROOT / "results" / "e018_board_emulator.json"


def build_job(e018: pathlib.Path = E018, mcs_names=None, pace: int = 0) -> list[dict]:
    ref = json.loads(e018.read_text())
    points, words = [], []
    for name, r in ref["results"].items():
        if mcs_names and name not in mcs_names:
            continue
        emu = EmuLink(NUM_10MHZ, MCS[name])
        for i, p in enumerate(r["board"]["points"]):
            w = board_words(emu, p["esno_db"], pace)
            ws = w["rx"] + (w["tx"] + w["ref"] if i == 0 else []) + w["regs"]
            points.append({"mcs": name, "esno_db": p["esno_db"], "nframes": p["frames"], "nwords": len(ws), "pace": pace,
                           "expected": {k: p[k] for k in ("frames", "frame_errors", "codeword_errors", "bit_errors")}})
            words.append(np.array(ws, np.uint32).reshape(-1))
            print(f"{name} {p['esno_db']:6.2f} dB: {p['frames']} frames, {len(ws)} words", flush=True)
    head = 4 + 4 * len(points)
    hdr = [JOB_MAGIC, len(points), 0, 0]
    off = head
    for p, w in zip(points, words):
        hdr += [off, p["nwords"], p["nframes"], 0]
        off += len(w)
    WORK.mkdir(parents=True, exist_ok=True)
    blob = np.concatenate([np.array(hdr, np.uint32)] + words).astype("<u4")
    (WORK / "job.bin").write_bytes(blob.tobytes())
    (WORK / "job.json").write_text(json.dumps(points, indent=1))
    print(f"job: {len(points)} points, {blob.nbytes / 1e6:.1f} MB")
    return points


def winpath(p: pathlib.Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


def run_board():
    points = json.loads((WORK / "job.json").read_text())
    out = OUT
    with zipfile.ZipFile(out / "sys.xsa") as z:
        name = next(n for n in z.namelist() if n.endswith("ps7_init.tcl"))
        (WORK / "ps7_init.tcl").write_bytes(z.read(name))
    args = [winpath(x) for x in (ROOT / "fpga" / "board.tcl", out / "sys.bit", WORK / "ps7_init.tcl",
                                 BUILD / "arm" / "loopback.elf", WORK / "job.bin", WORK / "results.bin")]
    t0 = time.time()
    subprocess.run(["cmd.exe", "/c", XSDB, *args, str(len(points))], check=True)
    (WORK / "run.json").write_text(json.dumps({"seconds": round(time.time() - t0)}))


def check(write: bool = True) -> bool:
    fclk_mhz = float((OUT / "fclk_mhz.txt").read_text())
    points = json.loads((WORK / "job.json").read_text())
    res = np.frombuffer((WORK / "results.bin").read_bytes(), "<u4").reshape(-1, 16)
    ref = json.loads(E018.read_text())
    pace = points[0].get("pace", 0)
    air = 1 / frame_seconds(NUM_10MHZ)
    ok = True
    rows = []
    for p, r in zip(points, res):
        got = {k: int(r[a - 16]) for k, a in LB_COUNTERS.items()}
        valid = int(r[15]) == RES_MAGIC
        match = valid and all(got[k] == v for k, v in p["expected"].items())
        clean = got["dropped"] == 0 and got["lost"] == 0 and got["unmarked"] == 0
        ok &= match and clean
        clocks = got["clocks_lo"] + (got["clocks_hi"] << 32)
        rows.append({**p, "board": got, "match": match, "no_drops": clean,
                     "board_frames_per_s": got["frames"] / (clocks / (fclk_mhz * 1e6)) if clocks else None})
        print(f"{p['mcs']:10s} {p['esno_db']:6.2f} dB  board {got['frame_errors']:5d}/{got['frames']:6d}  "
              f"model {p['expected']['frame_errors']:5d}/{p['expected']['frames']:6d}  "
              f"{rows[-1]['board_frames_per_s']:7.0f} frames/s  dropped {got['dropped']} lost {got['lost']} "
              f"FIFO {got['fifo_high']}  {'match' if match and clean else 'MISMATCH'}")
    print("all points match" if ok else "MISMATCH")
    if not write:
        return ok
    deltas = {m: r["board_minus_l2_db"] for m, r in ref["results"].items()}
    common = {"points": rows, "all_match": ok, "fclk_mhz": fclk_mhz, "pace_clocks_per_sample": pace,
              "board_minus_l2_db": deltas, "tolerance_db": ref["tolerance_db"],
              "run": json.loads((WORK / "run.json").read_text()) if (WORK / "run.json").exists() else None}
    if pace:
        frames = sum(r["board"]["frames"] for r in rows)
        rates = [r["board_frames_per_s"] for r in rows]
        write_result("e020_board_realtime", {
            **common, "air_frames_per_s": air, "frames": frames,
            "dropped": sum(r["board"]["dropped"] for r in rows), "lost": sum(r["board"]["lost"] for r in rows),
            "fifo_high_max": max(r["board"]["fifo_high"] for r in rows),
            "gate_met": ok,
            "summary": f"emulator at {fclk_mhz / pace:.0f} MS/s: {frames} frames, "
                       f"{'no frame dropped, no sample lost, ' if ok else 'DROPS OR MISMATCHES, '}"
                       f"counters {'equal' if ok else 'differ from'} the model at every point; "
                       f"{min(rates):.0f} to {max(rates):.0f} frames/s with the pipeline's fill and drain "
                       f"(the air carries {air:.0f})",
        })
    else:
        write_result("e019_board_loopback", {
            **common, "gate_met": ok and ref["gate_met"],
            "summary": f"board counters {'equal' if ok else 'DIFFER FROM'} the model at every point; "
                       f"board - L2 at 1 % FER per MCS (e018): {deltas}",
        })
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    args = sys.argv[2:]
    pace = int(args.pop(0)) if args and args[0].isdigit() else 0
    if cmd in ("job", "all"):
        build_job(mcs_names=args or None, pace=pace)
    if cmd in ("run", "all"):
        run_board()
    if cmd in ("check", "all"):
        sys.exit(0 if check() else 1)
    if cmd == "peek":                     # compare and print, write no result (tools/board_smoke.py)
        sys.exit(0 if check(write=False) else 1)
