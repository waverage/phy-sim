"""A small board job for checking the JTAG flow (tools/board.py): two
points of a few frames, with the expected counts from the model.
  python -I tools/board_smoke.py [pace]   then  tools/board.py run / peek with BOARD_WORK=.../board_smoke
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.loopback import EmuLink, board_words                  # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ                  # noqa: E402
from tools import board                                            # noqa: E402

POINTS = (("qpsk-1/2", 1.0, 20), ("64qam-3/4", 15.75, 10))


def main(pace: int = 0):
    mcs = {m.name: m for m in MCS_TABLE}
    points, words = [], []
    for name, esno, nframes in POINTS:
        emu = EmuLink(NUM_10MHZ, mcs[name])
        w = board_words(emu, esno, pace)
        ws = w["rx"] + w["tx"] + w["ref"] + w["regs"]
        exp = emu.run(nframes, esno)
        points.append({"mcs": name, "esno_db": esno, "nframes": nframes, "nwords": len(ws), "pace": pace,
                       "expected": {k: exp[k] for k in ("frames", "frame_errors", "codeword_errors", "bit_errors")}})
        words.append(np.array(ws, np.uint32).reshape(-1))
    hdr = [board.JOB_MAGIC, len(points), 0, 0]
    off = 4 + 4 * len(points)
    for p, w in zip(points, words):
        hdr += [off, p["nwords"], p["nframes"], 0]
        off += len(w)
    board.WORK.mkdir(parents=True, exist_ok=True)
    blob = np.concatenate([np.array(hdr, np.uint32)] + words).astype("<u4")
    (board.WORK / "job.bin").write_bytes(blob.tobytes())
    (board.WORK / "job.json").write_text(json.dumps(points, indent=1))
    print(json.dumps(points, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
