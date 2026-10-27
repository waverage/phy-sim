"""The simulator viewer's capture (harness/viewer.py): every setting runs,
the result is plain JSON, and its error counts agree with the pictures."""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import viewer                                   # noqa: E402

CASES = [
    dict(esno_db=30.0),
    dict(esno_db=30.0, receiver="fixed"),
    dict(esno_db=25.0, channel="tdl-c", delay_spread_ns=300, doppler_hz=600, mcs="16qam-1/2", wifi=True),
    dict(esno_db=25.0, channel="tdl-d", doppler_hz=2000, pilots=3, sync="genie", frames=4),
]


@pytest.mark.parametrize("p", CASES, ids=lambda p: "-".join(f"{k}={v}" for k, v in p.items()))
def test_capture_is_clean_at_high_snr(p):
    out = viewer.capture(p)
    json.dumps(out, allow_nan=False)                         # what the server sends
    e = out["errors"]
    assert e["frames"] == p.get("frames", 1) and e["missed"] == 0
    assert e["frame_errors"] == 0 and e["bit_errors"] == 0
    assert e["raw_bit_errors"] < 0.05 * e["coded_bits"]      # the LLR sign convention is right (fades leave some)
    assert out["sync"]["detected"] and abs(out["sync"]["start_error"]) <= 2
    c = out["constellation"]
    assert len(c["eq"]) == len(c["wrong"]) == out["info"]["subcarriers"] * (
        out["info"]["symbols"] - len(out["info"]["pilot_symbols"]))
    if not p.get("wifi"):                                    # the burst caps it near its own SIR
        assert c["snr_db"] > p["esno_db"] - 8


def test_capture_is_repeatable_and_fails_at_low_snr():
    a = viewer.capture(dict(esno_db=-3.0, frames=20, seed=7))
    b = viewer.capture(dict(esno_db=-3.0, frames=20, seed=7))
    assert a["errors"] == b["errors"] and a["waveform"]["rx"] == b["waveform"]["rx"]
    assert a["errors"]["frame_errors"] == 20
    assert np.isclose(a["constellation"]["snr_db"], -3.0, atol=3) or not a["sync"]["detected"]
