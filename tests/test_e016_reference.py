"""The committed e016 reference run: per-frame decoding thresholds on a
continuous channel, the PHY input of the M6 link-layer model."""
import json
import math
import pathlib

import numpy as np
import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
CHANNELS = ("awgn", "tdl-d-600", "tdl-d-50", "tdl-c-300-600")


@pytest.fixture(scope="module")
def e016():
    path = RESULTS / "e016_phy_traces.json"
    if not path.exists():
        pytest.skip("run experiments/e016_phy_traces.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def test_traces_are_committed(e016):
    for ch in CHANNELS:
        d = np.load(RESULTS / "e016_traces" / f"{ch}.npz")
        assert d["thr"].shape[0] == 7
        n = d["thr"][0].size
        assert n == e016["results"][ch]["frames"]
        assert n * float(d["frame_s"]) > (14 if ch == "awgn" else 56)       # seconds of air time
        assert d["gain_db"].size == n


def test_threshold_model_predicts_fresh_decoding(e016):
    """FER from the thresholds against frames decoded with a new noise draw."""
    for ch, r in e016["results"].items():
        for m, x in r["mcs"].items():
            for c in x["check"]:
                p, n = c["fer_fresh"], c["frames"]
                sigma = math.sqrt(max(p * (1 - p), 1e-4) / n)
                assert abs(c["fer_trace"] - p) <= 4 * sigma + 0.003, (ch, m, c)
                # decoding is near monotone in the noise scale: outcomes that
                # disagree with the threshold on the same noise
                assert c["disagree"] <= 0.02, (ch, m, c)


def test_one_percent_points_match_e015(e016):
    """Same receiver, independent frames, a 0.5 dB grid there (and e015's
    LLR scale cap of 8: e016 runs at 16, which only moves high-rate TDL-C)."""
    for ch, r in e016["results"].items():
        for m, x in r["mcs"].items():
            ref = x.get("e015_esno_db_at_fer1")
            if ref is None or (ch == "tdl-c-300-600" and m.endswith("3/4")):
                continue
            assert abs(x["esno_db_at_fer1"] - ref) <= 0.6, (ch, m, x["esno_db_at_fer1"], ref)


def test_slow_fading_makes_bursts(e016):
    """At 50 Hz a fade spans tens of frames; at 600 Hz frames are close to independent."""
    slow = e016["results"]["tdl-d-50"]["mcs"]["qpsk-1/2"]["bursts_at_fer1"]
    fast = e016["results"]["tdl-d-600"]["mcs"]["qpsk-1/2"]["bursts_at_fer1"]
    assert slow["p_loss_after_loss"] > 0.6 and slow["mean_run"] > 3
    assert fast["p_loss_after_loss"] < 0.3 and fast["mean_run"] < 1.5


def test_no_floor_on_rate_three_quarters_with_cap_16(e016):
    """The LLR scale cap of 8 floored qpsk-3/4 and 16qam-3/4 on TDL-C (1.6 and
    0.4 % of frames never decoded); 16 removes it."""
    r = e016["results"]["tdl-c-300-600"]["mcs"]
    for m in ("qpsk-3/4", "16qam-3/4"):
        assert r[m]["esno_db_at_fer1"] is not None, m
        assert r[m]["never"] < 1e-3, (m, r[m]["never"])
    assert e016["fixed_point"]["llr_cap"] == 16.0


def test_awgn_points_near_e001(e016):
    e001 = {"qpsk-1/2": 1.41, "qpsk-3/4": 4.34, "16qam-1/2": 6.53, "16qam-3/4": 10.32,
            "64qam-2/3": 14.26, "64qam-3/4": 15.73, "64qam-5/6": 17.29}
    for m, x in e016["results"]["awgn"]["mcs"].items():
        # fixed point, synchronisation, PLL and SCO: within half a dB of Sionna's float chain
        assert -0.1 <= x["esno_db_at_fer1"] - e001[m] <= 0.5, (m, x["esno_db_at_fer1"])
