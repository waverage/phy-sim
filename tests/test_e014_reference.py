"""The committed e014 reference run (the synchroniser's design in floating
point): the preamble's PAPR, acquisition at the lowest MCS, no false
alarms on noise, and the frequency-control comparison that set the M5
decisions (a warm loop, the tracker off behind it, the LoS Doppler aid)."""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def e014():
    path = RESULTS / "e014_sync_design.json"
    if not path.exists():
        pytest.skip("run experiments/e014_sync_design.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def test_preamble(e014):
    assert e014["preamble"]["length"] == 288
    assert e014["preamble"]["papr_db"] < 4.5


def test_acquisition_at_the_operating_point(e014):
    for ch, a in e014["acquisition"].items():
        g = a["esno_db_at_fer1_genie"]
        assert g is not None, ch
        op = [p for p in a["points"] if p["esno_db"] <= g][-1]
        assert op["success_rate"] > 0.99, (ch, op["esno_db"], op["success_rate"])
        assert op["missed"] / op["frames"] < 0.005, ch
        # the medium CFO stage leaves the tracker's +-1.9 kHz range unused by half
        assert op["cfo_error_p999_hz"] < 1500, ch
        # and the coarse stage stays inside the medium stage's +-19.5 kHz
        assert op["coarse_cfo_error_max_hz"] < 19.5e3 / 1.2, ch


def test_no_false_alarms(e014):
    fa = e014["false_alarms"]
    assert fa["samples"] >= 3e7
    assert fa["triggered"] == 0 and fa["fixed_triggered"] == 0
    # the largest metric anywhere in 33 M noise samples
    assert fa["metric_max"] < 0.75 * fa["threshold"]


def fer(e014, ch, mcs):
    return e014["frequency"][ch][mcs]["fer"]


def test_warm_loop_matches_genie_without_line_of_sight(e014):
    for ch in ("awgn", "tdl-c-300-600"):
        for m in e014["frequency"][ch]:
            f = fer(e014, ch, m)
            assert f["warm_tg0"] <= 1.3 * f["genie_tg0"] + 0.003, (ch, m, f)


def test_cold_sync_costs_at_high_mcs_in_fading(e014):
    """Why the loop is warm: per-frame estimates carry the channel's own
    Doppler. TDL-C 64-QAM loses several times the genie's FER cold."""
    f = fer(e014, "tdl-c-300-600", "64qam-3/4")
    assert f["sync_tg0"] > 2.5 * f["warm_tg0"]


def test_aided_loop_recovers_the_genie(e014):
    for ch, by in e014["frequency"].items():
        for m in by:
            f = fer(e014, ch, m)
            assert f["aided_tg0"] <= 1.3 * f["genie_tg0"] + 0.003, (ch, m, f)


def test_line_of_sight_doppler_removal_breaks_three_pilots_at_2000_hz(e014):
    f = fer(e014, "tdl-d-2000-p3", "64qam-3/4")
    assert f["genie_afc_tg0"] > 0.3 and f["genie_tg0"] < 0.03


def test_tracker_off_behind_a_warm_loop(e014):
    for ch, by in e014["frequency"].items():
        for m in by:
            f = fer(e014, ch, m)
            assert f["warm_tg0"] <= 1.15 * f["warm_tg05"] + 0.002, (ch, m, f)
