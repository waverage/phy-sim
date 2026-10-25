"""Gate on the committed e006-e009 reference runs: complete, and the trends
are physical (worse impairment never helps beyond Monte Carlo noise). These
experiments answer design questions rather than define a baseline, so the
gate checks shape, not values."""
import json
import math
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
TOL_DB = 0.4                       # about 3 sigma of a reference-profile 1 % crossing


def load(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"run experiments/{name}.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r["configs"]


def s1(curve):
    """Es/N0 at 1 % FER, +inf for an error floor."""
    v = curve["esno_db_at_fer1"]
    return math.inf if v is None else v


def assert_not_better(cfgs, order, mcs_names, what):
    for m in mcs_names:
        for a, b in zip(order, order[1:]):
            assert s1(cfgs[b][m]) >= s1(cfgs[a][m]) - TOL_DB, \
                f"{what}: {b} needs less SNR than {a} on {m}"


def test_e006_channel_set():
    c = load("e006_channel_set")
    mcs = ("qpsk-1/2", "16qam-1/2", "64qam-3/4")
    assert set(c) == {"D100-600", "D100-1200", "D100-2000", "D100-2000-perfect",
                      "C100-600", "C300-600", "C300-600-perfect"}
    assert all(set(v) == set(mcs) for v in c.values())
    for m in mcs:                       # perfect CSI never loses to the estimate
        assert s1(c["D100-2000-perfect"][m]) <= s1(c["D100-2000"][m]) + TOL_DB, m
        assert s1(c["C300-600-perfect"][m]) <= s1(c["C300-600"][m]) + TOL_DB, m
    assert s1(c["D100-600"]["qpsk-1/2"]) < math.inf


def test_e007_phase_noise_monotonic():
    c = load("e007_phase_noise")
    order = ["none", "pn-100", "pn-94", "pn-88", "pn-82", "pn-76"]
    assert list(c) == order
    assert_not_better(c, order, ("16qam-3/4", "64qam-5/6"), "phase noise")


def test_e008_iq_adc_monotonic():
    c = load("e008_iq_adc")
    m = ("16qam-3/4", "64qam-5/6")
    assert_not_better(c, ["ideal", "iq0.25dB-1deg", "iq0.5dB-2deg", "iq1dB-4deg", "iq2dB-8deg"], m, "IQ")
    assert_not_better(c, ["adc12", "adc10", "adc8", "adc6"], m, "ADC bits")
    assert_not_better(c, ["adc12", "agc+3dB", "agc+6dB"], m, "AGC too hot")
    assert_not_better(c, ["adc12", "agc-3dB", "agc-6dB"], m, "AGC too cold")


def test_e009_wifi_burst_monotonic():
    c = load("e009_wifi_burst")
    m = ("qpsk-1/2", "16qam-1/2", "64qam-3/4")
    for burst in ("100us", "403.2us"):
        assert_not_better(c, ["clean"] + [f"{burst}-sir{s}" for s in (20, 10, 5, 0)], m, f"burst {burst}")
    for s in (20, 10, 5, 0):           # a longer burst never helps
        assert_not_better(c, [f"100us-sir{s}", f"403.2us-sir{s}"], m, f"SIR {s}")
