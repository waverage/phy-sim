"""Gate on the committed e010 reference run. Unlike e006-e009 these values
carry the pilot decision (no continual pilots below about 1500 Hz Doppler,
residual CFO tracked on the block pilots), so the gate checks the claims
that decision rests on, with a margin for Monte Carlo noise."""
import json
import math
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
TOL_DB = 0.4                       # about 3 sigma of a reference-profile 1 % crossing
OSC_MCS = ("16qam-3/4", "64qam-5/6")


@pytest.fixture(scope="module")
def e010():
    path = RESULTS / "e010_phase_tracking.json"
    if not path.exists():
        pytest.skip("run experiments/e010_phase_tracking.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r["oscillator"]["configs"], r["fast"]["configs"]


def s1(curve):
    """Es/N0 at 1 % FER, +inf for an error floor."""
    v = curve["esno_db_at_fer1"]
    return math.inf if v is None else v


def loss(cfgs, cfg, base, m):
    return s1(cfgs[cfg][m]) - s1(cfgs[base][m])


def test_tracker_is_free_when_there_is_nothing_to_track(e010):
    osc, _ = e010
    for m in OSC_MCS:
        assert loss(osc, "lin", "none", m) < TOL_DB, m


def test_ad9363_pll_needs_no_tracking(e010):
    osc, _ = e010
    for cfg in ("pll2.4", "pll5.8"):
        for m in OSC_MCS:
            assert loss(osc, cfg, "none", m) < 0.5, (cfg, m)


def test_residual_cfo_tracked_on_block_pilots(e010):
    osc, _ = e010
    assert loss(osc, "cfo100", "none", "64qam-5/6") > 1.0          # untracked it costs
    for cfg in ("cfo100-lin", "cfo300-lin"):
        for m in OSC_MCS:
            assert loss(osc, cfg, "none", m) < TOL_DB, (cfg, m)


def test_high_doppler_needs_more_pilots(e010):
    _, fast = e010
    assert s1(fast["D2000"]["64qam-3/4"]) == math.inf
    assert s1(fast["D2000-cp16"]["64qam-3/4"]) < math.inf


def test_whole_receiver_at_5g8(e010):
    _, fast = e010
    for m in ("16qam-1/2", "64qam-3/4"):
        assert loss(fast, "D600-pll-cfo100-lin", "D600", m) < 1.0, m
