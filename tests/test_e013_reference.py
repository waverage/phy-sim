"""M4 gate on the committed e013 reference run: the fixed-point receiver's
FER curves lie within 0.3 dB of the same algorithm in floating point on
every configuration, at 1 % and at 10 % FER. All three receivers decoded
the same samples. The product algorithm's own cost against the genie
receiver (the e011 curves) is bounded too, so a worse estimator design
cannot slip in under the fixed-point gate. It is 0.1 dB or less up to
16-QAM and 0.6 to 1.0 dB on 64-QAM in fading, the most on TDL-C 300 ns
(no line of sight), where the genie's exact channel statistics are worth
the most against one fixed design; an estimator that picks its design
from the measured channel is M5 work."""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
ALGORITHM_BOUND_DB = 1.25


@pytest.fixture(scope="module")
def e013():
    path = RESULTS / "e013_fixed_point.json"
    if not path.exists():
        pytest.skip("run experiments/e013_fixed_point.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def curves(e013):
    for cfg, by_mcs in e013["results"].items():
        for m, r in by_mcs.items():
            yield f"{cfg}/{m}", r


def test_every_curve_reaches_1_percent(e013):
    for name, r in curves(e013):
        for rx in ("fixed", "float", "genie"):
            assert r[rx]["esno_db_at_fer1"] is not None, (name, rx)


def test_fixed_point_loss_under_tolerance(e013):
    tol = e013["tolerance_db"]
    assert tol <= 0.3
    for name, r in curves(e013):
        assert abs(r["fixed_loss_db"]) <= tol, (name, r["fixed_loss_db"])
        assert abs(r["fixed_loss_db_at_fer10"]) <= tol, (name, r["fixed_loss_db_at_fer10"])


def test_product_algorithm_close_to_genie(e013):
    for name, r in curves(e013):
        assert r["algorithm_loss_db"] <= ALGORITHM_BOUND_DB, (name, r["algorithm_loss_db"])


def test_fixed_point_matches_the_modem_defaults(e013):
    import sys
    sys.path.insert(0, str(RESULTS.parent))
    from harness.fxmodem import FixedPoint
    assert e013["fixed_point"] == FixedPoint().as_dict()
