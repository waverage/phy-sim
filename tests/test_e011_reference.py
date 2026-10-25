"""M3 gate on the committed e011 reference run: the Zig modem's FER curves
lie within 0.2 dB of the Sionna chain's on every configuration. Both
receivers decoded the same received samples, so the tolerance is not
eaten by Monte Carlo noise."""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def e011():
    path = RESULTS / "e011_zig_overlay.json"
    if not path.exists():
        pytest.skip("run experiments/e011_zig_overlay.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def curves(e011):
    for cfg, by_mcs in e011["results"].items():
        for m, r in by_mcs.items():
            yield f"{cfg}/{m}", r


def test_every_curve_reaches_1_percent(e011):
    for name, r in curves(e011):
        assert r["zig"]["esno_db_at_fer1"] is not None, name
        assert r["sionna"]["esno_db_at_fer1"] is not None, name


def test_zig_overlays_sionna_within_tolerance(e011):
    tol = e011["tolerance_db"]
    assert tol <= 0.2
    for name, r in curves(e011):
        assert abs(r["delta_db_at_fer1"]) <= tol, (name, r["delta_db_at_fer1"])
        z10, s10 = r["zig"]["esno_db_at_fer10"], r["sionna"]["esno_db_at_fer10"]
        assert abs(z10 - s10) <= tol, (name, z10, s10)


def test_receivers_disagree_on_few_codewords(e011):
    for name, r in curves(e011):
        assert r["disagree_codewords"] <= 0.01 * r["codewords"], name


def test_awgn_curves_match_e001(e011):
    """Independent check against the M1 baseline (AwgnLink, no OFDM, perfect
    CSI): the OFDM chain with the genie LMMSE estimate sits a little above
    it, never below, and the gap stays small on AWGN."""
    e001 = json.loads((RESULTS / "e001_awgn_baseline.json").read_text())["curves"]
    for m, r in e011["results"]["awgn"].items():
        gap = r["zig"]["esno_db_at_fer1"] - e001[m]["esno_db_at_fer1"]
        assert -0.3 < gap < 0.8, (m, gap)
