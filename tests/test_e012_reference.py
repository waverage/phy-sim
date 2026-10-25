"""The e012 decoder decision on its committed reference run: the max-log
demapper costs nothing, and the chosen fixed-point decoder (layered offset
min-sum, 6-bit LLRs and messages, 8-bit a-posteriori values, 0.25 nat per
LSB, offset 2 LSB, 20 iterations) is no worse than Sionna's flooding
sum-product with exact LLRs on any MCS. The 7-bit a-posteriori variant
shows why 8 bits: it floors."""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
CHOSEN = "q6/6/8 d0.25 o2"


@pytest.fixture(scope="module")
def e012():
    path = RESULTS / "e012_decoder_fixed.json"
    if not path.exists():
        pytest.skip("run experiments/e012_decoder_fixed.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def test_chosen_decoder_matches_the_modem_defaults(e012):
    import sys
    sys.path.insert(0, str(RESULTS.parent))
    from harness.fxmodem import FixedPoint
    fp, d = FixedPoint(), e012["decoders"][CHOSEN]
    assert (d["kwargs"]["llr_bits"], d["kwargs"]["msg_bits"], d["kwargs"]["app_bits"]) == (
        fp.llr_bits, fp.msg_bits, fp.app_bits)
    assert d["kwargs"]["offset"] == fp.offset_lsb and d["llr_step_nat"] == fp.llr_step_nat


def test_maxlog_costs_nothing(e012):
    for m, r in e012["results"].items():
        assert abs(r["decoders"]["bp maxlog"]["loss_db"]) <= 0.05, m


def test_chosen_fixed_decoder_no_worse_than_flooding_bp(e012):
    for m, r in e012["results"].items():
        d = r["decoders"][CHOSEN]
        assert d["loss_db"] is not None and d["loss_db"] <= 0.05, (m, d["loss_db"])
        # and as good as the same algorithm in floating point at its offset (0.5 nat)
        assert d["esno_db_at_bler1"] - r["decoders"]["oms float o0.5"]["esno_db_at_bler1"] <= 0.1, m


def test_seven_bit_app_floors(e012):
    floors = [min(p["q6/6/7 d0.5 o1"]["bler"] for p in r["points"]) for r in e012["results"].values()]
    assert max(floors) > 1e-3
