"""M8: the board loopback's software side. The RTL itself (rtl/ldpc_dec,
rtl/rx, rtl/chan_emu, rtl/loopback) is tested bit exact by cocotb (make
test-rtl); these tests hold the model, the generated tables and the
configuration words in step with it."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import chanemu, rtlcfg, twin                    # noqa: E402
from harness.fxmodem import FixedPoint                        # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords  # noqa: E402
from tools import gen_rtl                                     # noqa: E402

MCS = {m.name: m for m in MCS_TABLE}


def test_radius_rom_is_generated_from_the_model():
    assert (ROOT / "rtl" / "chan_emu" / "chan_emu_rtab.v").read_text() == gen_rtl.radius_rom(), \
        "rtl/chan_emu/chan_emu_rtab.v is stale: make rtl-tables"


def test_emulator_noise_is_unit_complex_gaussian():
    nr, ni = chanemu.noise(1234, 400_000)
    x = np.concatenate([nr, ni]).astype(np.float64) / 2 ** 20
    assert abs(x.mean()) < 0.005
    assert abs(x.var() - 1) < 0.01
    # Gaussian tails: P(|x| > 3) = 0.0027, P(|x| > 4) = 6.3e-5
    assert abs((np.abs(x) > 3).mean() / 0.0027 - 1) < 0.1
    assert 0.5 < (np.abs(x) > 4).mean() / 6.3e-5 < 1.6
    assert np.abs(x).max() < 6
    # re and im independent, frames with other seeds uncorrelated
    assert abs(np.corrcoef(nr, ni)[0, 1]) < 0.01
    assert abs(np.corrcoef(nr[:100_000], chanemu.noise(1235, 100_000)[0])[0, 1]) < 0.01


def test_emulator_snr_and_agc_level():
    fp = FixedPoint()
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal((50_000, 2)) * 300).astype(np.int64)
    p = float((x.astype(float) ** 2).sum(-1).mean()) / fp.tx_scale ** 2
    for esno in (0.0, 10.0, 20.0):
        no = 10 ** (-esno / 10)
        gx, gn, g = chanemu.gains(no, p, fp)
        y = chanemu.apply(x, 7, gx, gn).astype(np.float64)
        sig = chanemu.apply(x, 7, gx, 0).astype(np.float64)
        noise_p = ((y - sig) ** 2).sum(-1).mean()
        assert abs(10 * np.log10(noise_p / (g * g * no)) ) < 0.1, esno
        assert abs(10 * np.log10((y ** 2).sum(-1).mean() / fp.agc_power)) < 0.1, esno


def test_ldpc_config_of_every_code():
    """Every code of the MCS table fits the decoder (z a multiple of 16,
    its memories), for both pilot layouts."""
    import dataclasses
    p3 = dataclasses.replace(NUM_10MHZ, pilot_ofdm_symbols=(1, 6, 11))
    for num in (NUM_10MHZ, p3):
        for m in MCS_TABLE:
            p = plan_codewords(num, m)
            code = twin.ldpc_code(p.k, p.n)
            cfg = rtlcfg.ldpc_cfg(code)
            edges = [d for a, d in cfg if a >= rtlcfg.LDPC_EDGE_BASE]
            assert len(edges) == sum(len(r) for r in code.rows)
            assert sum(d >> (rtlcfg.LDPC_AW + 9) for d in edges) == code.dec_rows     # one 'last' per row


def test_e018_gate():
    import json
    path = ROOT / "results" / "e018_board_emulator.json"
    if not path.exists():
        pytest.skip("run experiments/e018_board_emulator.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    assert set(r["results"]) == set(MCS)
    for m, v in r["results"].items():
        assert v["board_minus_l2_db"] is not None, m
        assert abs(v["board_minus_l2_db"]) <= r["tolerance_db"], (m, v["board_minus_l2_db"])
        # both curves resolved at 1 %: a point on either side with enough errors
        for side in ("l2", "board"):
            fers = [p["fer"] for p in v[side]["points"]]
            assert any(f > 0.01 for f in fers) and any(f < 0.01 for f in fers), (m, side)
    assert r["gate_met"]


def test_e019_board_equals_the_model():
    """The board run (tools/board.py): every counter equal to e018's model
    at every point, hence the board's FER within the gate of L2."""
    import json
    path = ROOT / "results" / "e019_board_loopback.json"
    if not path.exists():
        pytest.skip("run make board with the Arty Z7-20 connected first")
    r = json.loads(path.read_text())
    assert len(r["points"]) >= 7 * 5
    assert {p["mcs"] for p in r["points"]} == set(MCS)
    assert all(p["match"] for p in r["points"])
    assert r["all_match"] and r["gate_met"]
