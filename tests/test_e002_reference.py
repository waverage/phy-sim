"""M2 gate: the committed e002 reference run is complete and plausible.

e002 is the fading and channel-estimation baseline that later chains (a
real synchroniser, a product estimator, the Zig modem) are compared
against, so its numbers are protected the way e001's are. The windows are
wide enough for Monte Carlo noise and for small model changes, narrow
enough to catch a broken noise scaling, interleaver, estimator or channel
normalisation. The other M2 experiments (e003 to e005) are decisions, not
baselines; their JSON only has to load and be complete.
"""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
MCS = ["qpsk-1/2", "qpsk-3/4", "16qam-1/2", "16qam-3/4", "64qam-2/3", "64qam-3/4", "64qam-5/6"]


def load(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"run experiments/{name}.py --profile reference first")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def e002():
    return load("e002_tdl_doppler")


@pytest.fixture(scope="module")
def e001():
    return load("e001_awgn_baseline")


def s1(result, cfg, mcs):
    v = result["configs"][cfg][mcs]["esno_db_at_fer1"]
    assert v is not None, f"{cfg} {mcs} never crossed 1 % FER"
    return v


def test_complete_and_reference_profile(e002):
    assert e002["profile"] == "reference"
    assert set(e002["configs"]) == {"awgn-perfect", "tdl0-perfect", "tdl600-perfect", "tdl600-lmmse", "tdl600-ls"}
    for cfg in e002["configs"]:
        assert set(e002["configs"][cfg]) == set(MCS)


def test_ofdm_awgn_reproduces_e001(e002, e001):
    """OFDM over AWGN with perfect CSI is the same code and mapping as the
    per-RE chain, so the two reference curves must agree."""
    for m in MCS:
        d = s1(e002, "awgn-perfect", m) - e001["curves"][m]["esno_db_at_fer1"]
        assert abs(d) <= 0.3, f"{m}: OFDM chain {d:+.2f} dB off the e001 baseline"


def test_fading_loss_window(e002):
    """TDL-D (Rician, K about 13 dB) with perfect CSI: one fade per frame
    costs 3 to 7 dB at 1 % FER; no Doppler and 600 Hz are within 1.5 dB."""
    for m in MCS:
        awgn = s1(e002, "awgn-perfect", m)
        assert 3.0 <= s1(e002, "tdl600-perfect", m) - awgn <= 7.0, m
        assert 3.0 <= s1(e002, "tdl0-perfect", m) - awgn <= 7.0, m
        assert abs(s1(e002, "tdl600-perfect", m) - s1(e002, "tdl0-perfect", m)) <= 1.5, m


def test_estimator_ordering(e002):
    """Genie LMMSE sits within 1 dB of perfect CSI; raw LS with linear
    interpolation is 1.5 to 4.5 dB behind LMMSE."""
    for m in MCS:
        perfect, lmmse, ls = (s1(e002, c, m) for c in ("tdl600-perfect", "tdl600-lmmse", "tdl600-ls"))
        assert -0.5 <= lmmse - perfect <= 1.0, f"{m}: lmmse - perfect = {lmmse - perfect:+.2f}"
        assert 1.5 <= ls - lmmse <= 4.5, f"{m}: ls - lmmse = {ls - lmmse:+.2f}"


def test_other_m2_experiments_complete():
    e003 = load("e003_pilot_density")
    assert set(e003["configs"]) == {f"p{n}-{e}" for n in (1, 2, 3, 4) for e in ("ls", "lmmse")}
    e004 = load("e004_cp_delay_spread")
    assert set(e004["configs"]) == {f"ds{d}-cp{c}" for d in (100, 300, 1000) for c in (8, 16, 32, 64)}
    e005 = load("e005_pa_backoff")
    assert set(e005["spectral"]) == {"linear", "ibo12", "ibo10", "ibo8", "ibo6", "ibo4", "ibo2"}
    assert set(e005["curves"]) == {"16qam-3/4", "64qam-5/6"}
    for r in (e003, e004, e005):
        assert r["profile"] == "reference"
