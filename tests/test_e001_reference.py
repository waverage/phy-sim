"""M1 gate: the committed AWGN baseline is plausible and a re-run stays on it.

Plausibility windows come from information theory and published LDPC
results: the Es/N0 at 1 % FER must sit 0.8 to 3 dB above the Shannon limit
for the MCS's spectral efficiency (capacity-achieving codes of this length
are about 1 to 1.5 dB away). The windows catch gross mistakes such as a
wrong noise scaling or a broken code-rate plan, not fine detail.
"""
import json
import math
import pathlib

import numpy as np
import pytest

RESULT = pathlib.Path(__file__).resolve().parents[1] / "results" / "e001_awgn_baseline.json"


def shannon_esno_db(bits_per_re: float) -> float:
    """Es/N0 needed for `bits_per_re` bit/s/Hz on a complex AWGN channel."""
    return 10 * math.log10(2 ** bits_per_re - 1)


@pytest.fixture(scope="module")
def result():
    if not RESULT.exists():
        pytest.skip("run experiments/e001_awgn_baseline.py first")
    return json.loads(RESULT.read_text())


def test_all_mcs_present(result):
    assert set(result["curves"]) == {"qpsk-1/2", "qpsk-3/4", "16qam-1/2", "16qam-3/4",
                                     "64qam-2/3", "64qam-3/4", "64qam-5/6"}


def test_fer_at_1pct_within_window_of_capacity(result):
    for name, c in result["curves"].items():
        s1 = c["esno_db_at_fer1"]
        assert s1 is not None, f"{name}: curve never crossed 1 % FER"
        eff = c["mcs"]["bits_per_symbol"] * c["mcs"]["code_rate"]
        gap = s1 - shannon_esno_db(eff)
        assert 0.8 <= gap <= 3.0, f"{name}: {gap:.2f} dB from capacity, outside 0.8..3.0"


def test_curves_are_monotonic_and_ordered(result):
    order = ["qpsk-1/2", "qpsk-3/4", "16qam-1/2", "16qam-3/4", "64qam-2/3", "64qam-3/4", "64qam-5/6"]
    prev = -99
    for name in order:
        pts = result["curves"][name]["points"]
        fers = [p["fer"] for p in pts]
        assert all(a >= b - 0.02 for a, b in zip(fers, fers[1:])), f"{name}: FER not decreasing with SNR"
        s1 = result["curves"][name]["esno_db_at_fer1"]
        assert s1 > prev, f"{name}: needs less SNR than the MCS below it"
        prev = s1


def test_rerun_drift(result):
    """Regression: a fresh short run of one MCS must land within 0.2 dB of
    the committed curve at 10 % FER. Kept short so `make check` stays quick."""
    import sys
    sys.path.insert(0, str(RESULT.parents[1]))
    from harness.chain import AwgnLink
    from harness.sweep import run_sweep, snr_at_fer
    from harness.waveform import MCS_TABLE, NUM_10MHZ
    ref = result["curves"]["qpsk-1/2"]["esno_db_at_fer10"]
    link = AwgnLink(NUM_10MHZ, MCS_TABLE[0], compile=False)   # compiling costs more than this run
    pts = run_sweep(link, np.arange(ref - 1.0, ref + 1.01, 0.25), batch=200,
                    min_frame_errors=60, max_frames=1500, seed=7, log=None)
    got = snr_at_fer(pts, 0.10)
    assert got is not None and abs(got - ref) <= 0.2, f"10 % FER moved from {ref:.2f} to {got} dB"
