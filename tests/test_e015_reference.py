"""M5 gate on the committed e015 reference run: with the oscillators' full
offset (CFO in +-116 kHz, 20 ppm SCO, PLL noise) and the frame at a random
time, the fixed-point synchroniser finds more than 99 % of the frames at
the lowest MCS's operating point on every channel, and costs little
against the genie (true start and offset) there.

At higher MCS a frequency loop that averages many frames (warm) is
needed: per-frame estimates (cold) carry the channel's own Doppler and
cost 64-QAM about 2 dB on TDL-D. Warm costs nothing without line of sight.
With Sionna's TDL-D line of sight, any frequency estimate removes the LoS
Doppler with the CFO, while the diffuse paths stay centred on the carrier:
0.6 dB on 64-QAM at 600 Hz with two pilot symbols, and a floor with three
pilot symbols at 2000 Hz (even cold qpsk-1/2 loses 0.6 dB there at 1 %
FER). Subtracting the predicted LoS Doppler (aided) recovers the genie
everywhere."""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
FIXED_BOUND_DB = 0.3
AIDED_BOUND_DB = 0.3
# cold sync at qpsk-1/2, and the warm loop at higher MCS, per channel: the
# line-of-sight channels pay for removing the LoS Doppler (see above)
COLD_QPSK_BOUND_DB = {"awgn": 0.3, "tdl-d-600": 0.3, "tdl-c-300-600": 0.3, "tdl-d-2000-p3": 0.75}
WARM_BOUND_DB = {"awgn": 0.3, "tdl-c-300-600": 0.3, "tdl-d-600": 0.75}


@pytest.fixture(scope="module")
def e015():
    path = RESULTS / "e015_sync_gate.json"
    if not path.exists():
        pytest.skip("run experiments/e015_sync_gate.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def losses(r, name):
    """The loss at 1 % and, where the grid reaches it, at 10 % FER."""
    out = [r["loss_db"][name]]
    if r["loss_db_at_fer10"][name] is not None:
        out.append(r["loss_db_at_fer10"][name])
    return out


def test_sync_success_above_99_percent_at_the_lowest_mcs(e015):
    """The M5 gate."""
    assert e015["cfo_hz"] >= 116e3
    for ch, by in e015["results"].items():
        assert by["qpsk-1/2"]["success_at_operating_point"] > 0.99, ch


def test_cold_sync_costs_little_at_the_lowest_mcs(e015):
    for ch, by in e015["results"].items():
        r = by["qpsk-1/2"]
        assert r["loss_db"]["sync_fixed"] is not None, ch
        for v in losses(r, "sync_fixed"):
            assert v <= COLD_QPSK_BOUND_DB[ch], (ch, v)


def test_fixed_point_chain_matches_float(e015):
    """Fixed synchroniser and receiver against both in floating point."""
    for ch, by in e015["results"].items():
        c = by["qpsk-1/2"]["curves"]
        for key in ("esno_db_at_fer1", "esno_db_at_fer10"):
            if c["sync_fixed"][key] is None and c["sync_float"][key] is None:
                continue
            d = c["sync_fixed"][key] - c["sync_float"][key]
            assert abs(d) <= FIXED_BOUND_DB, (ch, key, d)


def test_warm_frequency_loop(e015):
    for ch, bound in WARM_BOUND_DB.items():
        for m, r in e015["results"][ch].items():
            if m == "qpsk-1/2":
                continue
            assert r["loss_db"]["warm_tg0"] is not None, (ch, m)
            for v in losses(r, "warm_tg0"):
                assert v <= bound, (ch, m, v)


def test_cold_sync_is_not_enough_at_64qam(e015):
    """Why the loop is warm."""
    r = e015["results"]["tdl-d-600"]["64qam-3/4"]
    assert r["loss_db"]["sync_tg0"] > 1.0 > r["loss_db"]["warm_tg0"]


def test_aided_loop_recovers_the_genie(e015):
    for ch, by in e015["results"].items():
        for m, r in by.items():
            if m == "qpsk-1/2":
                continue
            assert r["loss_db"]["aided_tg0"] is not None, (ch, m)
            for v in losses(r, "aided_tg0"):
                assert v <= AIDED_BOUND_DB, (ch, m, v)


def test_line_of_sight_doppler_floors_three_pilots_at_2000_hz_without_aid(e015):
    r = e015["results"]["tdl-d-2000-p3"]["64qam-3/4"]
    assert r["loss_db"]["warm_tg0"] is None and r["loss_db"]["aided_tg0"] is not None


def test_tracker_off_when_the_loop_is_warm(e015):
    """Gain 0 is no worse than 0.5 behind a warm or aided loop."""
    for ch, by in e015["results"].items():
        for m, r in by.items():
            if m == "qpsk-1/2":
                continue
            c = r["curves"]
            for s in ("warm", "aided"):
                a, b = c[f"{s}_tg0"]["esno_db_at_fer1"], c[f"{s}_tg05"]["esno_db_at_fer1"]
                if a is not None and b is not None:
                    assert a <= b + 0.15, (ch, m, s, a, b)
                else:
                    assert b is None, (ch, m, s)          # gain 0 reaches 1 % wherever 0.5 does


def test_synchroniser_throughput(e015):
    for ch, by in e015["results"].items():
        assert by["qpsk-1/2"]["sync_fixed_frames_per_s"] > 10_000, ch
