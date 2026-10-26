"""The committed e017 reference run: the link layer and video model on the
e016 traces, the M6 gate (freeze seconds per minute per SNR, the MCS
policy chosen)."""
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
FADING = ("tdl-d-600", "tdl-d-50", "tdl-c-300-600")


@pytest.fixture(scope="module")
def e017():
    path = RESULTS / "e017_video_link.json"
    if not path.exists():
        pytest.skip("run experiments/e017_video_link.py --profile reference first")
    r = json.loads(path.read_text())
    assert r["profile"] == "reference"
    return r


def test_freeze_reported_per_snr(e017):
    r = e017["results"]
    chosen = r["chosen_policy"]
    grid = r["policy_grid_db"]
    for ch in ("awgn",) + FADING:
        p = r["policies"][ch][chosen]
        assert len(p["freeze_s_per_min"]) == len(grid)
        # the picture is frozen with no signal and nearly clean from 12 dB up.
        # AWGN keeps up to 0.5 s/min: its waterfall is a cliff, so every probe
        # one MCS up loses a block before the outer loop backs off
        assert p["freeze_s_per_min"][0] > 50
        hi = [f for s, f in zip(grid, p["freeze_s_per_min"]) if s >= 12]
        assert max(hi) <= (0.6 if ch == "awgn" else 0.3), (ch, hi)


def test_custom_rules_beat_nestua_at_the_30_ms_budget(e017):
    r = e017["results"]["rules"]
    for ch in FADING:
        c, n30 = r[ch]["custom"]["esno_db_for"], r[ch]["nestua-30"]["esno_db_for"]
        for t, v in c.items():
            assert v is not None, (ch, t)
            assert n30[t] is None or v <= n30[t], (ch, t, v, n30[t])
        # and need no more Es/N0 than nestua's rules with their 70 ms window, for 5 Mbit/s and up
        n70 = r[ch]["nestua"]["esno_db_for"]
        for t in ("5", "8"):
            assert c[t] <= n70[t], (ch, t, c[t], n70[t])


def test_chosen_policy_on_the_flight(e017):
    r = e017["results"]
    fl, sc, chosen = r["flight"], r["flight_score"], r["chosen_policy"]
    base = sc["fixed qpsk-1/2"]
    c = sc[chosen]
    # as few freezes as the most robust MCS (the rest are outages no MCS survives) ...
    assert c["freeze_s_per_min"] <= 1.1 * base["freeze_s_per_min"] + 0.05, (c, base)
    # ... at twice its bitrate or more
    assert c["mean_bitrate_bps"] >= 2.0 * base["mean_bitrate_bps"], (c, base)
    # better than nestua-zig's ladder on both counts
    lad = sc["ladder (nestua rules)"]
    assert c["freeze_s_per_min"] < 0.5 * lad["freeze_s_per_min"]
    assert c["mean_bitrate_bps"] > lad["mean_bitrate_bps"]
    # no adaptive policy freezes less by more than 0.05 s/min
    adaptive = {k: v for k, v in sc.items() if not k.startswith("fixed")}
    assert c["freeze_s_per_min"] <= min(v["freeze_s_per_min"] for v in adaptive.values()) + 0.05
    for ch in FADING:
        assert fl[ch][chosen]["freeze_s_per_min"] <= 1.15 * fl[ch]["fixed qpsk-1/2"]["freeze_s_per_min"] + 0.1


def test_chosen_policy_at_fixed_snr(e017):
    r = e017["results"]
    for ch, by in r["policies"].items():
        p = by[r["chosen_policy"]]
        # mean over every Es/N0 where some fixed rung is freeze-free (one minute each)
        assert p["freeze_s_per_min_mean"] <= 0.15, (ch, p["freeze_s_per_min_mean"])
        assert p["bitrate_over_envelope"] >= 0.6, (ch, p["bitrate_over_envelope"])


def test_ablations_that_carry_the_decisions(e017):
    a = e017["results"]["ablations"]
    # 16+4 over 8+4: 8 Mbit/s needs 3 dB less on every fading channel
    for ch in FADING:
        assert a[ch]["fec 8+4"]["delta_db"]["8"] >= 2.0, ch
    # retransmission, and asking until 3 ms before the deadline, matter most in slow fading
    assert a["tdl-d-50"]["no retransmission"]["delta_db"]["1.5"] >= 2.0
    assert a["tdl-d-50"]["stop min(w/2, 20 ms)"]["delta_db"]["1.5"] >= 2.0
