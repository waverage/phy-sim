"""M6: the continuous fading channel (harness/fading.py), the per-frame
decoding thresholds (harness/phytrace.py) and the link layer and video
model (harness/linklayer.py)."""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch
from scipy.special import j0

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import linklayer as L                            # noqa: E402
from harness import modem as zm                               # noqa: E402
from harness.fading import ContinuousTdl                     # noqa: E402
from harness.waveform import MCS_TABLE                       # noqa: E402

SLOT = 432e-6


# ---- the channel ---------------------------------------------------------------------------------

def test_continuous_tdl_matches_sionna_statistics():
    ct = ContinuousTdl(3000, model="D", delay_spread_s=100e-9, doppler_hz=600.0, seed=3)
    t = torch.tensor([0.0, 0.2e-3, 0.5e-3, 120.0], dtype=torch.float64)[None].repeat(3000, 1)
    a = ct.gains(t).numpy()
    p = (np.abs(a) ** 2).mean(0).sum(0)
    assert np.all(np.abs(p - 1) < 0.06), p                       # unit mean power, at any time
    x = a[:, 1]                                                  # a Rayleigh path: Jakes correlation
    for k, dt in ((1, 0.2e-3), (2, 0.5e-3)):
        rho = ((x[:, k] * np.conj(x[:, 0])).mean() / (np.abs(x[:, 0]) ** 2).mean()).real
        assert abs(rho - j0(2 * np.pi * 600 * dt)) < 0.05, (dt, rho)


def test_continuous_tdl_is_one_process_across_calls():
    """Taps for the second half of a buffer equal the taps of a call that starts there."""
    ct = ContinuousTdl(4, model="C", delay_spread_s=300e-9, doppler_hz=600.0, seed=1)
    h = ct.time_channel(np.zeros(4), 2000)
    h2 = ct.time_channel(np.full(4, 1000.0), 1000)
    # the gains are interpolated on a decimated grid that starts at each call's
    # first sample, so the two agree to the interpolation error
    err = np.abs(h[:, 1000:1000 + h2.shape[1] - 64].numpy() - h2[:, :h2.shape[1] - 64].numpy()).max()
    assert err < 2e-3, err


def test_time_channel_agrees_with_frequency_response():
    ct = ContinuousTdl(3, model="D", delay_spread_s=100e-9, doppler_hz=0.0, seed=2)
    h = ct.time_channel(np.zeros(3), 64)[:, 30].numpy()          # static: every row the same
    f = np.fft.fftfreq(256, 1 / 10e6)
    taps = np.zeros((3, 256), complex)
    taps[:, : ct.l_tot] = h
    lags = np.arange(ct.l_min, ct.l_max + 1)
    H_t = (h[:, None, :] * np.exp(-2j * np.pi * f[None, :, None] * lags[None, None, :] / 10e6)).sum(-1)
    H_f = ct.frequency_response(np.zeros((3, 1)), f)[:, 0]
    in_band = np.abs(f) < 4e6
    assert np.abs(H_t[:, in_band] - H_f[:, in_band]).max() < 0.05


# ---- the thresholds ------------------------------------------------------------------------------

@pytest.mark.skipif(not zm.LIB.exists(), reason="build the modem first: make modem")
def test_thresholds_on_awgn_bracket_the_decoder():
    from harness.phytrace import AWGN_1PCT_DB, TraceConfig, TraceLink
    mcs = (MCS_TABLE[0], MCS_TABLE[2])
    cfg = TraceConfig(model="awgn", num_links=4, hi_db=25.0, seed=7)
    tl = TraceLink(mcs=mcs, cfg=cfg, device="cpu", num_threads=2)
    chk = {m.name: [AWGN_1PCT_DB[m.name] - 1.0, AWGN_1PCT_DB[m.name] + 1.0] for m in mcs}
    out = tl.block(0, 24, check_db=chk)
    for m in mcs:
        thr = out["thr"][m.name]
        assert np.all(np.isfinite(thr))
        # a steep AWGN waterfall: the thresholds sit within a dB of the 1 % point
        assert abs(np.median(thr) - (AWGN_1PCT_DB[m.name] - 0.4)) < 1.0, np.median(thr)
        for i, s in enumerate(chk[m.name]):
            same = out["check"][m.name]["same"][i]
            assert ((thr <= s) == same).mean() >= 0.95


# ---- the link layer ------------------------------------------------------------------------------

def test_split_blocks_follows_nestua():
    # 25 packets, 8+3: two full blocks and a merged tail of 9 (8 < 9 < 13)
    assert L.split_blocks(30000, 1200, 8, 3, 2200) == [(8, 3), (8, 3), (9, 3)]
    # a short tail gets proportional parity: ceil(3 * 3/8 + 0.001) = 2
    assert L.split_blocks(11 * 1000 + 1, 1000, 8, 3, 0) == [(12, 4)]
    assert L.split_blocks(3 * 1000, 1000, 8, 3, 0) == [(3, 2)]
    assert L.split_blocks(500, 1000, 8, 3, 0) == [(1, 0)]          # one packet, no parity
    assert L.split_blocks(2000, 1000, 8, 3, 2200) == [(2, 0)]      # a small frame is retransmission only


def run(thr_fn, cfg=None, policy=None, snr=20.0, seconds=10.0, n=40000):
    thr = np.stack([thr_fn(i, n) for i in range(len(L.MCS_NAMES))])
    return L.LinkSim(thr, np.zeros(n, np.float32), SLOT, cfg or L.LinkConfig(),
                     policy or L.FixedPolicy("qpsk-1/2"), snr).run(seconds)


def test_clean_channel_never_freezes():
    r = run(lambda i, n: np.full(n, -50.0))
    assert r["freeze_s_per_min"] == 0 and r["frames_lost_per_min"] == 0
    assert r["latency_p99_ms"] < L.LinkConfig().latency_ms
    assert abs(r["frames_shown"] / 10.0 - 60) < 1


def test_dead_channel_freezes_all_the_time():
    r = run(lambda i, n: np.full(n, 99.0))
    assert r["frames_shown"] == 0 and r["freeze_s_per_min"] > 59


def iid_losses(p, seed=0):
    rng = np.random.default_rng(seed)
    return lambda i, n: np.where(rng.random(n) < p, 99.0, -50.0)


def test_parity_and_retransmission_each_repair_scattered_losses():
    cfg = L.LinkConfig(latency_ms=50.0, small_frame_bytes=0, stop_scaled=False, stop_ms=3.0, rerequest_ms=5.0,
                       bitrate_fraction=0.5, resync_dup=1)
    fec = run(iid_losses(0.03), replace_fec(cfg, 4, 4, retransmit=False))
    assert fec["frames_lost_per_min"] == 0
    retx = run(iid_losses(0.03), replace_fec(cfg, 4, 0, retransmit=True))
    nothing = run(iid_losses(0.03), replace_fec(cfg, 4, 0, retransmit=False))
    assert retx["retx_share"] > 0.01
    assert retx["frames_lost_per_min"] < 0.1 * nothing["frames_lost_per_min"]


def replace_fec(cfg, d, e, **kw):
    from dataclasses import replace
    return replace(L.with_fec(cfg, d, e), **kw)


def test_an_outage_recovers_through_an_ltr_reanchor():
    """60 ms of nothing: the picture freezes, the ground asks for a resync,
    the air sends a re-anchor on the last acknowledged anchor."""
    def outage(i, n):
        t = np.full(n, -50.0)
        t[int(1.0 / SLOT):int(1.06 / SLOT)] = 99.0
        return t
    from dataclasses import replace
    cfg = replace(L.LinkConfig(), resync_dup=1, ltr_cooldown_ms=20.0, frame_cap=0.6, bitrate_fraction=0.6)
    r = run(outage, cfg, seconds=4.0)
    assert r["frames_R"] >= 1 and r["frames_I"] <= 2               # the first IDR and the GOP's at 2 s
    assert 0.05 < r["longest_freeze_s"] < 0.2, r["longest_freeze_s"]


def test_snr_policy_climbs_with_snr_and_ladder_steps_down():
    from dataclasses import replace
    cfg = replace(L.LinkConfig(), bitrate_fraction=0.6)
    awgn = lambda i, n: np.full(n, L.AWGN_1PCT_DB[L.MCS_NAMES[i]])      # noqa: E731
    lo = run(awgn, cfg, L.SnrPolicy(target=0.01), snr=5.0)
    hi = run(awgn, cfg, L.SnrPolicy(target=0.01), snr=25.0)
    assert hi["mean_bitrate_bps"] > 3 * lo["mean_bitrate_bps"]
    assert max(hi["mcs_share"], key=hi["mcs_share"].get) == "64qam-5/6"
    lad = run(awgn, cfg, L.LadderPolicy(start_mcs="64qam-3/4"), snr=5.0)
    assert lad["mcs_share"]["qpsk-1/2"] + lad["mcs_share"]["qpsk-3/4"] > 0.3


def test_flight_profile():
    snr, d = L.flight_snr(SLOT, snr_1km_db=20.0, dips_per_min=0.0, shadow_db=0.0)
    assert abs(d.max() - 4000) < 1 and abs(d.min() - 200) < 1
    assert abs(snr.min() - (20 - 20 * math.log10(4))) < 0.01
    i = int(np.argmax(d))
    assert abs(i * SLOT - (3800 / 30)) < 0.01
