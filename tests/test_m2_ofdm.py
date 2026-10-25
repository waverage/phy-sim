"""M2 gates: the impairment models do what their docstrings say, and the
OFDM chain is consistent with the M1 AWGN baseline.

The impairment checks are known-answer tests on synthetic signals. The link
checks catch scaling mistakes: OFDM over AWGN with perfect CSI must land on
the e001 curve, and the TDL channel must have unit mean energy so that
fading curves are comparable with AWGN ones.
"""
import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import impairments as imp                      # noqa: E402
from harness.chain import OfdmLink                          # noqa: E402
from harness.sweep import default_device, run_sweep, snr_at_fer  # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

FS = 10e6


def tone(f_hz, n=4096, frames=1):
    t = torch.arange(n, dtype=torch.float64)
    return torch.exp(1j * 2 * math.pi * f_hz / FS * t).repeat(frames, 1)


def gaussian(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g)) / math.sqrt(2)


def peak_hz(x):
    f = torch.fft.fftfreq(x.shape[-1], d=1 / FS)
    return float(f[torch.fft.fft(x[0]).abs().argmax()])


# --- impairments -----------------------------------------------------------

def test_cfo_shifts_tone():
    x = tone(1e6)
    assert torch.allclose(imp.cfo(x, FS, 0.0), x)
    assert abs(peak_hz(imp.cfo(x, FS, 312_500.0)) - 1.3125e6) < FS / 4096


def test_sco_matches_resampled_tone():
    x = tone(200e3)
    assert torch.equal(imp.sco(x, FS, 0.0), x)
    ppm = 1000.0                       # exaggerated so the drift is visible
    y = imp.sco(x, FS, ppm)
    t = torch.arange(4096, dtype=torch.float64) * (1 + ppm * 1e-6)
    want = torch.exp(1j * 2 * math.pi * 200e3 / FS * t)
    mid = slice(100, 3900)             # away from the zero-padded ends
    assert float((y[0, mid] - want[mid]).abs().max()) < 2e-3


def test_phase_noise_variance_growth():
    torch.manual_seed(0)
    x = torch.ones(2000, 4000, dtype=torch.complex128)
    y = imp.phase_noise(x, FS, dbc_hz=-80.0, offset_hz=100e3)
    want = 4 * math.pi ** 2 * 100e3 ** 2 * 10 ** (-8.0) / FS * 4000
    got = float(torch.angle(y[:, -1]).var())
    assert abs(got / want - 1) < 0.1, f"phase variance {got:.3e}, want {want:.3e}"


def test_pll_phase_noise_matches_mask_and_datasheet():
    """The generated phase has the mask's variance (over the offsets a frame
    resolves), and -100 dBc/Hz in a 100 kHz loop is the AD9361's 0.37 deg rms
    at 2.4 GHz."""
    assert abs(math.degrees(math.sqrt(imp.integrated_phase_noise_rad2(-100.0))) - 0.37) < 0.02
    torch.manual_seed(0)
    n = 4032
    y = imp.pll_phase_noise(torch.ones(500, n, dtype=torch.complex128), FS, inband_dbc_hz=-90.0)
    got = float(torch.angle(y).var())
    want = imp.integrated_phase_noise_rad2(-90.0, f_lo=FS / (8 * n), f_hi=FS / 2)
    assert abs(got / want - 1) < 0.1, f"phase variance {got:.3e}, want {want:.3e}"


def test_rapp_linear_far_from_saturation_and_bounded():
    x = gaussian((4, 8192))
    assert torch.allclose(imp.rapp_pa(x, FS, backoff_db=40.0), x, rtol=1e-3, atol=1e-4)
    y = imp.rapp_pa(x, FS, backoff_db=0.0, p=2.0)
    a_sat = math.sqrt(float(imp.mean_power(x)))
    assert float(y.abs().max()) <= a_sat * (1 + 1e-6)


def test_iq_imbalance_image_level():
    f0 = 400 * FS / 4096                # on a bin, so there is no leakage
    x = tone(f0)
    assert torch.allclose(imp.iq_imbalance(x, FS, 0.0, 0.0), x)
    y = torch.fft.fft(imp.iq_imbalance(x, FS, 0.5, 2.0)[0])
    img = float(y[-400].abs() ** 2 / y[400].abs() ** 2)
    g, phi = 10 ** (0.5 / 20), math.radians(2.0)
    mu = (1 + g * complex(math.cos(phi), -math.sin(phi))) / 2
    nu = (1 - g * complex(math.cos(phi), math.sin(phi))) / 2
    assert abs(10 * math.log10(img) - 10 * math.log10(abs(nu / mu) ** 2)) < 0.01
    assert 28 < -10 * math.log10(img) < 30


def test_adc_sqnr():
    x = gaussian((16, 65536))

    def sqnr(**kw):
        return 10 * math.log10(float(imp.mean_power(x) / imp.mean_power(imp.adc(x, FS, bits=12, **kw) - x)))

    # Full scale at 4 sigma per rail: quantisation alone is -65 dB, but the
    # clipped tail, 2 ((1 + a^2) Q(a) - a phi(a)) = 8e-6 at a = 4, dominates.
    assert 50 < sqnr(headroom_db=12.0) < 54
    # 18 dB headroom (8 sigma): clipping vanishes, quantisation is -59 dB
    assert 57 < sqnr(headroom_db=18.0) < 61
    # AGC 9 dB too hot leaves 3 dB headroom: heavy clipping
    assert sqnr(headroom_db=12.0, agc_error_db=9.0) < 30


def test_wifi_burst_level_band_and_extent():
    torch.manual_seed(0)
    x = gaussian((64, 4096))
    y = imp.wifi_burst(x, FS, sir_db=10.0, band_hz=(0.0, 5e6), burst_s=100e-6)
    w = y - x
    on = w.abs() > 0
    assert int(on.sum(-1).max()) <= 1000                               # 100 us at 10 MHz
    assert abs(10 * math.log10(float(imp.mean_power(x) / w[on].abs().square().mean())) - 10.0) < 0.5
    wf = torch.fft.fft(w).abs().square().sum(0)
    f = torch.fft.fftfreq(4096, d=1 / FS)
    assert float(wf[f < 0].sum() / wf.sum()) < 0.02


def test_oversampled_identity_roundtrip():
    x = gaussian((3, 4032))
    assert torch.allclose(imp.oversampled(lambda v, fs: v, 4)(x, FS), x, atol=1e-5)


# --- link ------------------------------------------------------------------
# compile=False throughout: these runs are seconds long, compiling costs 20 s per block

def test_tdl_channel_unit_energy():
    torch.manual_seed(0)
    link = OfdmLink(NUM_10MHZ, MCS_TABLE[0], channel="tdl", doppler_hz=600.0, device=default_device(), compile=False)
    e = float(torch.cat([link.time_channel(100) for _ in range(4)]).abs().square().sum(-1).mean())
    assert abs(e - 1) < 0.03, f"mean channel energy {e:.3f}"


def test_ls_estimator_error_free_at_high_snr():
    link = OfdmLink(NUM_10MHZ, MCS_TABLE[3], estimator="ls", device=default_device(), compile=False)
    r = link.run(50, 30.0)
    assert r["bit_errors"] == 0


def test_ls_estimator_follows_fading_channel():
    """16qam-1/2 with LS needs about 13 dB for 1 % FER on TDL-D at 600 Hz
    (e002); at 20 dB it must be essentially error free."""
    torch.manual_seed(0)
    link = OfdmLink(NUM_10MHZ, MCS_TABLE[2], channel="tdl", doppler_hz=600.0, estimator="ls",
                    device=default_device(), compile=False)
    r = link.run(100, 20.0)
    assert r["frame_errors"] <= 2, r


def test_genie_lmmse_tracks_channel():
    """The genie LMMSE estimate must be close to the true per-symbol channel
    and its predicted error variance must be in the right range (the TDL
    sum-of-sinusoids process does not follow the model covariance exactly)."""
    from sionna.phy.channel import time_to_ofdm_channel
    torch.manual_seed(0)
    link = OfdmLink(NUM_10MHZ, MCS_TABLE[0], channel="tdl", doppler_hz=600.0, estimator="lmmse",
                    device=default_device(), compile=False)
    batch, no = 100, torch.tensor(0.1, device=link.device)         # 10 dB
    x = link.mapper(torch.randint(0, 2, (batch, NUM_10MHZ.num_data_res * 2), device=link.device).float())
    x_t = link.modulator(link.rg_mapper(x.reshape(batch, 1, 1, -1)))
    h_t = link.time_channel(batch)
    y_f = link.demodulator(link.awgn(link.apply_channel(x_t, h_t), no))
    h_true = link.remove_nulled(time_to_ofdm_channel(h_t, link.rg, link.l_min))
    h_hat, err_var = link.estimator(y_f, no)
    mse = float((h_hat - h_true).abs().square().mean())
    assert mse < 10 ** (-20 / 10), f"estimate error {10 * math.log10(mse):.1f} dB"
    assert abs(10 * math.log10(mse / float(err_var.mean()))) < 5


def test_continual_pilot_grid():
    from harness.waveform import with_continual_pilots
    num = with_continual_pilots(NUM_10MHZ, 8)
    link = OfdmLink(num, MCS_TABLE[0], estimator="lmmse", phase_track="symbol", device=default_device(),
                    compile=False)
    mask = link.rg.pilot_pattern.mask[0, 0].bool().cpu()
    assert mask[:, list(num.continual_pilot_subcarriers)].all()
    assert mask[list(num.pilot_ofdm_symbols)].all()
    assert int((~mask).sum()) == num.num_data_res == link.rg.num_data_symbols
    with pytest.raises(ValueError):
        OfdmLink(NUM_10MHZ, MCS_TABLE[0], estimator="lmmse", phase_track="symbol", compile=False)


@pytest.mark.parametrize("pilots,mode", [(0, "linear"), (8, "linear"), (8, "symbol")])
def test_phase_tracker_removes_residual_cfo(pilots, mode):
    """300 Hz turns the phase by 44 deg over a frame: without tracking
    64qam-5/6 is lost entirely, every tracker (the linear one even on the
    block pilots alone) must recover it."""
    import functools
    from harness.waveform import with_continual_pilots
    torch.manual_seed(0)
    off = functools.partial(imp.cfo, offset_hz=300.0)
    kw = dict(estimator="lmmse", channel_impairments=off, device=default_device(), compile=False)
    num = with_continual_pilots(NUM_10MHZ, pilots) if pilots else NUM_10MHZ
    plain = OfdmLink(num, MCS_TABLE[6], **kw).run(50, 22.0)
    tracked = OfdmLink(num, MCS_TABLE[6], phase_track=mode, **kw).run(50, 22.0)
    assert plain["frame_errors"] > 40, plain
    assert tracked["frame_errors"] == 0, tracked


def test_ofdm_awgn_matches_e001():
    """OFDM over AWGN with perfect CSI must reproduce the e001 curve at 10 %
    FER: proves the time-domain noise scaling and the interleaver."""
    ref = json.loads((ROOT / "results" / "e001_awgn_baseline.json").read_text())
    s10 = ref["curves"]["qpsk-1/2"]["esno_db_at_fer10"]
    link = OfdmLink(NUM_10MHZ, MCS_TABLE[0], estimator="perfect", device=default_device(), compile=False)
    pts = run_sweep(link, np.arange(s10 - 1.0, s10 + 1.01, 0.25), batch=200,
                    min_frame_errors=60, max_frames=1500, seed=7, log=None)
    got = snr_at_fer(pts, 0.10)
    assert got is not None and abs(got - s10) <= 0.2, f"10 % FER at {got} dB, e001 has {s10:.2f}"
