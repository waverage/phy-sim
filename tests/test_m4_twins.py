"""M4 gates: every fixed-point block of the Zig modem is bit exact against
its NumPy twin (harness/twin.py), from the generated tables to the decoded
bits of whole frames through the channel.

Blocks: pilot and interleaver tables, the 256-point FFT and IFFT (with
saturation), CORDIC vectoring and rotation, the 5G LDPC code structure,
the layered offset min-sum decoder, and the frame TX and RX with the
estimator, phase tracker and demapper compared through their probes.
The floating-point twin of the same algorithm (FxModem(fixed=False)) is
checked for consistency, not exactness.
"""
import dataclasses
import functools
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import fxmodem as fm                            # noqa: E402
from harness import impairments as imp                      # noqa: E402
from harness import modem as zm                              # noqa: E402
from harness import tables                                   # noqa: E402
from harness import twin                                     # noqa: E402
from harness.design import EstimatorDesign, design_filters   # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords  # noqa: E402

if not zm.LIB.exists():
    pytest.skip("build the modem first: make modem", allow_module_level=True)

MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
CODES = sorted({(plan_codewords(NUM_10MHZ, m).k, plan_codewords(NUM_10MHZ, m).n) for m in MCS_TABLE}
               | {(plan_codewords(P3, m).k, plan_codewords(P3, m).n) for m in MCS_TABLE}
               | {(100, 300), (400, 1000), (600, 1500), (1000, 2000)})


def test_tables_match_zig():
    for np_, bits in [(398, 4776), (597, 14328), (5, 7)]:
        signs, perm = fm.zig_tables(np_, bits)
        assert np.array_equal(signs, tables.pilot_signs(np_))
        assert np.array_equal(perm, tables.interleaver(bits))
        assert np.array_equal(np.sort(perm), np.arange(bits))


def test_gold_sequence_reference():
    """c_init = 1 against an independent shift-register reading of 38.211."""
    c = tables.gold(1, 16)
    x1 = [1] + [0] * 30
    x2 = [1] + [0] * 30
    for n in range(1600 + 16):
        x1.append(x1[n + 3] ^ x1[n])
        x2.append(x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n])
    assert c.tolist() == [x1[n + 1600] ^ x2[n + 1600] for n in range(16)]


@pytest.mark.parametrize("inverse", [False, True])
def test_fft_bit_exact(inverse):
    rng = np.random.default_rng(1)
    x = rng.integers(-2048, 2048, (6, 256, 2))
    x[0] = 0
    x[1, :, :] = 2047                          # DC at full scale: saturates
    x[2] = rng.integers(-32768, 32768, (256, 2))
    assert np.array_equal(fm.fft(x, inverse), twin.fft(x, inverse))
    assert np.array_equal(twin.twiddles(256)[:, 0][[0, 64]], [65536, 0])


def test_fft_is_unitary_and_inverts():
    rng = np.random.default_rng(2)
    x = rng.integers(-1500, 1500, (256, 2))
    X = twin.fft(x)
    assert abs((X.astype(float) ** 2).sum() / (x.astype(float) ** 2).sum() - 1) < 1e-3
    assert np.abs(twin.fft(X, inverse=True) - x).max() <= 4


def test_cordic_bit_exact():
    rng = np.random.default_rng(3)
    xy = rng.integers(-2 ** 30, 2 ** 30, (500, 2))
    xy[:4] = [[0, 0], [5, 0], [-5, 0], [0, -7]]
    want = np.array([twin.cordic_vector(int(a), int(b)) for a, b in xy])
    assert np.array_equal(fm.cordic_vector(xy), want)
    ang = np.concatenate([rng.integers(-200000, 200000, 500), [0, 16384, -16384, 32768, -32768, 65536]])
    assert np.array_equal(fm.cossin(ang), np.array([twin.cossin(int(a)) for a in ang]))
    # accuracy: angle within 4 units of 2^16 per turn (0.02 deg), cos/sin within 4e-4
    for (a, b), (mag, phase) in zip(xy[4:50], want[4:50]):
        assert abs((phase / 65536 * 2 * math.pi - math.atan2(b, a) + math.pi) % (2 * math.pi) - math.pi) < 4e-4
    cs = np.array([twin.cossin(int(a)) for a in ang]) / 32768
    assert np.abs(cs[:, 0] - np.cos(ang / 65536 * 2 * np.pi)).max() < 4e-4


@pytest.mark.parametrize("k,n", CODES)
def test_ldpc_code_structure(k, n):
    c = twin.ldpc_code(k, n)
    info = zm.ldpc_info(k, n)
    assert (c.bg, c.z, c.k_filler, c.n_ldpc, c.dec_rows, c.dec_cols) == tuple(
        info[x] for x in ("bg", "z", "k_filler", "n_ldpc", "dec_rows", "dec_cols"))


def _noisy_llr(k, n, count, esno_db, seed, step=0.25, bits=6):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, (count, k)).astype(np.uint8)
    x = 1.0 - 2.0 * zm.ldpc_encode(u, n)                      # BPSK, 0 -> +1
    no = 10 ** (-esno_db / 10)
    y = x + rng.standard_normal(x.shape) * math.sqrt(no / 2)
    logit = -4 * y / (2 * no) / step                          # log p1/p0 in LSBs
    return u, np.clip(np.floor(logit + 0.5), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)


@pytest.mark.parametrize("k,n", [(2388, 4776), (4776, 9552), (5970, 7164), (600, 1500)])
def test_ldpc_decoder_bit_exact(k, n):
    """Near the waterfall, so codewords take many iterations, some fail and
    the saturation is exercised."""
    esno = {(2388, 4776): -1.5, (4776, 9552): -1.5, (5970, 7164): 2.5, (600, 1500): -2.75}[(k, n)]
    u, llr = _noisy_llr(k, n, 12, esno, seed=k)
    bits, iters = fm.ldpc_decode(llr, k, "oms-fixed", offset=2)
    code = twin.ldpc_code(k, n)
    fails = 0
    for i in range(len(u)):
        b, it = twin.ldpc_decode(code, llr[i])
        assert np.array_equal(b, bits[i]) and it == iters[i], i
        fails += int((b != u[i]).any())
    assert 0 < fails < len(u), "pick an SNR in the waterfall"


def test_ldpc_float_decoder_matches_fixed_in_the_limit():
    """Without saturation pressure the float decoder decides the same."""
    u, llr = _noisy_llr(4776, 9552, 50, 1.0, seed=5)
    b_fix, _ = fm.ldpc_decode(llr, 4776, "oms-fixed", offset=2)
    b_flt, _ = fm.ldpc_decode(llr, 4776, "oms-float", offset=2)
    assert np.array_equal(b_fix, u) and np.array_equal(b_flt, u)


# ---- frames ---------------------------------------------------------------------------------

TDL = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0, carrier_hz=5.8e9)
FRAME_CASES = {
    "awgn": (NUM_10MHZ, dict(channel="awgn"), 0.0),
    "tdl-600": (NUM_10MHZ, TDL, 0.0),
    "tdl-cfo200-track": (NUM_10MHZ, dict(TDL, phase_track="linear",
                                         channel_impairments=functools.partial(imp.cfo, offset_hz=200.0)), 0.0),
    "p3-tdl-2000": (P3, dict(TDL, doppler_hz=2000.0), 0.0),
}


def _link(case, mcs_name):
    num, kw, _ = FRAME_CASES[case]
    return fm.FxOfdmLink(num, MCS[mcs_name], device="cpu", genie=False, num_threads=4, **kw)


def _twin_for(link):
    fp = link.fp
    return twin.TwinModem(link.num, link.mcs.bits_per_symbol, k=link.plan.k, n=link.plan.n,
                          num_codewords=link.plan.num_codewords,
                          pilot_mask=link.ref.rg.pilot_pattern.mask[0, 0].cpu().numpy(), tx_scale=fp.tx_scale,
                          dac_bits=fp.dac_bits, llr_bits=fp.llr_bits, app_bits=fp.app_bits, msg_bits=fp.msg_bits,
                          offset=fp.offset_lsb, num_iter=fp.num_iter, phase_track=link.ref.phase_track == "linear",
                          track_gain_q15=fm.track_gain_q15(link.design.track_gain))


@pytest.mark.parametrize("mcs_name", [m.name for m in MCS_TABLE])
def test_tx_frame_bit_exact(mcs_name):
    link = _link("awgn", mcs_name)
    tw = _twin_for(link)
    p = link.plan
    b = np.random.default_rng(7).integers(0, 2, (2, p.num_codewords, p.k)).astype(np.uint8)
    x = link.fixed.tx(b)
    for i in range(2):
        coded = zm.ldpc_encode(b[i], p.n)
        assert np.array_equal(x[i], tw.tx(coded))
    # the float transmitter is the same signal without rounding: the IFFT's
    # rounding (about 1 LSB rms against 320 LSB per rail) leaves a MER of
    # about 51 dB, where 64-QAM needs about 30
    xf = link.float.tx(b)
    err = (x[..., 0] + 1j * x[..., 1]) - xf
    mer = 10 * np.log10((np.abs(xf) ** 2).mean() / (np.abs(err) ** 2).mean())
    assert mer > 48, mer


def _received(link, esno_db, batch, seed):
    torch.manual_seed(seed)
    p = link.plan
    no = 10 ** (-esno_db / 10)
    link.set_noise(no)
    b = link.ref.source([batch, p.num_codewords, p.k])
    bb = b.to(torch.uint8).cpu().numpy()
    y, _ = link.ref.propagate(link.transmit(bb), torch.tensor(no))
    ys, yq, g = fm.agc_adc(y.reshape(batch, -1).cpu().numpy(), link.fp)
    return bb, ys, yq, link.llr_scale(no, g)


@pytest.mark.parametrize("case,mcs_name,esno", [
    ("awgn", "qpsk-1/2", 1.5), ("awgn", "16qam-3/4", 10.5), ("awgn", "64qam-5/6", 17.5),
    ("tdl-600", "16qam-1/2", 11.0), ("tdl-600", "64qam-3/4", 21.0),
    ("tdl-cfo200-track", "64qam-3/4", 22.0), ("p3-tdl-2000", "16qam-1/2", 12.0),
])
def test_rx_frame_bit_exact(case, mcs_name, esno):
    link = _link(case, mcs_name)
    tw = _twin_for(link)
    bb, ys, yq, scale = _received(link, esno, 2, seed=11)
    tw.set_filter(link.fixed.filt)
    bits, llr, pr = link.fixed.rx(yq, scale, link.ref.l_min, return_llr=True, probe=True)
    iters_total, _ = link.fixed.decoder_stats()
    iters = 0
    for i in range(2):
        t = tw.rx(yq[i], float(scale[i]), link.ref.l_min)
        assert np.array_equal(pr["y"][i], t["y"]), "FFT, timing phase or tracker"
        assert np.array_equal(pr["phi"][i], t["phi"]), "tracker phase"
        assert np.array_equal(pr["h"][i], t["h"]), "channel estimate"
        assert np.array_equal(llr[i], t["llr"]), "demapper"
        assert np.array_equal(bits[i], t["bits"]), "decoder"
        iters += t["iters"]
    assert iters == iters_total


def test_tracker_measures_cfo():
    """200 Hz over one 28.8 us symbol is 0.0362 rad, 377 binary units, of
    which the tracker removes track_gain. (Under TDL-D the line-of-sight
    path's Doppler adds to it.)"""
    FRAME_CASES["awgn-cfo200-track"] = (NUM_10MHZ, dict(
        channel="awgn", phase_track="linear", channel_impairments=functools.partial(imp.cfo, offset_hz=200.0)), 0)
    link = _link("awgn-cfo200-track", "16qam-1/2")
    bb, ys, yq, scale = _received(link, 15.0, 8, seed=2)
    _, pq = link.fixed.rx(yq, scale, link.ref.l_min, probe=True)
    b, pf = link.float.rx(ys, scale, link.ref.l_min, probe=True)
    slope = np.diff(pq["phi"], axis=-1).mean(-1) / link.design.track_gain
    # noise: 0.009 rad per pilot symbol at 15 dB over 199 subcarriers, so
    # about 15 units rms on the slope over 9 symbols
    assert abs(slope.mean() - 377) < 12 and np.abs(slope - 377).max() < 60, slope
    assert np.abs(pq["phi"] - pf["phi"] * 65536 / (2 * np.pi)).max() < 6
    assert np.array_equal(b, bb)


def test_float_receiver_matches_fixed_llrs():
    """Same algorithm, floating point: the LLRs agree to the quantisation
    (one LSB) wherever the fixed ones are not saturated."""
    link = _link("tdl-600", "16qam-1/2")
    bb, ys, yq, scale = _received(link, 10.0, 16, seed=3)
    _, lq = link.fixed.rx(yq, scale, link.ref.l_min, return_llr=True)
    _, lf = link.float.rx(ys, scale, link.ref.l_min, return_llr=True)
    # at 0.25 nat per LSB many LLRs sit at the rails, more in the strong frames
    inside = np.abs(lq) < 2 ** (link.fp.llr_bits - 1) - 1
    d = np.abs(lq - lf)[inside]
    assert inside.mean() > 0.1, inside.mean()
    assert np.mean(d <= 1.0) > 0.97 and np.median(d) < 0.6, (np.mean(d <= 1.0), np.median(d))


def test_design_filters_shapes_and_gain():
    d = EstimatorDesign()
    f = design_filters(NUM_10MHZ, d, 10.0)
    F, S = NUM_10MHZ.num_effective_subcarriers, NUM_10MHZ.num_ofdm_symbols
    assert f["bt"].shape == (F, d.taps) and f["at"].shape == (S, 2) and f["r"].shape == (S, F)
    assert np.all(f["start"] + d.taps <= F)
    # unbiased: every filter passes a constant channel with gain 1 (taps sum
    # to 1 before the folded 1/sqrt(2), time weights sum to 1)
    assert np.abs(f["bt"].sum(-1) * math.sqrt(2) - 1).max() < 1e-5
    assert np.abs(f["at"].sum(-1) - 1).max() < 1e-5
    biased = design_filters(NUM_10MHZ, dataclasses.replace(d, unbiased=False, doppler_hz=1000.0), 21.0)
    assert biased["at"][6].sum() > 1.1          # the plain Wiener filter overshoots mid-frame
    assert np.all((f["r"] > 0.5) & (f["r"] <= 1))
    assert np.abs(f["bt"]).max() * 32768 < 32767 and np.abs(f["at"]).max() * 16384 < 32767
