"""M3 gates: the Zig modem (modem/, through harness/modem.py) against the
Sionna chain it mirrors.

Bit-exact where the arithmetic allows it: LDPC code construction and
encoding, QAM mapping. Within f32 rounding: the transmitted samples, the
demapper and the receiver's LLRs on the same received samples.
Statistically: the LDPC decoder (same algorithm, different summation order
and phi evaluation), and the FER curves in tests/test_e011_reference.py.
"""
import dataclasses
import functools
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import impairments as imp                      # noqa: E402
from harness import modem as zm                              # noqa: E402
from harness.chain import OfdmLink                          # noqa: E402
from harness.sweep import default_device                    # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords, with_continual_pilots  # noqa: E402

if not zm.LIB.exists():
    pytest.skip("build the modem first: make modem", allow_module_level=True)

from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder  # noqa: E402
from sionna.phy.mapping import Demapper, Mapper              # noqa: E402

MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
# every MCS of the table, plus BG2 codes of the other k_b classes
CODES = sorted({(plan_codewords(NUM_10MHZ, m).k, plan_codewords(NUM_10MHZ, m).n) for m in MCS_TABLE}
               | {(100, 300), (400, 1000), (600, 1500), (1000, 2000), (3000, 9000)})


def _bits(shape, seed):
    return np.random.default_rng(seed).integers(0, 2, shape).astype(np.uint8)


@pytest.mark.parametrize("k,n", CODES)
def test_ldpc_code_construction_matches_sionna(k, n):
    enc = LDPC5GEncoder(k, n, device="cpu")
    dec = LDPC5GDecoder(enc, device="cpu")
    info = zm.ldpc_info(k, n)
    assert info["bg"] == int(enc._bg[-1])
    assert (info["z"], info["k_filler"], info["n_ldpc"]) == (enc.z, enc.k_filler, enc.n_ldpc)
    # pruning at block granularity keeps every node Sionna keeps
    assert info["dec_cols"] * enc.z >= dec._n_pruned > (info["dec_cols"] - 1) * enc.z


@pytest.mark.parametrize("k,n", CODES)
def test_ldpc_encoder_bit_exact(k, n):
    u = _bits((16, k), seed=k)
    want = LDPC5GEncoder(k, n, device="cpu")(torch.as_tensor(u, dtype=torch.float32)).numpy().astype(np.uint8)
    assert np.array_equal(zm.ldpc_encode(u, n), want)


@pytest.mark.parametrize("bits", [2, 4, 6])
def test_qam_mapper_bit_exact(bits):
    labels = np.arange(2 ** bits)
    b = ((labels[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)   # MSB first
    want = Mapper("qam", bits, device="cpu")(torch.as_tensor(b.reshape(1, -1), dtype=torch.float32)).numpy()[0]
    assert np.array_equal(zm.qam_map(b, bits), want)


@pytest.mark.parametrize("bits", [2, 4, 6])
def test_qam_demapper_matches_sionna(bits):
    rng = np.random.default_rng(bits)
    y = (rng.normal(size=4000) + 1j * rng.normal(size=4000)).astype(np.complex64) * 0.8
    no = (10 ** rng.uniform(-3, 0.5, size=4000)).astype(np.float32)
    want = Demapper("app", "qam", bits, device="cpu")(torch.as_tensor(y), torch.as_tensor(no)).numpy().reshape(-1, bits)
    got = zm.qam_demap(y, no, bits)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=2e-3)


def test_ldpc_decoder_matches_sionna_statistically():
    """BPSK over AWGN at the waterfall of 16qam-1/2's code: the two
    decoders fail on nearly the same codewords; early stopping changes no
    decision that 20 full iterations would make."""
    p = plan_codewords(NUM_10MHZ, MCS["16qam-1/2"])
    enc = LDPC5GEncoder(p.k, p.n, device="cpu")
    dec = LDPC5GDecoder(enc, num_iter=20, hard_out=True, device="cpu")
    u = _bits((300, p.k), seed=1)
    c = zm.ldpc_encode(u, p.n).astype(np.float32)
    sigma = 0.89                                   # about 40 % codeword errors
    y = (1 - 2 * c) + sigma * np.random.default_rng(2).normal(size=c.shape)
    llr = (-2 * y / sigma ** 2).astype(np.float32)        # log p(1) / p(0)
    ref = dec(torch.as_tensor(llr)).numpy().astype(np.uint8)
    got, it_full = zm.ldpc_decode(llr, p.k, 20, early_stop=False)
    got_es, it_es = zm.ldpc_decode(llr, p.k, 20, early_stop=True)
    ref_bad, got_bad = (ref != u).any(-1), (got != u).any(-1)
    assert 60 < ref_bad.sum() < 240
    assert (ref_bad != got_bad).sum() <= 6                 # 2 %
    assert abs(int(ref_bad.sum()) - int(got_bad.sum())) <= 6
    assert np.array_equal((got_es != u).any(-1), got_bad)
    assert it_es <= it_full


NUMEROLOGIES = [NUM_10MHZ, P3, with_continual_pilots(NUM_10MHZ, 8)]


@pytest.mark.parametrize("num", NUMEROLOGIES, ids=lambda n: n.name)
@pytest.mark.parametrize("mcs", MCS_TABLE, ids=lambda m: m.name)
def test_tx_matches_sionna(num, mcs):
    link = OfdmLink(num, mcs, device="cpu", compile=False)
    z = zm.zig_modem_for(link, num_threads=4)
    b = torch.as_tensor(_bits((4, link.plan.num_codewords, link.plan.k), seed=3), dtype=torch.float32)
    want = link.transmit(b).reshape(4, -1).numpy()
    got = z.tx(b.numpy().astype(np.uint8))
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 2e-6                 # f32 FFT rounding; samples are ~0.9 rms


RX_CASES = {
    "awgn": (NUM_10MHZ, "16qam-3/4", 11.0, dict(channel="awgn")),
    "tdl-600Hz": (NUM_10MHZ, "64qam-3/4", 21.0, dict(channel="tdl", doppler_hz=600.0, carrier_hz=5.8e9)),
    "cfo200-linear": (NUM_10MHZ, "64qam-5/6", 18.5, dict(
        channel="awgn", phase_track="linear",
        channel_impairments=functools.partial(imp.cfo, offset_hz=200.0))),
    "p3-tdl-2000Hz": (P3, "16qam-1/2", 9.0, dict(channel="tdl", doppler_hz=2000.0, carrier_hz=5.8e9)),
}


@pytest.mark.parametrize("case", list(RX_CASES))
def test_rx_matches_sionna_on_the_same_samples(case):
    """Sionna's receiver and the Zig receiver on identical received
    samples: LLRs equal up to f32 rounding, decoded codewords the same
    nearly always (the decoders differ in rounding only)."""
    num, mcs_name, esno_db, kw = RX_CASES[case]
    torch.manual_seed(5)
    link = zm.ZigOfdmLink(num, MCS[mcs_name], compare=True, device=default_device(), compile=False,
                          early_stop=False, **kw)
    ref, p = link.ref, link.plan
    no = torch.tensor(10 ** (-esno_db / 10), device=ref.device)
    link.set_noise(float(no))
    b = ref.source([64, p.num_codewords, p.k])
    x = torch.as_tensor(link.zig.tx(b.to(torch.uint8).cpu().numpy()), device=ref.device)
    y, h_t = ref.propagate(x.reshape(64, 1, 1, -1), no)
    b_ref, llr_ref = ref.receive(y, h_t, no, return_llr=True)
    b_zig, llr_zig = link.zig.rx(y.reshape(64, -1).cpu().numpy(), float(no), ref.l_min, return_llr=True)
    llr_ref = llr_ref.cpu().numpy()
    err = np.abs(llr_zig - llr_ref)
    scale = np.abs(llr_ref).mean()
    assert np.quantile(err, 0.999) < 1e-3 * scale + 1e-4, (np.quantile(err, 0.999), scale)
    assert np.array_equal(np.sign(llr_zig[np.abs(llr_ref) > 1e-2]), np.sign(llr_ref[np.abs(llr_ref) > 1e-2]))
    ref_bad = (b_ref.cpu().numpy().astype(np.uint8) != b.cpu().numpy()).any(-1)
    zig_bad = (b_zig != b.cpu().numpy()).any(-1)
    assert (ref_bad != zig_bad).sum() <= 2, (ref_bad.sum(), zig_bad.sum())


def test_rx_rejects_missing_filter_and_short_input():
    link = OfdmLink(NUM_10MHZ, MCS["qpsk-1/2"], estimator="lmmse", device="cpu", compile=False)
    z = zm.zig_modem_for(link)
    y = np.zeros((1, z.num_samples), np.complex64)
    with pytest.raises(RuntimeError):
        z.rx(y, 0.1)
    w, ev = link.estimator.filters(0.1)
    z.set_filter(w.numpy(), ev.numpy())
    with pytest.raises(RuntimeError):
        z.rx(y[:, :-1], 0.1)
    with pytest.raises(ValueError):
        zm.zig_modem_for(link, phase_track="linear").set_filter(w.numpy(), ev.numpy())
