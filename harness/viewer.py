"""One frame through the whole M5 chain, with every stage kept for the
viewer (tools/viewer.py).

capture() runs `frames` frames at one setting and returns frame 0's
pictures (spectrum, waterfall, waveform, synchroniser metric and channel
impulse response, constellation before and after equalisation, channel
estimate against the true channel, tracker phase, error vector per resource
element, LLRs) plus the error counts of all of them. The chain is
SyncLink's (harness/synclink.py), with the carrier offset set exactly
instead of drawn at random, and an optional Wi-Fi burst before the noise.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import sionna
import torch

from . import impairments as imp
from .design import EstimatorDesign, subcarrier_offsets
from .modem import ldpc_encode
from .sync import STF_PERIOD, SyncResult, _running_sum, buffer_time_channel, frame_cfo, inc_of
from .synclink import ApplyTime, SyncLink
from .waveform import MCS_TABLE, NUM_10MHZ, phy_rate_bps

MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
CHANNELS = {
    "awgn": dict(channel="awgn"),
    "tdl-d": dict(channel="tdl", tdl_model="D"),     # line of sight plus echoes
    "tdl-c": dict(channel="tdl", tdl_model="C"),     # echoes only (no line of sight)
}

DEFAULTS = dict(mcs="qpsk-1/2", esno_db=10.0, channel="awgn", delay_spread_ns=100.0, doppler_hz=0.0,
                cfo_hz=20e3, phase_noise=True, sco=True, wifi=False, sir_db=10.0, pilots=2,
                receiver="float", sync="sync", seed=1, frames=1)

_links: dict = {}


def _link(p: dict) -> SyncLink:
    num = P3 if p["pilots"] == 3 else NUM_10MHZ
    ch = CHANNELS[p["channel"]]
    kw = dict(ch)
    if ch["channel"] == "tdl":
        kw.update(delay_spread_s=p["delay_spread_ns"] * 1e-9, doppler_hz=p["doppler_hz"])
    # the e015 designs: 2 pilot symbols with the tracker, 3 without it
    design = (EstimatorDesign() if p["pilots"] == 2
              else dataclasses.replace(EstimatorDesign(), doppler_hz=1500.0, track_gain=0.0))
    key = (num.name, p["mcs"], tuple(sorted(kw.items())))
    if key not in _links:
        if len(_links) > 8:
            _links.clear()
        _links[key] = SyncLink(num, MCS[p["mcs"]], design=design, device="cpu", pll_dbc_hz=-92.0,
                               receivers={"float": {}, "fixed": {"fixed": True}}, **kw)
    return _links[key]


def _db(x, floor=-120.0):
    return np.maximum(10 * np.log10(np.maximum(x, 1e-30)), floor)


def _r(a, nd=3):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def _psd(x: np.ndarray, n: int) -> np.ndarray:
    """Averaged periodogram (Hann, half overlap), centred, power per bin."""
    w = np.hanning(n)
    segs = [x[i:i + n] * w for i in range(0, len(x) - n + 1, n // 2)]
    p = (np.abs(np.fft.fft(np.array(segs), axis=-1)) ** 2).mean(0) / (w ** 2).sum()
    return np.fft.fftshift(p)


def _grid(x: np.ndarray, num, start: int = 0) -> np.ndarray:
    """OFDM symbols of a frame at `start` -> [S, effective subcarriers], unitary FFT."""
    N, cp, S = num.fft_size, num.cp_len, num.num_ofdm_symbols
    sym = np.array([x[start + s * (N + cp) + cp:start + (s + 1) * (N + cp)] for s in range(S)])
    X = np.fft.fft(sym, axis=-1) / math.sqrt(N)
    return X[:, subcarrier_offsets(num) % N]


def _nearest(z: np.ndarray, levels: np.ndarray) -> np.ndarray:
    def q(v):
        return levels[np.abs(v[..., None] - levels).argmin(-1)]
    return q(z.real) + 1j * q(z.imag)


def _llr_hist(mag: np.ndarray, bad: np.ndarray, bins: int = 40) -> dict:
    top = float(np.percentile(mag, 99.5)) if mag.size else 1.0
    edges = np.linspace(0, max(top, 1e-3) * 1.0001, bins + 1)
    m = np.minimum(mag, edges[-1] * 0.9999)
    return {"edges": _r(edges, 3), "hist_ok": np.histogram(m[~bad], edges)[0].tolist(),
            "hist_bad": np.histogram(m[bad], edges)[0].tolist()}


def capture(params: dict) -> dict:
    p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    for k in ("esno_db", "delay_spread_ns", "doppler_hz", "cfo_hz", "sir_db"):
        p[k] = float(p[k])
    for k in ("pilots", "seed", "frames"):
        p[k] = int(p[k])
    p["frames"] = max(1, min(p["frames"], 500))
    t_start = time.perf_counter()
    link = _link(p)
    num, mcs, plan, fp = link.num, link.mcs, link.plan, link.fp
    N, cp, S, F = num.fft_size, num.cp_len, num.num_ofdm_symbols, num.num_effective_subcarriers
    fs = num.bandwidth_hz
    batch = p["frames"]
    rng = np.random.default_rng(p["seed"])
    torch.manual_seed(p["seed"])
    sionna.phy.config.seed = p["seed"]
    no = 10 ** (-p["esno_db"] / 10)
    link.set_noise(no)

    # ---- transmitter and channel (SyncLink.propagate, stage by stage) ----
    bits = rng.integers(0, 2, (batch, plan.num_codewords, plan.k), dtype=np.uint8)
    x_q = link.tx.tx(bits)
    x = (x_q[..., 0].astype(np.float32) + 1j * x_q[..., 1].astype(np.float32)) / fp.tx_scale
    L = len(link.pre)
    lead = rng.integers(link.lead_range[0], link.lead_range[1] + 1, batch)
    buf = np.zeros((batch, link.buf_len), np.complex64)
    for i in range(batch):
        buf[i, lead[i]:lead[i] + L] = link.pre
        buf[i, lead[i] + L:lead[i] + L + link.frame_len] = x[i]
    xt = torch.as_tensor(buf).reshape(batch, 1, 1, -1)
    ref = link.ref
    h_t = None
    if ref.channel == "tdl":
        h_t = buffer_time_channel(ref, batch, link.buf_len)
        y = ApplyTime(link.buf_len, ref.l_tot, "cpu")(xt, h_t)
        l_min = ref.l_min
    else:
        y, l_min = xt, 0
    after_channel = y.reshape(batch, -1)[0, :link.buf_len].numpy().copy()
    cfo = np.full(batch, p["cfo_hz"] / fs)
    if p["phase_noise"]:
        y = imp.pll_phase_noise(y, fs, inband_dbc_hz=link.pll)
    y = frame_cfo(y, torch.as_tensor(cfo))
    if p["sco"]:
        y = imp.sco(y, fs, ppm=abs(p["cfo_hz"]) / link.carrier_hz * 1e6)
    if p["wifi"]:
        y = imp.wifi_burst(y, fs, sir_db=p["sir_db"])
    y = ref.awgn(y, torch.tensor(no))
    r = y.reshape(batch, -1).numpy().astype(np.complex128)
    start = lead + L - l_min
    ys, yq, g = link.agc_adc(r, start)
    scale = link.llr_scale(no, g)

    # ---- synchroniser and receiver ----
    fixed = p["receiver"] == "fixed"
    res = link.fxsync(yq) if fixed else link.sync(ys)
    if p["sync"] == "genie":
        ones = np.ones(batch, bool)
        res = SyncResult(ones, start, cfo, start, cfo, res.metric, res.cir_ratio)
    if fixed:
        ya = link.fxsync.align(yq, res.start, inc_of(res.cfo), link.frame_len)
    else:
        ya = link.sync.align(ys, res, link.frame_len)
    m = link.rx[p["receiver"]]
    m.decoder_stats(reset=True)
    b_hat, llr, pr = m.rx(ya, scale, 0, return_llr=True, probe=True)
    iters, cws = m.decoder_stats(reset=True)
    b_hat[~res.detected] = 1 - bits[~res.detected]
    cw_err = (b_hat != bits).any(-1)                                     # [batch, cw]
    coded = ldpc_encode(bits, plan.n)                                    # [batch, cw, n]
    raw_err = (llr > 0) != (coded == 1)          # positive LLR: bit 1 (Sionna's logits)
    raw_err[~res.detected] = True

    # ---- pictures of frame 0 ----
    mask = ref.rg.pilot_pattern.mask[0, 0].cpu().numpy().reshape(-1).astype(bool)
    data_re = np.nonzero(~mask)[0]
    pilot_re = np.nonzero(mask)[0]
    yg = pr["y"][0, :, 0] + 1j * pr["y"][0, :, 1]                       # [S*F] after the tracker
    hd = pr["h"][0, :, 0] + 1j * pr["h"][0, :, 1]                       # [num_data]
    phi = pr["phi"][0] * (2 * math.pi / 65536 if fixed else 1.0)
    X = _grid(x[0], num).reshape(-1)                                     # transmitted REs (unit energy)
    xd = X[data_re]
    eq = yg[data_re] / np.where(np.abs(hd) > 0, hd, 1)
    a = link.a
    levels = (np.arange(-(2 ** (mcs.bits_per_symbol // 2)) + 1, 2 ** (mcs.bits_per_symbol // 2), 2) * a)
    wrong = np.abs(_nearest(eq, levels) - _nearest(xd, levels)) > 1e-6
    evm = np.abs(eq - xd) ** 2
    pre = yg[data_re] / math.sqrt(max(np.mean(np.abs(yg[data_re]) ** 2), 1e-30))
    err_grid = np.full(S * F, np.nan)
    err_grid[data_re] = _db(evm, -60)
    # channel: estimate (data REs of the symbol in the middle) and the true response
    s_mid = S // 2
    sel = (data_re // F) == s_mid
    if not sel.any():
        s_mid = int(data_re[0] // F)
        sel = (data_re // F) == s_mid
    h_est = hd[sel]
    f_est = (data_re[sel] % F).tolist()
    offs = subcarrier_offsets(num)
    if h_t is not None:
        taps = h_t[0, 0, 0, 0, 0].numpy()                                # [time, l_tot]
        t0 = int(start[0] + l_min + s_mid * (N + cp) + cp + N // 2)
        lags = np.arange(l_min, ref.l_max + 1)
        h_true = (taps[t0][None, :] * np.exp(-2j * math.pi * np.outer(offs, lags) / N)).sum(-1)
    else:
        h_true = np.ones(F, np.complex128)
    est_norm = math.sqrt(max(np.mean(np.abs(h_est) ** 2), 1e-30))

    r0 = r[0]
    rx_scale = math.sqrt(max(np.mean(np.abs(r0[start[0]:start[0] + link.frame_len]) ** 2), 1e-30))
    tx_scale = math.sqrt(max(np.mean(np.abs(buf[0, lead[0]:lead[0] + L + link.frame_len]) ** 2), 1e-30))
    f0, fl = lead[0], lead[0] + L + link.frame_len
    # waterfall: short FFTs over the whole buffer
    hop, wN = 32, 128
    w = np.hanning(wN)
    cols = np.array([r0[i:i + wN] * w for i in range(0, len(r0) - wN + 1, hop)])
    wf = _db(np.fft.fftshift(np.abs(np.fft.fft(cols, axis=-1)) ** 2, axes=-1) / (w ** 2).sum() / rx_scale ** 2)
    # synchroniser metric and the channel impulse response it found
    yb = ys[0:1]
    c = yb[:, STF_PERIOD:] * yb[:, :-STF_PERIOD].conj()
    e = np.abs(yb[:, STF_PERIOD:]) ** 2 + np.abs(yb[:, :-STF_PERIOD]) ** 2
    W = link.sp.window
    metric = (4 * np.abs(_running_sum(c, W)) ** 2 / np.maximum(_running_sum(e, W), 1e-30) ** 2)[0]
    nn = np.arange(ys.shape[-1])
    cir = link.sync.cir(ys[0:1] * np.exp(-2j * math.pi * res.coarse_cfo[0] * nn), res.coarse_start[0:1])[0]
    cir_p = np.abs(cir) ** 2
    cir_p = cir_p / max(cir_p.max(), 1e-30)
    cir_lags = list(range(-16, 49))
    cir_db = _db(cir_p[np.array(cir_lags) % N], -40)

    detected = bool(res.detected[0])
    out = {
        "params": p,
        "info": {
            "rate_mbps": phy_rate_bps(num, mcs) / 1e6, "frame_us": num.frame_duration_s * 1e6,
            "codewords": plan.num_codewords, "k": plan.k, "n": plan.n, "fs_mhz": fs / 1e6,
            "fft": N, "cp": cp, "symbols": S, "subcarriers": F, "pilot_symbols": list(num.pilot_ofdm_symbols),
            "spacing_khz": num.subcarrier_spacing_hz / 1e3, "lead": int(lead[0]), "preamble": L,
            "frame_len": int(link.frame_len), "buf_len": int(link.buf_len), "true_start": int(start[0]),
            "doppler_centroid_hz": link.doppler_centroid_hz,
        },
        "sync": {
            "detected": detected, "start": int(res.start[0]), "start_error": int(res.start[0] - start[0]),
            "cfo_hz": float(res.cfo[0] * fs), "cfo_error_hz": float((res.cfo[0] - cfo[0]) * fs),
            "metric_peak": float(res.metric[0]), "threshold": link.sp.threshold,
            "metric": _r(metric[::4], 3), "metric_step": 4, "metric_offset": STF_PERIOD + W,
            "cir_lags": cir_lags, "cir_db": _r(cir_db, 1), "backoff": link.sp.backoff,
        },
        "spectrum": {
            "f_mhz": _r((np.arange(N) - N // 2) * fs / N / 1e6, 4),
            "tx_db": _r(_db(_psd(buf[0, f0:fl], N) / tx_scale ** 2), 2),
            "rx_db": _r(_db(_psd(r0[f0:fl], N) / rx_scale ** 2), 2),
            # noise alone: the silence after the frame, clear of the last echoes
            "noise_db": _r(_db(_psd(r0[fl + ref.l_tot + 8:], N) / rx_scale ** 2), 2),
        },
        "waterfall": {"db": np.clip(np.round(wf), -90, 30).astype(int).tolist(), "hop": hop, "n": wN},
        "waveform": {
            "tx": _r(np.abs(buf[0]) / tx_scale, 3), "rx": _r(np.abs(r0) / rx_scale, 3),
            "tx_i": _r(buf[0].real / tx_scale, 3), "rx_i": _r(r0.real / rx_scale, 3),
        },
        "constellation": {
            "pre": _r(np.stack([pre.real, pre.imag], -1), 3), "eq": _r(np.stack([eq.real, eq.imag], -1), 3),
            "wrong": wrong.astype(int).tolist(), "sym": (data_re // F).astype(int).tolist(),
            # received signal over error, weighted by the channel: deep fades
            # do not swamp it the way an average of the equalised error would
            "levels": _r(levels, 4),
            "snr_db": float(-_db(np.sum(evm * np.abs(hd) ** 2) / max(np.sum(np.abs(xd * hd) ** 2), 1e-30))),
        },
        "channel": {
            "f": f_est, "est_db": _r(_db(np.abs(h_est / est_norm) ** 2, -40), 2),
            "est_phase": _r(np.angle(h_est), 3),
            "true_db": _r(_db(np.abs(h_true) ** 2, -40), 2), "symbol": s_mid, "offsets": offs.tolist(),
        },
        "tracker": {"phi": _r(phi, 4), "on": p["pilots"] == 2},
        "grid": {"S": S, "F": F, "err_db": [None if math.isnan(v) else round(v, 1) for v in err_grid],
                 "pilots": pilot_re.tolist()},
        "llr": _llr_hist(np.abs(llr[0]).reshape(-1), raw_err[0].reshape(-1)),
        "errors": {
            "frames": batch, "frame_errors": int(cw_err.any(-1).sum()), "codewords": int(cw_err.size),
            "codeword_errors": int(cw_err.sum()), "frame0_codewords": cw_err[0].astype(int).tolist(),
            "raw_bit_errors": int(raw_err.sum()), "coded_bits": int(raw_err.size),
            "bit_errors": int((b_hat != bits).sum()), "info_bits": int(bits.size),
            "missed": int((~res.detected).sum()),
            "mean_iterations": iters / cws if cws else 0.0, "max_iterations": fp.num_iter,
        },
    }
    out["elapsed_s"] = time.perf_counter() - t_start
    return out


def options() -> dict:
    return {"mcs": list(MCS), "channels": list(CHANNELS), "defaults": DEFAULTS}
