"""The M5 link: each frame behind its preamble at a random time in a sample
buffer, with the full carrier offset of the oscillators, then the
synchroniser and the product receiver.

Per frame: the fixed-point transmitter's frame (M4) behind the preamble,
`lead` samples of silence before it (uniform in lead_range), Sionna's TDL
(or AWGN) over the whole buffer, the receive oscillator (a carrier offset
per frame, uniform in +-cfo_hz, the same crystal's sampling clock offset,
PLL phase noise), noise, an ideal AGC on the frame and the ADC. Receivers
decode the same samples after one of these synchronisations:
  sync       the synchroniser (harness/sync.py), cold: this frame's estimates;
  warm       the synchroniser's timing, and a frequency loop that has
             averaged many frames: the true offset plus the Doppler centroid
             plus the mean of this batch's per-frame errors (frames are
             independent, so that is the loop's error over `batch` frames);
  aided      warm, less the Doppler centroid, which a control processor can
             predict from both ends' positions and velocities (telemetry);
  genie      the true frame start and carrier offset (what M4 assumed);
  genie_afc  the true start, and the carrier offset plus the channel's
             Doppler centroid (the line of sight's power times its
             Doppler): what an ideal frequency estimate removes.
Genie windows are placed like the synchroniser's (backoff before the first
path), so sync - genie is what synchronisation costs. The first receiver
after `sync` gives the plain counts; the others are prefixed
"{sync}_{receiver}_".
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from sionna.phy.channel import ApplyTimeChannel

from . import impairments as imp
from .chain import OfdmLink, count_errors
from .design import EstimatorDesign, design_filters
from .fxmodem import FixedPoint, FxModem, frame_tables
from .sync import FloatSync, FxSync, SyncParams, SyncResult, buffer_time_channel, frame_cfo, inc_of, preamble
from .twin import qam_norm
from .waveform import MCS, Numerology

SYNCS = ("sync", "warm", "aided", "genie", "genie_afc")


class SyncLink:
    def __init__(self, num: Numerology, mcs: MCS, *, design: EstimatorDesign = EstimatorDesign(),
                 sync: SyncParams = SyncParams(), fp: FixedPoint = FixedPoint(), cfo_hz: float = 116e3,
                 carrier_hz: float = 5.8e9, sco: bool = True, pll_dbc_hz: float | None = -92.0,
                 lead_range: tuple[int, int] = (64, 1088), preamble_boost_db: float = 0.0,
                 receivers: dict | None = None, syncs: tuple[str, ...] = ("sync", "genie"),
                 num_threads: int = 0, **link_kw):
        link_kw = dict(link_kw, estimator="lmmse", compile=False, carrier_hz=carrier_hz)
        self.ref = OfdmLink(num, mcs, **link_kw)
        self.num, self.mcs, self.plan, self.device = num, mcs, self.ref.plan, self.ref.device
        self.design, self.fp, self.sp = design, fp, sync
        self.cfo_hz, self.carrier_hz, self.sco, self.pll = cfo_hz, carrier_hz, sco, pll_dbc_hz
        self.lead_range = lead_range
        self.syncs = syncs
        assert set(syncs) <= set(SYNCS)
        doppler = link_kw.get("doppler_hz", 0.0)
        tdl = getattr(self.ref, "tdl", None)
        self.doppler_centroid_hz = (float(tdl.mean_power_los) * math.cos(float(tdl._los_angle_of_arrival)) * doppler
                                    if tdl is not None and tdl.los else 0.0)
        mask = self.ref.rg.pilot_pattern.mask[0, 0].cpu().numpy()
        pilots, _ = frame_tables(num, mcs, mask)
        p = self.plan
        kw = dict(k=p.k, n=p.n, num_codewords=p.num_codewords, pilot_mask=mask, fp=fp, taps=design.taps,
                  phase_track=True, num_threads=num_threads)
        self.tx = FxModem(num, mcs, fixed=True, track_gain=design.track_gain, **kw)
        receivers = receivers or {"float": {}}
        self.rx = {name: FxModem(num, mcs, fixed=r.get("fixed", False),
                                 track_gain=r.get("track_gain", design.track_gain), **kw)
                   for name, r in receivers.items()}
        self.sync = FloatSync(num, sync, pilots=pilots)
        self.fxsync = FxSync(num, sync, fp.tx_scale, fp.dac_bits, num_threads)
        if preamble_boost_db:
            self.pre = preamble(num, boost_db=preamble_boost_db).astype(np.complex64)
        else:   # the fixed-point transmitter's
            pq = self.fxsync.preamble().astype(np.float32)
            self.pre = ((pq[:, 0] + 1j * pq[:, 1]) / fp.tx_scale).astype(np.complex64)
        self.a = qam_norm(mcs.bits_per_symbol)
        self.frame_len = self.tx.num_samples
        self.buf_len = lead_range[1] + len(self.pre) + self.frame_len + 2 * num.fft_size
        self._design_snr = None

    def set_noise(self, no: float):
        snr = self.design.design_snr_db(-10 * math.log10(no))
        if snr != self._design_snr:
            filt = design_filters(self.num, self.design, snr)
            for m in self.rx.values():
                m.set_filter(filt)
            self._design_snr = snr

    def llr_scale(self, no: float, gain: np.ndarray) -> np.ndarray:
        fp = self.fp
        scale = self.a / (no * gain.astype(np.float64) ** 2 * fp.llr_step_nat)
        return np.minimum(scale, fp.llr_cap / (self.a * fp.agc_power))

    def propagate(self, b_np: np.ndarray, no: float):
        """Returns the received buffers [batch, len] complex128, the true frame
        start per frame (index of the frame's first sample, after the
        preamble, through the direct path) and the true CFO (cycles/sample)."""
        batch = b_np.shape[0]
        x_q = self.tx.tx(b_np)
        x = (x_q[..., 0].astype(np.float32) + 1j * x_q[..., 1].astype(np.float32)) / self.fp.tx_scale
        lead = np.random.randint(self.lead_range[0], self.lead_range[1] + 1, batch)
        buf = np.zeros((batch, self.buf_len), np.complex64)
        L = len(self.pre)
        for i in range(batch):
            buf[i, lead[i]:lead[i] + L] = self.pre
            buf[i, lead[i] + L:lead[i] + L + self.frame_len] = x[i]
        xt = torch.as_tensor(buf, device=self.device).reshape(batch, 1, 1, -1)
        ref = self.ref
        if ref.channel == "tdl":
            h = buffer_time_channel(ref, batch, self.buf_len)
            y = ApplyTime(self.buf_len, ref.l_tot, self.device)(xt, h)
            l_min = ref.l_min
        else:
            y, l_min = xt, 0
        fs = self.num.bandwidth_hz
        cfo = (np.random.uniform(-1, 1, batch) * self.cfo_hz / fs)
        if self.pll is not None:
            y = imp.pll_phase_noise(y, fs, inband_dbc_hz=self.pll)
        y = frame_cfo(y, torch.as_tensor(cfo, device=self.device))
        if self.sco:
            # the receiver's crystal sets both: ppm = CFO / carrier (worst case of the batch's sign)
            y = imp.sco(y, fs, ppm=float(self.cfo_hz / self.carrier_hz * 1e6))
        y = ref.awgn(y, torch.tensor(no, device=self.device))
        start = lead + L - l_min
        return y.reshape(batch, -1).cpu().numpy().astype(np.complex128), start, cfo

    def agc_adc(self, r: np.ndarray, start: np.ndarray):
        """Ideal AGC (gain from each frame's own power) and the ADC on the whole
        buffer: (float samples in ADC units, int16 [batch, len, 2], gain)."""
        idx = start[:, None] + np.arange(self.frame_len)[None, :]
        rms = np.sqrt((np.abs(np.take_along_axis(r, idx, -1)) ** 2).mean(-1)).clip(1e-12)
        g = math.sqrt(self.fp.agc_power) / rms
        ys = r * g[:, None]
        hi = 2 ** (self.fp.adc_bits - 1)
        q = np.stack([ys.real, ys.imag], -1)
        yq = np.clip(np.floor(q + 0.5), -hi, hi - 1).astype(np.int16)
        return ys, yq, g

    def _modes(self, res: SyncResult, start, cfo) -> dict:
        batch = len(start)
        ones = np.ones(batch, bool)
        centroid = self.doppler_centroid_hz / self.num.bandwidth_hz
        # warm: a frequency loop that has averaged many frames' estimates; its
        # error is the mean of this batch's per-frame errors (frames are
        # independent), the timing still per frame
        err = (res.cfo - cfo - centroid)[res.detected]
        warm = cfo + centroid + (err.mean() if err.size else 0.0)
        keep = (res.coarse_start, res.coarse_cfo, res.metric, res.cir_ratio)
        return {"sync": res,
                "warm": SyncResult(res.detected, res.start, warm, *keep),
                "aided": SyncResult(res.detected, res.start, warm - centroid, *keep),
                "genie": SyncResult(ones, start, cfo, start, cfo, ones, ones),
                "genie_afc": SyncResult(ones, start, cfo + centroid, start, cfo, ones, ones)}

    def run(self, batch: int, esno_db: float, return_sync: bool = False):
        no = 10 ** (-esno_db / 10)
        self.set_noise(no)
        p = self.plan
        b = self.ref.source([batch, p.num_codewords, p.k])
        bb = b.to(torch.uint8).cpu().numpy()
        r, start, cfo = self.propagate(bb, no)
        ys, yq, g = self.agc_adc(r, start)
        scale = self.llr_scale(no, g)
        out = {}
        res = {}
        if any(not m.fixed for m in self.rx.values()):
            t0 = time.perf_counter()
            res[False] = self.sync(ys)
            out["sync_float_s"] = time.perf_counter() - t0
        if any(m.fixed for m in self.rx.values()):
            t0 = time.perf_counter()
            res[True] = self.fxsync(yq)
            out["sync_fixed_s"] = time.perf_counter() - t0
        modes = {k: self._modes(v, start, cfo) for k, v in res.items()}
        first = True
        for name in self.syncs:
            for rname, m in self.rx.items():
                sr = modes[m.fixed][name]
                if m.fixed:
                    y = self.fxsync.align(yq, sr.start, inc_of(sr.cfo), self.frame_len)
                else:
                    y = self.sync.align(ys, sr, self.frame_len)
                b_hat = m.rx(y, scale, 0)
                b_hat[~sr.detected] = 1 - bb[~sr.detected]          # a missed frame is a frame error
                pre = "" if first else f"{name}_{rname}_"
                first = False
                e = count_errors(b, torch.as_tensor(b_hat, device=b.device).to(b.dtype))
                out.update({f"{pre}{k}": v for k, v in e.items() if not pre or k not in ("frames", "codewords", "bits")})
        stats = {}
        fs = self.num.bandwidth_hz
        for fixed, rs in res.items():
            tag = "fixed" if fixed else "float"
            dt = rs.start - start
            df = (rs.cfo - cfo) * fs - self.doppler_centroid_hz
            ok = rs.detected & (np.abs(dt) <= self.sp.backoff - 1) & (np.abs(df) <= 1000)
            out.update({f"sync_{tag}_missed": int((~rs.detected).sum()), f"sync_{tag}_failed": int((~ok).sum())})
            stats[tag] = {"dt": dt, "df_hz": df, "detected": rs.detected, "coarse_dt": rs.coarse_start - start,
                          "coarse_df_hz": (rs.coarse_cfo - cfo) * fs - self.doppler_centroid_hz,
                          "metric": rs.metric, "cir_ratio": rs.cir_ratio}
        if return_sync:
            return out, stats
        return out


class ApplyTime:
    """Sionna's ApplyTimeChannel, cached per length."""
    _cache: dict = {}

    def __new__(cls, n: int, l_tot: int, device: str):
        key = (n, l_tot, device)
        if key not in cls._cache:
            cls._cache[key] = ApplyTimeChannel(n, l_tot, device=device)
        return cls._cache[key]
