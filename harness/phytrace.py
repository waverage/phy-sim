"""PHY traces for the link layer (M6): the Es/N0 each frame needs.

The link layer and the video model need the PHY's error pattern over
minutes of consecutive frames, at any SNR and for every MCS, because the
MCS policy switches between them as the SNR moves. Running the modem once
per (SNR, MCS, policy) is too slow, so the trace stores, per frame and
MCS, its decoding threshold: the lowest mean Es/N0 at which that frame,
with its own channel, noise shape, timing and oscillators, decodes. Any
SNR trajectory then gives the frame's outcome as snr[i] >= thr[i, m].

Per frame, as in the M5 link (harness/synclink.py): the fixed-point
transmitter's frame behind its preamble at a random time in a buffer, the
continuous TDL of its link (harness/fading.py) at the frame's absolute
time, PLL noise, the link's carrier offset (one crystal per link, uniform
in +-cfo_hz) and sampling offset, then noise of one fixed shape scaled to
the Es/N0 under test, the AGC, the 12-bit ADC, the fixed-point
synchroniser (its detection and timing) with the warm frequency loop of
M5 (the true offset plus the Doppler centroid), and the fixed-point
receiver with the tracker off (M5's decision behind a warm loop).

The threshold is found by bisection on that one noise shape. Decoding is
close to monotone in the noise scale but not exactly (a decoder can fail
at a higher SNR on a different trajectory of its iterations); the
fraction of frames that break the rule is measured in tests.
"""
from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import impairments as imp
from .design import EstimatorDesign, design_filters, subcarrier_offsets
from .fading import ContinuousTdl
from .sync import SyncParams, inc_of
from .synclink import ApplyTime, SyncLink
from .waveform import MCS, MCS_TABLE, Numerology, NUM_10MHZ

# Es/N0 at 1 % FER on AWGN (e001), the start of each MCS's search
AWGN_1PCT_DB = {"qpsk-1/2": 1.41, "qpsk-3/4": 4.34, "16qam-1/2": 6.53, "16qam-3/4": 10.32,
                "64qam-2/3": 14.26, "64qam-3/4": 15.73, "64qam-5/6": 17.29}

NEVER = math.inf            # threshold of a frame that fails at every SNR up to hi_db


@functools.lru_cache(maxsize=None)
def _filters(num: Numerology, design: EstimatorDesign, snr_db: float) -> dict:
    """design_filters, once per design SNR (about 3 s each)."""
    return design_filters(num, design, snr_db)


@dataclass(frozen=True)
class TraceConfig:
    model: str = "D"                  # TDL model, or "awgn"
    delay_spread_s: float = 100e-9
    doppler_hz: float = 600.0
    carrier_hz: float = 5.8e9
    cfo_hz: float = 116e3
    pll_dbc_hz: float | None = -92.0
    num_links: int = 64               # independent channels, run side by side
    gap_samples: int = 0              # idle samples between frames (TX duty cycle)
    lo_db: float = -8.0               # search range around the AWGN 1 % point: lo below it
    hi_db: float = 45.0               # absolute: frames failing here are NEVER
    resolution_db: float = 0.2
    bracket_db: float = 6.0
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class TraceLink:
    def __init__(self, num: Numerology = NUM_10MHZ, mcs: tuple[MCS, ...] = MCS_TABLE, *,
                 cfg: TraceConfig = TraceConfig(), design: EstimatorDesign | None = None,
                 sync: SyncParams = SyncParams(), device: str = "cpu", num_threads: int = 0):
        self.num, self.cfg, self.device = num, cfg, device
        self.design = design or EstimatorDesign(track_gain=0.0)
        rx = {"fixed": {"fixed": True, "track_gain": self.design.track_gain}}
        self.links = {m.name: SyncLink(num, m, design=self.design, sync=sync, cfo_hz=cfg.cfo_hz,
                                       carrier_hz=cfg.carrier_hz, pll_dbc_hz=cfg.pll_dbc_hz, receivers=rx,
                                       syncs=("warm",), channel="awgn", device=device, num_threads=num_threads)
                      for m in mcs}
        self.mcs = tuple(m.name for m in mcs)
        first = next(iter(self.links.values()))
        self.pre_len, self.frame_len = len(first.pre), first.frame_len
        self.lead_range = first.lead_range
        self.buf_len = first.buf_len
        # one frame on air: preamble + frame (+ gap); frame i of a link starts at i * period
        self.period = self.pre_len + self.frame_len + cfg.gap_samples
        self.fs = num.bandwidth_hz
        rng = np.random.default_rng(cfg.seed)
        self.cfo = rng.uniform(-1, 1, cfg.num_links) * cfg.cfo_hz / self.fs          # cycles/sample, per link
        self.rng = np.random.default_rng(cfg.seed + 1)
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed + 2)   # the noise shapes
        if cfg.model == "awgn":
            self.tdl, self.l_min, self.centroid = None, 0, 0.0
        else:
            self.tdl = ContinuousTdl(cfg.num_links, model=cfg.model, delay_spread_s=cfg.delay_spread_s,
                                     doppler_hz=cfg.doppler_hz, carrier_hz=cfg.carrier_hz, fs=self.fs,
                                     seed=cfg.seed, device=device)
            self.l_min, self.centroid = self.tdl.l_min, self.tdl.doppler_centroid_hz / self.fs
        # the channel's output is l_tot - 1 samples longer than its input (ApplyTimeChannel)
        self.out_len = self.buf_len + (self.tdl.l_tot - 1 if self.tdl is not None else 0)
        self.freqs = subcarrier_offsets(num) * num.subcarrier_spacing_hz
        self.timing = {"channel": 0.0, "decode": 0.0, "decodes": 0}

    @property
    def frame_seconds(self) -> float:
        return self.period / self.fs

    # ---- one block of frames ----------------------------------------------------------------

    def _clean(self, frames: np.ndarray, link: np.ndarray, lead: np.ndarray, h, sl: SyncLink, bits: np.ndarray):
        """Received buffers before noise [rows, buf_len] complex128."""
        rows = len(frames)
        x_q = sl.tx.tx(bits)
        x = (x_q[..., 0].astype(np.float32) + 1j * x_q[..., 1].astype(np.float32)) / sl.fp.tx_scale
        buf = np.zeros((rows, self.buf_len), np.complex64)
        L = self.pre_len
        for i in range(rows):
            buf[i, lead[i]:lead[i] + L] = sl.pre
            buf[i, lead[i] + L:lead[i] + L + self.frame_len] = x[i]
        y = torch.as_tensor(buf, device=self.device).reshape(rows, 1, 1, -1)
        if h is not None:
            y = ApplyTime(self.buf_len, self.tdl.l_tot, self.device)(y, h)
        if self.cfg.pll_dbc_hz is not None:
            y = imp.pll_phase_noise(y, self.fs, inband_dbc_hz=self.cfg.pll_dbc_hz)
        # the link's offset, continuous in absolute time (the buffer starts at frame * period - lead)
        n0 = frames * self.period - lead
        # output sample j is input time j + l_min
        n = torch.arange(self.out_len, device=self.device, dtype=torch.float64) + self.l_min
        cfo = torch.as_tensor(self.cfo[link], device=self.device)
        ph = 2 * math.pi * cfo[:, None] * (torch.as_tensor(n0, device=self.device, dtype=torch.float64)[:, None] + n)
        y = y.reshape(rows, -1) * torch.exp(1j * torch.remainder(ph, 2 * math.pi)).to(y.dtype)
        y = imp.sco(y, self.fs, ppm=float(self.cfg.cfo_hz / self.cfg.carrier_hz * 1e6))
        return y.to(torch.complex64)

    def _agc_adc(self, sl: SyncLink, r: torch.Tensor, start: np.ndarray):
        """SyncLink.agc_adc on the device: the gain from each frame's own
        power, then the ADC. Returns int16 [rows, len, 2] and the gain."""
        idx = torch.as_tensor(start, device=self.device)[:, None] + torch.arange(self.frame_len, device=self.device)
        rms = torch.sqrt((torch.take_along_dim(r, idx, -1).abs() ** 2).mean(-1)).clamp_min(1e-12)
        g = math.sqrt(sl.fp.agc_power) / rms
        ys = r * g[:, None]
        hi = 2 ** (sl.fp.adc_bits - 1)
        q = torch.stack([ys.real, ys.imag], -1)
        yq = torch.clamp(torch.floor(q + 0.5), -hi, hi - 1).to(torch.int16)
        return yq.cpu().numpy(), g.cpu().numpy().astype(np.float64)

    def _decode(self, sl: SyncLink, r_clean, w, start, link, bits, esno_db):
        """ok [rows] at per-row Es/N0 esno_db."""
        no = 10 ** (-esno_db / 10)
        r = r_clean + torch.as_tensor(np.sqrt(no), device=self.device, dtype=torch.float32)[:, None] * w
        yq, g = self._agc_adc(sl, r, start)
        scale = sl.llr_scale(no, g)
        res = sl.fxsync(yq)
        cfo = self.cfo[link] + self.centroid                        # warm loop (M5)
        ok = np.zeros(len(r), bool)
        m = sl.rx["fixed"]
        snr_class = np.array([self.design.design_snr_db(s) for s in esno_db])
        for c in np.unique(snr_class):
            sel = np.nonzero((snr_class == c) & res.detected)[0]
            if not len(sel):
                continue
            m.set_filter(_filters(self.num, self.design, float(c)))
            y = sl.fxsync.align(yq[sel], res.start[sel], inc_of(cfo[sel]), self.frame_len)
            b_hat = m.rx(y, scale[sel], 0)
            ok[sel] = (b_hat == bits[sel]).all(axis=(1, 2))
        return ok

    def block(self, first_frame: int, frames_per_link: int, links: np.ndarray | None = None,
              check_db: dict | None = None) -> dict:
        """Frames first_frame .. first_frame + frames_per_link - 1 of each link:
        {"thr": {mcs: [links, frames] dB}, "gain_db", "geo_gain_db": [links, frames]}.
        check_db {mcs: [Es/N0, ...]}: also decode every frame at those Es/N0
        with the search's noise shape ("same") and with a fresh one
        ("fresh"), for the tests: {"check": {mcs: {"same"|"fresh": [points, links, frames]}}}."""
        cfg = self.cfg
        links = np.arange(cfg.num_links) if links is None else np.asarray(links)
        fr = np.arange(first_frame, first_frame + frames_per_link)
        link = np.repeat(links, frames_per_link)
        frames = np.tile(fr, len(links))
        rows = len(frames)
        lead = self.rng.integers(self.lead_range[0], self.lead_range[1] + 1, rows)
        t0 = time.perf_counter()
        h = None
        if self.tdl is not None:
            h = self.tdl.time_channel(frames * self.period - lead, self.buf_len, link)
            h = h.reshape(rows, 1, 1, 1, 1, *h.shape[1:])
            # the channel each frame sees, mid-frame (for the policy's measurements and the search's start)
            tm = (frames * self.period + self.pre_len + self.frame_len / 2) / self.fs
            H = self.tdl.frequency_response(tm[:, None], self.freqs, link)[:, 0]   # [rows, F]
            p = np.abs(H) ** 2
            gain_db = 10 * np.log10(p.mean(-1))
            geo_db = 10 * np.log10(np.exp(np.log(np.maximum(p, 1e-12)).mean(-1)))
        else:
            gain_db = geo_db = np.zeros(rows)
        w = torch.randn((rows, self.out_len), generator=self.gen, device=self.device, dtype=torch.complex64)
        start = lead + self.pre_len - self.l_min
        self.timing["channel"] += time.perf_counter() - t0
        thr, chk = {}, {}
        for name, sl in self.links.items():
            p = sl.plan
            bits = self.rng.integers(0, 2, (rows, p.num_codewords, p.k), dtype=np.uint8)
            t0 = time.perf_counter()
            clean = self._clean(frames, link, lead, h, sl, bits)
            self.timing["channel"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            thr[name] = self._search(sl, clean, w, start, link, bits, AWGN_1PCT_DB[name] - geo_db).reshape(len(links), -1)
            self.timing["decode"] += time.perf_counter() - t0
            if check_db and name in check_db:
                w2 = torch.randn(w.shape, generator=self.gen, device=self.device, dtype=w.dtype)
                chk[name] = {k: np.stack([self._decode(sl, clean, ww, start, link, bits, np.full(rows, s, float))
                                          .reshape(len(links), -1) for s in check_db[name]])
                             for k, ww in (("same", w), ("fresh", w2))}
        out = {"thr": thr, "gain_db": gain_db.reshape(len(links), -1), "geo_gain_db": geo_db.reshape(len(links), -1)}
        if check_db:
            out["check"] = chk
        return out

    def _search(self, sl, clean, w, start, link, bits, guess_db):
        """Bisection for each row's threshold, starting from a bracket around guess_db."""
        cfg = self.cfg
        name = sl.mcs.name
        lo_cap, hi_cap = AWGN_1PCT_DB[name] + cfg.lo_db, cfg.hi_db
        rows = len(clean)
        mid = np.clip(guess_db, lo_cap + cfg.bracket_db / 2, hi_cap - cfg.bracket_db / 2)
        lo, hi = mid - cfg.bracket_db / 2, mid + cfg.bracket_db / 2
        known_lo = np.zeros(rows, bool)          # lo is known to fail
        known_hi = np.zeros(rows, bool)          # hi is known to decode
        never = np.zeros(rows, bool)
        always = np.zeros(rows, bool)

        def dec(idx, s):
            self.timing["decodes"] += len(idx)
            ti = torch.as_tensor(idx, device=self.device)
            return self._decode(sl, clean[ti], w[ti], start[idx], link[idx], bits[idx], s)

        # establish the bracket: hi must decode, lo must fail; widen outwards until the caps
        while True:
            idx = np.nonzero(~known_hi & ~never)[0]
            if not len(idx):
                break
            ok = dec(idx, hi[idx])
            good, bad = idx[ok], idx[~ok]
            known_hi[good] = True
            atcap = bad[hi[bad] >= hi_cap - 1e-9]
            never[atcap] = True
            up = bad[hi[bad] < hi_cap - 1e-9]
            lo[up], known_lo[up] = hi[up], True
            hi[up] = np.minimum(hi[up] + cfg.bracket_db, hi_cap)
        while True:
            idx = np.nonzero(~known_lo & ~never & ~always)[0]
            if not len(idx):
                break
            ok = dec(idx, lo[idx])
            good, bad = idx[ok], idx[~ok]
            known_lo[bad] = True
            atcap = good[lo[good] <= lo_cap + 1e-9]
            always[atcap] = True
            down = good[lo[good] > lo_cap + 1e-9]
            hi[down] = lo[down]
            lo[down] = np.maximum(lo[down] - cfg.bracket_db, lo_cap)
        act = ~never & ~always
        while True:
            idx = np.nonzero(act & (hi - lo > cfg.resolution_db))[0]
            if not len(idx):
                break
            m = (lo[idx] + hi[idx]) / 2
            ok = dec(idx, m)
            hi[idx[ok]] = m[ok]
            lo[idx[~ok]] = m[~ok]
        thr = (lo + hi) / 2
        thr[always] = lo_cap
        thr[never] = NEVER
        return thr
