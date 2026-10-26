"""The link layer and video model (M6), slot by slot over a PHY trace.

The custom radio is time-division: a superframe of `sf_slots` PHY frame
slots (432 us each), the last one uplink (ground to air: retransmission
requests, the decode acknowledgement, the receiver's SNR and loss report),
the others downlink video. Each downlink slot carries one link packet
in one PHY frame at the slot's MCS; the packet arrives iff the slot's mean
Es/N0 reaches the frame's decoding threshold in the trace (e016). The
uplink slot is qpsk-1/2 on the same (reciprocal) channel plus
`ul_offset_db`.

Above the PHY, the rules of nestua-zig (the Wi-Fi link this radio
replaces), configurable so they can be tuned for a radio that owns its
airtime:
  - packetisation: a video frame never shares a block; blocks of D data
    packets plus EC Reed-Solomon parity, the tail rules of
    tx_buffer.zig (merged tail, proportional tail parity, no parity on a
    one-packet block or on frames up to small_frame_bytes);
  - retransmission: the ground asks for the missing data packets of
    incomplete blocks (a whole block when nothing of it arrived), once a
    hole is stable (a later block seen, or `idle_ms` without packets), not
    again for `rerequest_ms`, up to `max_entries` per request, and stops
    when less than `stop_ms` of the frame's window is left; the air sends
    retransmissions before new packets;
  - recovery: a frame that is lost, late or whose reference is missing
    freezes the picture; the ground asks for a resync every uplink slot
    until one decodes; the air answers with an LTR re-anchor (a P frame on
    the newest long-term reference the ground has acknowledged) or, with
    no confirmed anchor, an IDR, within cooldowns;
  - the encoder: capped bitrate, periodic IDRs, LTR anchors every
    ltr_frames in two alternating slots; recovery frames cost more than a
    P frame (nestua-zig docs/video-ltr-recovery-roadmap.md).
A frame not complete `latency_ms` after it left the encoder is lost (the
display shows frames as they decode, without a jitter buffer), and the air
drops packets of frames past that deadline.

The MCS and the encoder bitrate come from a policy (fixed, SNR-driven with
an outer loop, or nestua-zig's slow loss-driven ladder), fed by the
uplink reports.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict, replace

import numpy as np

from .waveform import MCS_TABLE, NUM_10MHZ, plan_codewords

MCS_NAMES = tuple(m.name for m in MCS_TABLE)
# frame payload (information bytes) per MCS: one LDPC frame, 432 us with the preamble
PAYLOAD_BYTES = {m.name: plan_codewords(NUM_10MHZ, m).info_bits_per_frame // 8 for m in MCS_TABLE}
AWGN_1PCT_DB = {"qpsk-1/2": 1.41, "qpsk-3/4": 4.34, "16qam-1/2": 6.53, "16qam-3/4": 10.32,
                "64qam-2/3": 14.26, "64qam-3/4": 15.73, "64qam-5/6": 17.29}
# nestua-zig per-packet overhead on air: packet header 24 (CRC32, ids, type), video
# segment header 36, the 3-byte header inside the FEC payload, AEAD counter 8 + tag 16
NESTUA_HEADER_BYTES = 24 + 36 + 3 + 24
# LTR re-anchor size over a normal P frame by the gap it spans (s), nestua-zig's x265
# measurement (gap 5, 15, 30, 60 frames at 30 fps)
LTR_GAP_S = (0.0, 5 / 30, 15 / 30, 1.0, 2.0)
LTR_FACTOR = (1.0, 1.5, 2.2, 3.0, 3.9)


@dataclass(frozen=True)
class LinkConfig:
    # ---- MAC
    sf_slots: int = 23                  # superframe: 23 x 432 us = 9.94 ms
    ul_offset_db: float = 0.0           # uplink Es/N0 over the downlink's
    header_bytes: int = NESTUA_HEADER_BYTES
    # FEC per MCS (data, parity); nestua-zig's ladder: 8+3 on the 64-QAM rungs, 4+4 below
    fec: tuple = (("qpsk-1/2", 4, 4), ("qpsk-3/4", 4, 4), ("16qam-1/2", 4, 4), ("16qam-3/4", 4, 4),
                  ("64qam-2/3", 8, 3), ("64qam-3/4", 8, 3), ("64qam-5/6", 8, 3))
    small_frame_bytes: int = 2200       # frames up to this size get no parity (retransmission only)
    fill_parity: int = 0                # idle downlink slots send up to this many extra parity per block
    # ---- retransmission (nestua-zig defaults)
    retransmit: bool = True
    idle_ms: float = 4.0
    rerequest_ms: float = 10.0
    max_entries: int = 10
    stop_ms: float = 20.0               # stop asking when min(window/2, stop_ms) of the window is left
    stop_scaled: bool = True            # False: stop when stop_ms is left
    # ---- video
    fps: float = 60.0
    latency_ms: float = 30.0            # encoder output to decoded, else lost
    gop_s: float = 2.0                  # periodic IDR
    # IDR size over a P frame. The product encoder (i.MX 8M Plus Hantro, capped VBR,
    # intraQpDelta 0) made a scheduled I at 4 Mbit/s 23 KB, 1.4 P frames at 30 fps
    # (nestua-zig docs/imx8mp-encoder-tuning.md); x265 at constant QP makes it 9.4.
    idr_factor: float = 1.5             # periodic IDR, inside the GOP's rate
    idr_recovery_factor: float = 1.5    # a recovery IDR, on top of the rate
    size_sigma: float = 0.25            # lognormal frame size spread
    ltr_frames: int = 40                # anchor period (20 at 30 fps in nestua-zig)
    ltr_max_age_s: float = 5.0
    ltr_cooldown_ms: float = 100.0      # recover_cooldown_ms
    idr_min_gap_ms: float = 1000.0
    resync_dup: int = 2                 # copies of each recovery-frame packet (reanchor_dup)
    bitrate_fraction: float = 0.85      # encoder bitrate over the rung's goodput
    # the encoder's per-frame size cap, as a fraction of what the rung's goodput carries
    # within latency_ms (copies included); 0: no cap (nestua-zig)
    frame_cap: float = 0.0
    max_bitrate_bps: float = 20e6
    # ---- receiver's SNR estimate (pilots): error in dB, scaled by (1 + 1/snr)
    snr_est_db: float = 0.22
    freeze_ms: float = 50.0             # a display gap above this is a freeze

    def fec_of(self, mcs: str) -> tuple[int, int]:
        for m, d, e in self.fec:
            if m == mcs:
                return d, e
        raise KeyError(mcs)

    def payload(self, mcs: str) -> int:
        return PAYLOAD_BYTES[mcs] - self.header_bytes

    def goodput_bps(self, mcs: str, slot_s: float) -> float:
        d, e = self.fec_of(mcs)
        dl = (self.sf_slots - 1) / (self.sf_slots * slot_s)
        return self.payload(mcs) * 8 * dl * d / (d + e)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["fec"] = [list(x) for x in self.fec]
        return d


def with_fec(cfg: LinkConfig, d: int, e: int, mcs=MCS_NAMES) -> LinkConfig:
    fec = tuple((m, d, e) if m in mcs else (m, dd, ee) for m, dd, ee in cfg.fec)
    return replace(cfg, fec=fec)


# ---- the policies -------------------------------------------------------------------------------

class FixedPolicy:
    def __init__(self, mcs: str):
        self.mcs = mcs

    def start(self, link):
        return MCS_NAMES.index(self.mcs)

    def report(self, link, rep):
        return None

    def missed(self, link):
        return None


class SnrPolicy:
    """MCS from the reported SNR with an outer loop (OLLA, as LTE does on
    first-transmission BLER): the highest MCS whose required Es/N0
    (req_db, default the AWGN 1 % points) plus margin_db is below the
    smoothed estimate less an offset. The offset rises by up_db per lost
    downlink packet and falls by up_db * target / (1 - target) per
    delivered one, so the pre-FEC packet loss settles near `target`, which
    the FEC and retransmission absorb; a failed block adds block_db at
    once. Steps down at once, up one MCS after hold_ms above the next
    threshold."""

    def __init__(self, margin_db: float = 3.0, tau_ms: float = 50.0, target: float = 0.05, up_db: float = 0.05,
                 block_db: float = 1.0, hold_ms: float = 200.0, start_mcs: str = "qpsk-1/2", olla: bool = True,
                 req_db: dict | None = None, off_range: tuple = (-6.0, 20.0), penalty_db: float = 0.0,
                 penalty_decay_db_s: float = 0.3, probe_s: float = 1.0):
        self.margin, self.tau, self.target, self.up = margin_db, tau_ms / 1e3, target, up_db
        self.block_db, self.hold, self.start_mcs, self.olla = block_db, hold_ms / 1e3, start_mcs, olla
        self.req = dict(req_db or AWGN_1PCT_DB)
        self.off_range = off_range
        # a step up that loses a block within probe_s raises that MCS's threshold by
        # penalty_db, decaying at penalty_decay_db_s (nestua-zig's probe backoff, in dB)
        self.penalty_db, self.decay, self.probe_s = penalty_db, penalty_decay_db_s, probe_s

    def start(self, link):
        self.s, self.off, self.t_last, self.above_since = None, 0.0, None, None
        self.pen = [0.0] * len(MCS_NAMES)
        self.t_pen, self.up_at = 0.0, None
        return MCS_NAMES.index(self.start_mcs)

    def report(self, link, rep):
        t = rep["t"]
        if rep["snr_db"] is not None:
            if self.s is None:
                self.s = rep["snr_db"]
            else:
                a = 1 - math.exp(-(t - self.t_last) / self.tau)
                self.s += a * (rep["snr_db"] - self.s)
            self.t_last = t
        if self.olla:
            ok = rep["sent"] - rep["lost"]
            self.off += self.up * rep["lost"] - self.up * self.target / (1 - self.target) * ok
            self.off += self.block_db * rep["blocks_failed"]
            self.off = min(max(self.off, self.off_range[0]), self.off_range[1])
        cur = link.mcs_i
        if self.penalty_db:
            dec = self.decay * (t - self.t_pen)
            self.pen = [max(p - dec, 0.0) for p in self.pen]
            self.t_pen = t
            if self.up_at is not None:
                if rep["blocks_failed"] > 0 and t - self.up_at[0] <= self.probe_s:
                    self.pen[self.up_at[1]] += self.penalty_db
                    self.up_at = None
                elif t - self.up_at[0] > self.probe_s:
                    self.up_at = None
        if self.s is None:
            return None
        eff = self.s - self.off - self.margin
        best = 0
        for i, m in enumerate(MCS_NAMES):
            if eff >= self.req[m] + self.pen[i]:
                best = i
        if best < cur:
            self.above_since = None
            return best
        if best > cur:
            if self.above_since is None:
                self.above_since = t
            if t - self.above_since >= self.hold:
                self.above_since = t
                self.up_at = (t, cur + 1)
                return cur + 1
        else:
            self.above_since = None
        return None

    def missed(self, link):
        return None


class LadderPolicy:
    """nestua-zig's ladder (nestua-vehicle/src/link_profile.zig), reduced to
    the rules that matter on a link with one rung per MCS: 1 s health
    windows; a hard failure (a frame lost after FEC and retransmission, or
    3 or more blocks failed) steps one rung down at once; a soft one (pre-FEC loss >= 6 %) after a streak of 3
    (one clean window takes one off the streak); a probe one rung up at
    most every 7 s, kept after 2 good windows, else back down with a
    backoff that doubles from 5 s to 180 s and is forgiven after
    max(12, backoff) good windows; 2 s of uplink silence: the lowest rung."""

    def __init__(self, start_mcs: str = "16qam-1/2"):
        self.start_mcs = start_mcs

    def start(self, link):
        self.win_t0, self.acc = 0.0, self._zero()
        self.streak, self.probe = 0, None
        self.next_probe, self.backoff, self.good_run = 7.0, 5.0, 0
        self.last_report = 0.0
        return MCS_NAMES.index(self.start_mcs)

    @staticmethod
    def _zero():
        return {"sent": 0, "lost": 0, "frames_lost": 0, "blocks_failed": 0}

    def report(self, link, rep):
        t = rep["t"]
        self.last_report = t
        for k in self.acc:
            self.acc[k] += rep[k]
        if t - self.win_t0 < 1.0:
            return None
        a, self.acc, self.win_t0 = self.acc, self._zero(), t
        cur = link.mcs_i
        # hard: frames lost after FEC and retransmission (nestua's forced advances and
        # loss collapse); soft: the FEC margin share, here the pre-FEC packet loss
        hard = a["frames_lost"] > 0 or a["blocks_failed"] >= 3
        soft = a["sent"] > 0 and a["lost"] / a["sent"] >= 0.06
        if self.probe is not None:
            if hard or soft:
                self.probe = None
                self.next_probe = t + self.backoff
                self.backoff = min(self.backoff * 2, 180.0)
                self.good_run = 0
                return max(cur - 1, 0)
            self.probe += 1
            if self.probe >= 2:
                self.probe = None
                self.next_probe = t          # a successful probe skips the wait
            return None
        if hard:
            self.streak, self.good_run = 0, 0
            self.next_probe = max(self.next_probe, t + 7.0)
            return max(cur - 1, 0)
        if soft:
            self.streak += 1
            self.good_run = 0
            if self.streak >= 3:
                self.streak = 0
                self.next_probe = max(self.next_probe, t + 7.0)
                return max(cur - 1, 0)
            return None
        self.streak = max(self.streak - 1, 0)
        self.good_run += 1
        if self.good_run >= max(12, self.backoff):
            self.backoff = 5.0
        if t >= self.next_probe and cur < len(MCS_NAMES) - 1:
            self.probe = 0
            self.next_probe = t + 7.0
            return cur + 1
        return None

    def missed(self, link):
        return 0 if link.t - self.last_report >= 2.0 else None


# ---- a flight -----------------------------------------------------------------------------------

def flight_snr(slot_s: float, *, snr_1km_db: float, d_min_m: float = 200.0, d_max_m: float = 4000.0,
               speed_mps: float = 30.0, shadow_db: float = 2.0, shadow_m: float = 50.0, dips_per_min: float = 4.0,
               dip_db: float = 10.0, dip_s: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean Es/N0 per slot of an out-and-back flight at constant speed:
    free-space path loss from snr_1km_db, log-normal shadowing (AR(1) over
    distance travelled, Gudmundson), and antenna-pattern dips (banking: dip_db
    for dip_s, raised-cosine edges, Poisson in time). Returns (snr_db, distance_m)."""
    rng = np.random.default_rng(seed)
    t_leg = (d_max_m - d_min_m) / speed_mps
    n = int(round(2 * t_leg / slot_s))
    t = np.arange(n) * slot_s
    d = d_min_m + speed_mps * np.where(t < t_leg, t, 2 * t_leg - t)
    snr = snr_1km_db - 20 * np.log10(d / 1000.0)
    # shadowing on a coarse grid of 1 m steps of travel, interpolated
    step_m = 1.0
    m = int(math.ceil(2 * t_leg * speed_mps / step_m)) + 2
    rho = math.exp(-step_m / shadow_m)
    s = np.zeros(m)
    w = rng.standard_normal(m) * shadow_db * math.sqrt(1 - rho ** 2)
    s[0] = rng.standard_normal() * shadow_db
    for i in range(1, m):
        s[i] = rho * s[i - 1] + w[i]
    snr += np.interp(t * speed_mps / step_m, np.arange(m), s)
    # dips
    k = rng.poisson(dips_per_min * 2 * t_leg / 60)
    for c in rng.uniform(0, 2 * t_leg, k):
        x = (t - c) / dip_s
        win = np.where(np.abs(x) < 0.5, 1.0, np.where(np.abs(x) < 1.0, 0.5 + 0.5 * np.cos(2 * math.pi * (np.abs(x) - 0.5)), 0.0))
        snr -= dip_db * win
    return snr.astype(np.float32), d.astype(np.float32)


# ---- the simulation -----------------------------------------------------------------------------

class Frame:
    __slots__ = ("idx", "kind", "ref", "size", "ready", "deadline", "blocks", "anchor", "complete", "state",
                 "dup", "decoded_t")

    def __init__(self, idx, kind, ref, size, ready, deadline, anchor, dup):
        self.idx, self.kind, self.ref, self.size = idx, kind, ref, size
        self.ready, self.deadline, self.anchor, self.dup = ready, deadline, anchor, dup
        self.blocks, self.complete, self.state, self.decoded_t = [], None, None, None


class Block:
    __slots__ = ("bid", "frame", "k", "ec", "size", "got", "n_got", "asked", "extra", "first_rx")

    def __init__(self, bid, frame, k, ec, size):
        self.bid, self.frame, self.k, self.ec, self.size = bid, frame, k, ec, size
        self.got, self.n_got, self.asked, self.extra, self.first_rx = set(), 0, {}, 0, False


def split_blocks(nbytes: int, payload: int, d: int, ec: int, small_frame_bytes: int):
    """nestua-zig's generateBlocksFromData: [(data, parity), ...] for one frame."""
    total = max(1, -(-nbytes // payload))
    ec_blk = 0 if (small_frame_bytes and nbytes <= small_frame_bytes) else ec
    out = []
    left = total
    while left > 0:
        k, e = d, ec_blk
        if d < left < d + 5 and left < 16:            # merged tail
            k = left
            e = 0 if ec_blk == 0 else min(16, (k * ec_blk) // d)
        if left < k:                                     # short tail: proportional parity
            if left > 1:
                e2 = math.ceil(left * (ec_blk / d) + 0.001) if ec_blk else 0
                if e2 == 0 and ec_blk > 0:
                    e2 = 1
                e = min(e2, left)
            else:
                e = 0
            k = left
        out.append((k, e))
        left -= k
    return out


class LinkSim:
    """One run: trace thresholds thr [mcs, N] (dB), per-slot gain_db [N],
    the mean Es/N0 per slot (scalar or array over slots), a policy."""

    def __init__(self, thr: np.ndarray, gain_db: np.ndarray, slot_s: float, cfg: LinkConfig, policy,
                 snr_db, seed: int = 0, mcs_names=MCS_NAMES):
        self.thr, self.gain, self.slot_s, self.cfg, self.policy = thr, gain_db, slot_s, cfg, policy
        self.n_trace = thr.shape[1]
        self.snr = snr_db
        self.rng = np.random.default_rng(seed)
        self.mcs_names = mcs_names
        assert tuple(mcs_names) == MCS_NAMES

    def snr_at(self, n):
        s = self.snr
        return float(s) if np.isscalar(s) or np.ndim(s) == 0 else float(s[n])

    # ---- the encoder
    def _new_frame(self, j, t):
        c = self.cfg
        bitrate = self.bitrate
        n_gop = max(1, int(round(c.gop_s * c.fps)))
        base = bitrate / c.fps / 8 * n_gop / (n_gop - 1 + c.idr_factor)
        spread = math.exp(self.rng.normal(0, c.size_sigma) - c.size_sigma ** 2 / 2)
        kind, ref, dup = "P", j - 1, 1
        size = base * spread
        if j == 0 or j % n_gop == 0 or (self.need_resync and self._resync_ok(t) == "I"):
            kind, ref = "I", None
            size = base * (c.idr_factor if (j % n_gop == 0 and not self.need_resync) else c.idr_recovery_factor) * spread
            if self.need_resync:
                self.last_idr_resync, self.last_resync = t, t
                dup = c.resync_dup
            self.need_resync = False
            self.anchors = []
        elif self.need_resync and self._resync_ok(t) == "R":
            a = self.confirmed
            gap = (j - a) / c.fps
            kind, ref, dup = "R", a, c.resync_dup
            size = base * float(np.interp(gap, LTR_GAP_S, LTR_FACTOR)) * spread
            self.need_resync = False
            self.last_resync = t
        anchor = c.ltr_frames > 0 and j % c.ltr_frames == 0
        if anchor:
            self.anchors = (self.anchors + [j])[-2:]
            if self.confirmed is not None and self.confirmed not in self.anchors:
                self.confirmed = None
        if kind == "I":
            self.confirmed = None
            if anchor:
                self.anchors = [j]
        if c.frame_cap:
            cap = c.frame_cap * c.latency_ms / 1e3 * c.goodput_bps(MCS_NAMES[self.mcs_i], self.slot_s) / 8 / dup
            size = min(size, cap)
        f = Frame(j, kind, ref, max(1, int(size)), t, t + c.latency_ms / 1e3, anchor, dup)
        self.stats["bytes_encoded"] += f.size
        self.stats[f"frames_{kind}"] += 1
        return f

    def _resync_ok(self, t):
        """What the encoder may answer a resync request with now: "R", "I" or None."""
        c = self.cfg
        if t - self.last_resync < c.ltr_cooldown_ms / 1e3:
            return None
        if (self.confirmed is not None and self.confirmed in self.anchors
                and (self.j - self.confirmed) / c.fps <= c.ltr_max_age_s):
            return "R"
        if t - self.last_idr_resync >= c.idr_min_gap_ms / 1e3:
            return "I"
        return None

    def _packetise(self, f):
        c = self.cfg
        mcs = MCS_NAMES[self.mcs_i]
        pay = c.payload(mcs)
        d, e = c.fec_of(mcs)
        for k, ec in split_blocks(f.size, pay, d, e, c.small_frame_bytes):
            b = Block(self.next_bid, f, k, ec, pay)
            self.next_bid += 1
            f.blocks.append(b)
            for _ in range(f.dup):
                for p in range(k + ec):
                    self.txq.append((b, p))
            self.blocks.append(b)

    # ---- one run
    def run(self, seconds: float, warmup_s: float = 0.0) -> dict:
        """Simulate `seconds`; the statistics leave out the first warmup_s
        (the policy's start from its first MCS)."""
        c = self.cfg
        n_slots = int(round(seconds / self.slot_s))
        warm_n = int(round(warmup_s / self.slot_s))
        self.stats0, disp0, j0 = None, 0, 0
        self.mcs_i = self.policy.start(self)
        self.bitrate = self._rate()
        self.stats = {k: 0 for k in ("bytes_encoded", "frames_I", "frames_P", "frames_R", "slots_tx", "slots_ok",
                                     "retx_pkts", "fill_pkts", "ul_ok", "ul_slots", "mcs_changes",
                                     "dropped_expired", "blocks_failed", "resync_requests", "lost_late", "lost_ref")}
        self.txq, self.rtxq, self.rtx_set = deque(), deque(), set()
        self.blocks = deque()                 # blocks the ground may still need, oldest first
        self.frames = deque()                 # frames not yet decided at the ground
        self.next_bid, self.j = 0, 0
        self.anchors, self.confirmed = [], None
        self.need_resync, self.last_resync, self.last_idr_resync = False, -1e9, -1e9
        self.ground_waiting = False           # ground: frozen, asking for a resync
        self.ground_anchor = None             # ground: newest anchor decoded
        self.max_bid_seen, self.last_rx_t = -1, -1e9
        self.last_dec_idx = -1
        self.decoded = {}                     # frame idx -> decoded (ground's reference state)
        self.display = []                     # decode times of displayed frames
        self.latency = []
        self.mcs_time = np.zeros(len(MCS_NAMES))
        self.rate_time = 0.0
        rep = self._zero_rep()
        next_frame_t = 0.0
        frame_period = 1 / c.fps
        lost_frames = 0
        current = None                         # (block, packet, fragments left, ok so far)
        trace_mcs = self.thr
        for n in range(n_slots):
            t = n * self.slot_s
            self.t = t
            tn = n % self.n_trace
            if n == warm_n and warm_n:
                self.stats0 = dict(self.stats)
                lost_frames, disp0, j0 = 0, len(self.display), self.j
                self.latency = []
                self.mcs_time[:] = 0
                self.rate_time = 0.0
            # the encoder
            while next_frame_t <= t:
                f = self._new_frame(self.j, next_frame_t)
                self.j += 1
                self._packetise(f)
                self.frames.append(f)
                next_frame_t += frame_period
            snr = self.snr_at(n)
            if n % c.sf_slots == c.sf_slots - 1:
                # ---- uplink slot
                self.stats["ul_slots"] += 1
                ok = snr + c.ul_offset_db >= trace_mcs[0, tn]
                nack = self._nack(t) if c.retransmit else []
                rep["t"] = t
                rep["anchor"] = self.ground_anchor
                rep["resync"] = self.ground_waiting
                if ok:
                    self.stats["ul_ok"] += 1
                    self._air_receive(rep, nack, t)
                    new = self.policy.report(self, rep)
                    rep = self._zero_rep()
                else:
                    new = self.policy.missed(self)
                if new is not None and new != self.mcs_i:
                    self.mcs_i = new
                    self.bitrate = self._rate()
                    self.stats["mcs_changes"] += 1
            else:
                # ---- downlink slot
                if current is None:
                    current = self._next_packet(t)
                if current is not None:
                    b, p, frags, okacc = current
                    m = self.mcs_i
                    ok = snr >= trace_mcs[m, tn]
                    self.stats["slots_tx"] += 1
                    self.stats["slots_ok"] += int(ok)
                    rep["sent"] += 1
                    rep["lost"] += int(not ok)
                    est = snr + float(self.gain[tn])
                    lin = 10 ** (est / 10)
                    rep["snr_sum"] += est + self.rng.normal(0, c.snr_est_db * (1 + 1 / max(lin, 1e-3)))
                    rep["snr_n"] += 1
                    okacc = okacc and ok
                    frags -= 1
                    if frags > 0:
                        current = (b, p, frags, okacc)
                    else:
                        current = None
                        if okacc:
                            self._ground_rx(b, p, t + self.slot_s)
                self.mcs_time[self.mcs_i] += 1
            self.rate_time += self.bitrate
            lost_frames += self._ground_decode(t + self.slot_s, rep)
        if self.stats0 is not None:
            self.stats = {k: v - self.stats0[k] for k, v in self.stats.items()}
        return self._summary(warm_n * self.slot_s, n_slots * self.slot_s, lost_frames, disp0, j0, n_slots - warm_n)

    def _rate(self):
        c = self.cfg
        return min(c.bitrate_fraction * c.goodput_bps(MCS_NAMES[self.mcs_i], self.slot_s), c.max_bitrate_bps)

    @staticmethod
    def _zero_rep():
        return {"t": 0.0, "sent": 0, "lost": 0, "snr_sum": 0.0, "snr_n": 0, "blocks_failed": 0, "frames_lost": 0,
                "snr_db": None, "anchor": None, "resync": False}

    def _next_packet(self, t):
        c = self.cfg
        pay = c.payload(MCS_NAMES[self.mcs_i])
        for q, retx in ((self.rtxq, True), (self.txq, False)):
            while q:
                b, p = q.popleft()
                if retx:
                    self.rtx_set.discard((b.bid, p))
                if b.frame.deadline <= t or b.frame.state is not None:
                    self.stats["dropped_expired"] += 1
                    continue
                self.stats["retx_pkts"] += int(retx)
                return (b, p, max(1, -(-b.size // pay)), True)
        if c.fill_parity:
            # idle: one more parity packet for the newest blocks still open
            for b in reversed(self.blocks):
                f = b.frame
                if f.deadline <= t or f.state is not None:
                    break
                if b.extra < c.fill_parity and b.k + b.ec + b.extra < 32:
                    b.extra += 1
                    self.stats["fill_pkts"] += 1
                    return (b, b.k + b.ec + b.extra - 1, max(1, -(-b.size // pay)), True)
        return None

    def _ground_rx(self, b, p, t):
        self.last_rx_t = t
        if b.bid > self.max_bid_seen:
            self.max_bid_seen = b.bid
        b.first_rx = True
        if p not in b.got:
            b.got.add(p)
            b.n_got += 1
            f = b.frame
            if b.n_got == b.k and f.complete is None and all(x.n_got >= x.k for x in f.blocks):
                f.complete = t

    def _nack(self, t):
        """The ground's retransmission request: [(block, [packets])]."""
        c = self.cfg
        out, n = [], 0
        for b in self.blocks:
            f = b.frame
            if f.state is not None or b.n_got >= b.k:
                continue
            left = f.deadline - t
            stop = min(c.latency_ms / 2, c.stop_ms) if c.stop_scaled else c.stop_ms
            if left < stop / 1e3:
                continue
            known = b.bid < self.max_bid_seen or (b.bid == self.max_bid_seen and t - self.last_rx_t >= c.idle_ms / 1e3)
            if not known:
                break                          # newer blocks are unknown or still arriving
            if not b.first_rx:
                pk = [p for p in range(b.k) if t - b.asked.get(p, -1e9) >= c.rerequest_ms / 1e3]
                if not pk:
                    continue
                for p in pk:
                    b.asked[p] = t
                out.append((b, pk))
                n += 1
            else:
                need = b.k - b.n_got
                pk = []
                for p in range(b.k):
                    if len(pk) >= need:
                        break
                    if p not in b.got and t - b.asked.get(p, -1e9) >= c.rerequest_ms / 1e3:
                        pk.append(p)
                if not pk:
                    continue
                for p in pk:
                    b.asked[p] = t
                out.append((b, pk))
                n += len(pk)
            if n >= c.max_entries:
                break
        return out

    def _air_receive(self, rep, nack, t):
        """An uplink report that reached the air."""
        for b, pk in nack:
            for p in pk:
                if (b.bid, p) not in self.rtx_set:          # already queued: once is enough
                    self.rtx_set.add((b.bid, p))
                    self.rtxq.append((b, p))
        if rep["anchor"] is not None and rep["anchor"] in self.anchors:
            if self.confirmed is None or rep["anchor"] > self.confirmed:
                self.confirmed = rep["anchor"]
        if rep["resync"]:
            self.stats["resync_requests"] += 1
            self.need_resync = True
        rep["snr_db"] = rep["snr_sum"] / rep["snr_n"] if rep["snr_n"] else None

    def _ground_decode(self, t, rep):
        """Decide frames in order: decoded when complete and their reference
        decoded, lost at the deadline. Returns frames lost now."""
        lost = 0
        while self.frames:
            f = self.frames[0]
            if f.complete is not None and f.complete <= t:
                ok = f.ref is None or self.decoded.get(f.ref, False)
                if f.kind == "P" and self.ground_waiting:
                    ok = False
            elif f.deadline <= t:
                ok = False
                for b in f.blocks:
                    if b.n_got < b.k:
                        self.stats["blocks_failed"] += 1
                        rep["blocks_failed"] += 1
            else:
                break
            self.frames.popleft()
            f.state = ok
            self.decoded[f.idx] = ok
            self.decoded.pop(f.idx - 400, None)
            if ok:
                f.decoded_t = max(t, self.display[-1] if self.display else t)
                self.display.append(f.decoded_t)
                self.latency.append(f.decoded_t - f.ready)
                if f.kind in ("I", "R"):
                    self.ground_waiting = False
                if f.anchor:
                    self.ground_anchor = f.idx
                if f.kind == "I":
                    self.ground_anchor = f.idx if f.anchor else None
            else:
                lost += 1
                rep["frames_lost"] += 1
                self.stats["lost_late" if f.complete is None or f.complete > t else "lost_ref"] += 1
                self.ground_waiting = True
            while self.blocks and self.blocks[0].frame.state is not None:
                self.blocks.popleft()
        return lost

    def _summary(self, t0, t1, lost_frames, disp0, j0, slots):
        c = self.cfg
        d = np.asarray(self.display[disp0:])
        dur = t1 - t0
        nominal = 1 / c.fps
        gaps = np.diff(np.concatenate([[t0], d, [t1]]))
        frz = gaps[gaps > c.freeze_ms / 1e3]
        stale = np.clip(gaps - nominal, 0, None).sum()
        per_min = 60.0 / dur
        lat = np.asarray(self.latency) if self.latency else np.zeros(1)
        s = self.stats
        total = len(MCS_NAMES)
        return {
            "seconds": round(dur, 3),
            "freeze_s_per_min": float((frz - nominal).sum() * per_min),
            "freezes_per_min": float(len(frz) * per_min),
            "longest_freeze_s": float(frz.max()) if len(frz) else 0.0,
            "stale_s_per_min": float(stale * per_min),
            "frames_lost_per_min": float(lost_frames * per_min),
            "frames_shown": int(len(d)),
            "mean_bitrate_bps": float(self.rate_time / max(slots, 1)),
            "delivered_bps": float(s["bytes_encoded"] * 8 / dur * (len(d) / max(self.j - j0, 1))),
            "latency_p50_ms": float(np.percentile(lat, 50) * 1e3),
            "latency_p99_ms": float(np.percentile(lat, 99) * 1e3),
            "mcs_share": {m: float(self.mcs_time[i] / max(self.mcs_time.sum(), 1)) for i, m in enumerate(MCS_NAMES)},
            "slot_use": float(s["slots_tx"] / max(self.mcs_time.sum(), 1)),
            "pre_fec_loss": float(1 - s["slots_ok"] / max(s["slots_tx"], 1)),
            "retx_share": float(s["retx_pkts"] / max(s["slots_tx"], 1)),
            "ul_success": float(s["ul_ok"] / max(s["ul_slots"], 1)),
            **{k: s[k] for k in ("frames_I", "frames_R", "mcs_changes", "blocks_failed", "dropped_expired",
                                 "resync_requests", "fill_pkts", "lost_late", "lost_ref")},
        }
