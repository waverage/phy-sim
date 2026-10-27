"""The board loopback (M8) in software: what rtl/loopback on the Arty Z7
computes, bit exact, at the speed of the Zig receiver.

On the board the ARM loads a set of transmitted frames (the fixed-point
TX of BOARD_FRAMES random frames per MCS) and the receiver's tables, then
the PL runs frames r = 0, 1, ...: TX frame r mod BOARD_FRAMES through the
channel emulator with noise seed seed0 + r (harness/chanemu.py), the RTL
receiver, and a comparison with the frame's information bits. EmuLink
does the same with the Zig fixed-point receiver, which is bit exact with
the RTL (rtl/rx, cocotb), so for the same seed0 and frame count the board's
error counters must equal EmuLink's.
"""
from __future__ import annotations

import numpy as np

from . import chanemu, rtlcfg, twin
from .design import EstimatorDesign
from .fxmodem import FixedPoint, FxOfdmLink, track_gain_q15
from .waveform import MCS, Numerology

BOARD_FRAMES = 8
INFO_SEED = 18
RUN_SEED = 0x5EED0000


def point_seed(esno_db: float) -> int:
    """seed0 of an SNR point: frames of different points never share noise."""
    return (RUN_SEED + (int(round(esno_db * 100)) << 12)) & chanemu.M32


class EmuLink:
    """The loopback's frames, emulator and receiver. run(batch, esno_db)
    returns the counts of chain.count_errors; consecutive calls at one
    Es/N0 continue the frame sequence, a new Es/N0 restarts it at r = 0."""

    def __init__(self, num: Numerology, mcs: MCS, *, design: EstimatorDesign = EstimatorDesign(),
                 fp: FixedPoint = FixedPoint(), num_frames: int = BOARD_FRAMES, info_seed: int = INFO_SEED):
        self.link = FxOfdmLink(num, mcs, design=design, fp=fp, device="cpu", genie=False, channel="awgn")
        self.num, self.mcs, self.fp, self.plan = num, mcs, fp, self.link.plan
        p = self.plan
        rng = np.random.default_rng(info_seed)
        self.info = rng.integers(0, 2, (num_frames, p.num_codewords, p.k)).astype(np.uint8)
        self.tx = self.link.fixed.tx(self.info)                        # [F, samples, 2] int16
        x = self.tx.astype(np.float64) / fp.tx_scale
        self.signal_power = float((x ** 2).sum(-1).mean())
        self._esno, self.r = None, 0

    def setting(self, esno_db: float) -> dict:
        """The emulator gains and the LLR scale of an Es/N0: the registers
        the ARM writes for that point."""
        no = 10 ** (-esno_db / 10)
        gx, gn, g = chanemu.gains(no, self.signal_power, self.fp)
        scale = float(self.link.llr_scale(no, np.array([g]))[0])
        return {"esno_db": esno_db, "no": no, "gx": gx, "gn": gn, "agc_gain": g, "llr_scale": scale,
                "seed0": point_seed(esno_db)}

    def frames(self, esno_db: float, r0: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        """ADC samples [count, samples, 2] of frames r0 .. r0 + count - 1 and
        their information bits."""
        st = self.setting(esno_db)
        r = np.arange(r0, r0 + count)
        y = np.stack([chanemu.apply(self.tx[i % len(self.tx)], chanemu.seed_of(st["seed0"], int(i)), st["gx"],
                                    st["gn"], self.fp.adc_bits) for i in r])
        return y, self.info[r % len(self.info)]

    def run(self, batch: int, esno_db: float) -> dict:
        if esno_db != self._esno:
            self._esno, self.r = esno_db, 0
            self.link.set_noise(10 ** (-esno_db / 10))
        st = self.setting(esno_db)
        y, info = self.frames(esno_db, self.r, batch)
        self.r += batch
        self.link.fixed.decoder_stats(reset=True)
        bits = self.link.fixed.rx(y, np.full(batch, st["llr_scale"]), self.link.ref.l_min)
        iters, _ = self.link.fixed.decoder_stats(reset=True)
        err = bits != info
        cw = err.any(-1)
        return {"frames": batch, "frame_errors": int(cw.any(-1).sum()), "codewords": int(cw.size),
                "codeword_errors": int(cw.sum()), "bits": int(err.size), "bit_errors": int(err.sum()),
                "iterations": iters}


def twin_modem(link: FxOfdmLink) -> twin.TwinModem:
    """The NumPy twin of a link's fixed receiver, with its current filter
    (for rtlcfg.rx_cfg)."""
    fp, p = link.fp, link.plan
    mask = link.ref.rg.pilot_pattern.mask[0, 0].cpu().numpy()
    tw = twin.TwinModem(link.num, link.mcs.bits_per_symbol, k=p.k, n=p.n, num_codewords=p.num_codewords,
                        pilot_mask=mask, tx_scale=fp.tx_scale, dac_bits=fp.dac_bits, llr_bits=fp.llr_bits,
                        app_bits=fp.app_bits, msg_bits=fp.msg_bits, offset=fp.offset_lsb, num_iter=fp.num_iter,
                        phase_track=link.ref.phase_track == "linear",
                        track_gain_q15=track_gain_q15(link.design.track_gain))
    tw.set_filter(link.fixed.filt)
    return tw


def board_words(emu: EmuLink, esno_db: float) -> dict:
    """What the ARM writes for one SNR point of rtl/loopback, as word
    (address, data) lists: the receiver's configuration, the TX frames,
    the information bits and the run registers (NFRAMES excluded)."""
    emu.link.set_noise(10 ** (-esno_db / 10))
    st = emu.setting(esno_db)
    tw = twin_modem(emu.link)
    rx = [(LB_RX + a, d) for a, d in rtlcfg.rx_cfg(tw, emu.link.ref.l_min) + rtlcfg.rx_scale_cfg(st["llr_scale"])]
    tx = [(LB_TX + 4096 * f + i, ((int(x[0]) & 0xFFF) << 12) | (int(x[1]) & 0xFFF))
          for f in range(len(emu.tx)) for i, x in enumerate(emu.tx[f])]
    ref = []
    for f in range(len(emu.info)):
        bits = emu.info[f].reshape(-1).astype(np.uint64)
        bits = np.concatenate([bits, np.zeros((-len(bits)) % 32, np.uint64)]).reshape(-1, 32)
        words = (bits << np.arange(32, dtype=np.uint64)).sum(-1)
        ref += [(LB_REF + 512 * f + w, int(v)) for w, v in enumerate(words)]
    regs = [(3, st["seed0"]), (4, st["gx"]), (5, st["gn"]), (6, len(emu.tx)), (7, emu.tx.shape[1]),
            (8, emu.plan.k)]
    return {"rx": rx, "tx": tx, "ref": ref, "regs": regs, "setting": st}


LB_RX, LB_TX, LB_REF = 0x10000, 0x20000, 0x28000
LB_CTRL, LB_STATUS, LB_NFRAMES = 0, 1, 2
LB_COUNTERS = {"frames": 16, "frame_errors": 17, "codeword_errors": 18, "bit_errors": 19, "iterations": 20,
               "clocks_lo": 21, "clocks_hi": 22}


def frame_seconds(num: Numerology) -> float:
    """Air time of one frame."""
    return num.num_ofdm_symbols * (num.fft_size + num.cp_len) / num.bandwidth_hz
