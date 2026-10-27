"""The whole radio on the chip (M9), in software: what rtl/onchip computes,
bit exact.

The loopback of M8 fed the receiver aligned frames. Here the chip's own
transmitter (rtl/tx) sends frames r = 0, 1, ... behind their preambles,
each after `gap` samples of silence; the emulator turns the carrier by a
fixed frequency offset (an NCO over the whole stream, as a second radio's
oscillator would), scales, adds noise (segment r: the gap, the preamble and
the frame, seed seed0 + r, the counters restarting) and quantises to the
ADC; the synchroniser (rtl/sync) finds the frames in the stream and the
receiver (rtl/rx, phase tracker on: a little offset is left after the
synchroniser) decodes them.

A found frame belongs to TX frame r when its start is within MATCH samples
of r's true start (r (gap + pre + S slot) + gap + pre); TX frames passed
over are missed, a frame matching none is spurious. Missed frames count as
frame errors.
"""
from __future__ import annotations

import numpy as np

from . import chanemu, rtlcfg, twin
from .loopback import EmuLink, ref_words, twin_modem
from .sync import sync_config
from .waveform import MCS, Numerology

MATCH = 64
REARM = 64
OC_RX, OC_TX, OC_REF, OC_SYNC = 0x10000, 0x20000, 0x28000, 0x2C000
OC_CTRL, OC_STATUS, OC_NFRAMES = 0, 1, 2
OC_COUNTERS = {"frames": 16, "frame_errors": 17, "codeword_errors": 18, "bit_errors": 19, "iterations": 20,
               "clocks_lo": 21, "clocks_hi": 22, "missed": 23, "spurious": 24, "sync_overruns": 25,
               "sync_rejects": 26}


def cfo_inc_of(cfo_hz: float, fs_hz: float = 10e6) -> int:
    """The emulator's NCO increment for a carrier offset."""
    return int(round(cfo_hz / fs_hz * 2 ** 32))


class OnChipLink(EmuLink):
    """run(nframes, esno_db) -> counts, like EmuLink.run (frames from r = 0)."""

    def __init__(self, num: Numerology, mcs: MCS, *, cfo_inc: int = 0, gap: int = 128, **kw):
        super().__init__(num, mcs, phase_track="linear", **kw)
        self.cfo_inc, self.gap = cfo_inc, gap
        self.scfg = sync_config(num, tx_scale=self.fp.tx_scale, dac_bits=self.fp.dac_bits)
        self.ts = twin.TwinSync(self.scfg)
        self.pre = self.ts.preamble().astype(np.int64)
        self.seg = gap + len(self.pre) + self.tx.shape[1]

    def stream(self, esno_db: float, nframes: int) -> np.ndarray:
        """The ADC stream of frames 0 .. nframes - 1 (and a last gap)."""
        st = self.setting(esno_db)
        segs = []
        for r in range(nframes):
            x = np.concatenate([np.zeros((self.gap, 2), np.int64), self.pre, self.tx[r % len(self.tx)]])
            n = r * self.seg + np.arange(len(x))
            x = twin.rot16(x, twin.nco(n, -self.cfo_inc))       # the offset: +cfo_inc a sample
            segs.append(chanemu.apply(x, chanemu.seed_of(st["seed0"], r), st["gx"], st["gn"], self.fp.adc_bits))
        tail = np.zeros((self.seg, 2), np.int64)                 # silence after the last frame, its noise too
        segs.append(chanemu.apply(tail, chanemu.seed_of(st["seed0"], nframes), st["gx"], st["gn"], self.fp.adc_bits))
        return np.concatenate(segs).astype(np.int64)

    def run(self, nframes: int, esno_db: float) -> dict:
        self.link.set_noise(10 ** (-esno_db / 10))
        st = self.setting(esno_db)
        tw = twin_modem(self.link)
        found = self.ts.stream(self.stream(esno_db, nframes))
        true0 = self.gap + len(self.pre)
        out = dict(frames=nframes, synced=0, missed=0, spurious=0, frame_errors=0, codewords=0, codeword_errors=0,
                   bit_errors=0, iterations=0)
        r = 0
        for f in found:
            while r < nframes and f["start"] > r * self.seg + true0 + MATCH:
                r += 1
                out["missed"] += 1
            if r >= nframes or f["start"] < r * self.seg + true0 - MATCH:
                out["spurious"] += 1
                continue
            res = tw.rx(f["frame"], st["llr_scale"], 0)
            err = res["bits"] != self.info[r % len(self.info)]
            cw = err.any(-1)
            out["synced"] += 1
            out["frame_errors"] += int(cw.any())
            out["codewords"] += int(cw.size)
            out["codeword_errors"] += int(cw.sum())
            out["bit_errors"] += int(err.sum())
            out["iterations"] += int(res["iters"])
            r += 1
        out["missed"] += nframes - r
        out["frame_errors"] += out["missed"]
        return out

    def board_words(self, esno_db: float, pace: int = 0) -> dict:
        """What the ARM writes for one SNR point of rtl/onchip: (address,
        data) lists of the receiver, transmitter, synchroniser and the
        information bits, and the run registers (NFRAMES excluded)."""
        self.link.set_noise(10 ** (-esno_db / 10))
        st = self.setting(esno_db)
        tw = twin_modem(self.link)
        outw = rtlcfg.ldpc_out_words(tw.code)
        rx = [(OC_RX + a, d) for a, d in rtlcfg.rx_cfg(tw, 0) + rtlcfg.rx_scale_cfg(st["llr_scale"])]
        tx = [(OC_TX + a, d) for a, d in rtlcfg.tx_cfg(tw, self.pre)]
        sy = [(OC_SYNC + a, d) for a, d in rtlcfg.sync_cfg(self.scfg, REARM)]
        ref = [(OC_REF + 1024 * f + c * outw + w, int(v))
               for f in range(len(self.info)) for c, row in enumerate(ref_words(tw.code, self.info[f]))
               for w, v in enumerate(row)]
        regs = [(3, st["seed0"]), (4, st["gx"]), (5, st["gn"]), (6, len(self.info)), (7, len(self.pre) + self.tx.shape[1]),
                (8, self.plan.num_codewords), (9, outw), (10, pace), (11, self.cfo_inc & chanemu.M32), (12, self.gap)]
        return {"rx": rx, "tx": tx, "sync": sy, "ref": ref, "regs": regs, "setting": st}
