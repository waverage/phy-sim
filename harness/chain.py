"""Sionna link chains.

AwgnLink: bits -> 5G LDPC -> QAM -> AWGN per resource element -> demap -> decode.
No OFDM modulation; with a flat AWGN channel and perfect knowledge the per-RE
model is exact, and the numerology only sets how many REs a frame has.
OFDM, pilots, estimation and fading enter in M2 with a separate chain.
"""
from __future__ import annotations

import torch
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import BinarySource, Demapper, Mapper

from .waveform import MCS, Numerology, plan_codewords


class AwgnLink:
    def __init__(self, num: Numerology, mcs: MCS, device: str = "cpu", num_iter: int = 20):
        self.num, self.mcs, self.device = num, mcs, device
        self.plan = plan_codewords(num, mcs)
        # device= at construction, not .to() afterwards: .to() leaves the LDPC
        # index tables on Sionna's default device (cuda:0 when a GPU exists)
        self.source = BinarySource(device=device)
        self.encoder = LDPC5GEncoder(self.plan.k, self.plan.n, device=device)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_iter, hard_out=True, device=device)
        self.mapper = Mapper(mcs.constellation, mcs.bits_per_symbol, device=device)
        self.demapper = Demapper("app", mcs.constellation, mcs.bits_per_symbol, device=device)
        self.awgn = AWGN(device=device)

    def run(self, batch: int, esno_db: float) -> dict:
        """One batch of `batch` frames at Es/N0 = esno_db (per resource element)."""
        p = self.plan
        no = torch.tensor(10 ** (-esno_db / 10), device=self.device)
        b = self.source([batch * p.num_codewords, p.k])
        c = self.encoder(b)
        x = self.mapper(c)
        y = self.awgn(x, no)
        llr = self.demapper(y, no)
        b_hat = self.decoder(llr)
        err = (b_hat != b)
        cw_err = err.any(dim=-1).view(batch, p.num_codewords)
        return {
            "frames": batch,
            "frame_errors": int(cw_err.any(dim=-1).sum()),
            "codewords": batch * p.num_codewords,
            "codeword_errors": int(cw_err.sum()),
            "bits": int(err.numel()),
            "bit_errors": int(err.sum()),
        }
