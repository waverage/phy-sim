"""Sionna link chains.

AwgnLink: bits -> 5G LDPC -> QAM -> AWGN per resource element -> demap -> decode.
No OFDM modulation; with a flat AWGN channel and perfect knowledge the per-RE
model is exact, and the numerology only sets how many REs a frame has.

OfdmLink: the same code and mapping through a time-domain OFDM chain, with
pilots, channel estimation, an optional TDL fading channel and impairment
hooks (see harness/impairments.py).
"""
from __future__ import annotations

import numpy as np
import torch
from sionna.phy.channel import AWGN, ApplyTimeChannel, time_lag_discrete_time_channel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import BinarySource, Demapper, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (LMMSEEqualizer, LSChannelEstimator, OFDMDemodulator, OFDMModulator,
                             PilotPattern, RemoveNulledSubcarriers, ResourceGrid, ResourceGridMapper,
                             tdl_freq_cov_mat, tdl_time_cov_mat)

from .impairments import Impairment
from .waveform import MCS, Numerology, plan_codewords

SPEED_OF_LIGHT = 299_792_458.0


# One compiled variant per block type and input shape: 7 MCS x 3 blocks
# would exceed Dynamo's default of 8, after which it silently runs eagerly.
torch._dynamo.config.cache_size_limit = 64


class _Compiled:
    """torch.compile a Sionna block. The LDPC decoder runs 5 to 8x faster
    compiled (gather/scatter fusion), the demapper and equalizer 2 to 6x,
    with identical output. The first call per block type and input shape
    costs about 20 s of compilation; later instances with the same shapes
    reuse it. CUDA graphs (mode="reduce-overhead") fail inside Sionna's
    blocks. The first call runs eagerly: Sionna blocks build their state
    then and flip a flag that a compiled guard would otherwise trip on,
    forcing a second compilation."""

    def __init__(self, block):
        self.block = block
        self.compiled = None

    def __call__(self, *args, **kwargs):
        if self.compiled is None:
            out = self.block(*args, **kwargs)
            self.compiled = torch.compile(self.block)
            return out
        return self.compiled(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.block, name)


def _compiled(block, enabled: bool):
    return _Compiled(block) if enabled else block


class AwgnLink:
    def __init__(self, num: Numerology, mcs: MCS, device: str = "cpu", num_iter: int = 20,
                 compile: bool = True):
        self.num, self.mcs, self.device = num, mcs, device
        self.plan = plan_codewords(num, mcs)
        # device= at construction, not .to() afterwards: .to() leaves the LDPC
        # index tables on Sionna's default device (cuda:0 when a GPU exists)
        self.source = BinarySource(device=device)
        self.encoder = LDPC5GEncoder(self.plan.k, self.plan.n, device=device)
        self.decoder = _compiled(LDPC5GDecoder(self.encoder, num_iter=num_iter, hard_out=True, device=device), compile)
        self.mapper = Mapper(mcs.constellation, mcs.bits_per_symbol, device=device)
        self.demapper = _compiled(Demapper("app", mcs.constellation, mcs.bits_per_symbol, device=device), compile)
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


def resource_grid(num: Numerology, device: str = "cpu") -> ResourceGrid:
    """Sionna resource grid for a numerology: one stream, Kronecker (block)
    pilots on num.pilot_ofdm_symbols, which fill those symbols' subcarriers,
    plus num.continual_pilot_subcarriers in every symbol. Pilots are QPSK
    from a fixed seed, mapped in Sionna's order (symbol-major)."""
    if not num.continual_pilot_subcarriers:
        return ResourceGrid(num.num_ofdm_symbols, num.fft_size, num.subcarrier_spacing_hz,
                            cyclic_prefix_length=num.cp_len, num_guard_carriers=num.num_guard,
                            dc_null=num.dc_null, pilot_pattern="kronecker",
                            pilot_ofdm_symbol_indices=list(num.pilot_ofdm_symbols), device=device)
    mask = np.zeros([1, 1, num.num_ofdm_symbols, num.num_effective_subcarriers], np.float32)
    mask[0, 0, list(num.pilot_ofdm_symbols), :] = 1
    mask[0, 0, :, list(num.continual_pilot_subcarriers)] = 1
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (2, int(mask.sum())))
    pilots = ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])) / np.sqrt(2)
    pattern = PilotPattern(torch.as_tensor(mask, device=device),
                           torch.as_tensor(pilots[None, None].astype(np.complex64), device=device), device=device)
    return ResourceGrid(num.num_ofdm_symbols, num.fft_size, num.subcarrier_spacing_hz,
                        cyclic_prefix_length=num.cp_len, num_guard_carriers=num.num_guard,
                        dc_null=num.dc_null, pilot_pattern=pattern, device=device)


def symbol_average_ofdm_channel(h_t: torch.Tensor, rg: ResourceGrid, l_min: int) -> torch.Tensor:
    """Frequency response of each OFDM symbol from the discrete-time taps
    h_t [..., num_time_samples + l_tot - 1, l_tot], averaged over the useful
    part of the symbol. Sionna's time_to_ofdm_channel takes the taps at the
    first sample after the CP instead; with Doppler the channel moves within
    the symbol and that snapshot is a worse match to what the data sees than
    a genie LMMSE estimate (at 600 Hz, 0.9 dB on 64qam-5/6). The average is
    the ICI-free part of the channel, the right "perfect CSI".
    Output [..., num_ofdm_symbols, fft_size], same conventions as Sionna."""
    n, cp = rg.fft_size, rg.cyclic_prefix_length
    s = rg.num_ofdm_symbols
    h = h_t[..., : s * (n + cp), :].reshape(*h_t.shape[:-2], s, n + cp, h_t.shape[-1])
    h = h[..., cp:, :].mean(dim=-2)                                    # [..., S, l_tot]
    h = torch.nn.functional.pad(h, (0, n - h.shape[-1]))
    h = torch.roll(h, shifts=l_min, dims=-1)                           # lag 0 first, negative lags at the end
    return torch.fft.fftshift(torch.fft.fft(h), dim=-1)


class GenieLmmseEstimator:
    """Joint 2D LMMSE channel estimate from block pilots with known channel
    statistics, R = R_time (x) R_freq. The pilot pattern and the noise
    variance are the same for every frame of an SNR point, so the filter
    W = R_hp (R_pp + N0 I)^-1 and the error variance are computed once per
    N0 and applied as one matmul per frame. Sionna's LMMSEChannelEstimator
    gives a comparable estimate (separable, frequency then time) but rebuilds
    its filters per frame at about 65 frames/s on an RTX 3060 Ti.
    Any pilot pattern works; `use_symbols` restricts the estimate to the
    pilots on those OFDM symbols (the block pilots, for the phase tracker's
    first pass). Inputs and outputs follow LSChannelEstimator:
    y [b,1,1,S,fft] -> h_hat, err_var [b,1,1,1,1,S,F]."""

    def __init__(self, rg: ResourceGrid, cov_t: torch.Tensor, cov_f: torch.Tensor, device: str,
                 use_symbols: tuple[int, ...] | None = None):
        self.device = device
        self.remove_nulled = RemoveNulledSubcarriers(rg, device=device)
        S, F = rg.num_ofdm_symbols, rg.num_effective_subcarriers
        mask = rg.pilot_pattern.mask[0, 0].reshape(-1).bool()    # [S F], symbol-major like the pilots
        all_idx = torch.nonzero(mask).reshape(-1).to(device)
        pilots = rg.pilot_pattern.pilots[0, 0].to(torch.complex128).to(device)
        keep = torch.ones_like(all_idx, dtype=torch.bool)
        if use_symbols is not None:
            keep = torch.isin(all_idx // F, torch.as_tensor(use_symbols, device=device))
        pidx = all_idx[keep]
        self.pidx = pidx
        self.inv_pilots = (1 / pilots[keep]).to(torch.complex64).reshape(1, 1, 1, -1)
        r = torch.kron(cov_t.to(torch.complex128), cov_f.to(torch.complex128))   # [SF, SF], (s, f) order
        self.r_hp = r[:, pidx]                                    # [SF, Np]
        self.r_pp = r[pidx][:, pidx]                              # [Np, Np]
        self.r_diag = r.diagonal().real                           # [SF]
        self.S, self.F = S, F
        self._cache: tuple[float, torch.Tensor, torch.Tensor] | None = None

    def filters(self, no: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is None or self._cache[0] != no:
            a = self.r_pp + no * torch.eye(self.r_pp.shape[0], dtype=self.r_pp.dtype, device=self.device)
            w = torch.linalg.solve(a, self.r_hp.mH).mH          # [SF, Np], R_hp (R_pp + no I)^-1
            err_var = (self.r_diag - (w * self.r_hp.conj()).sum(-1).real).clamp_min(0)
            self._cache = (no, w.to(torch.complex64), err_var.to(torch.float32).reshape(self.S, self.F))
        return self._cache[1], self._cache[2]

    def __call__(self, y: torch.Tensor, no: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, err_var = self.filters(float(no))
        y_eff = self.remove_nulled(y)                             # [b,1,1,S,F]
        h_ls = y_eff.reshape(*y.shape[:3], -1)[..., self.pidx] * self.inv_pilots   # [b,1,1,Np]
        h_hat = (h_ls @ w.T).reshape(*y.shape[:3], 1, 1, self.S, self.F)
        return h_hat, err_var.expand(h_hat.shape)


class OfdmLink:
    """bits -> 5G LDPC -> frame bit interleaver -> QAM -> resource grid ->
    OFDM modulator -> tx_impairments -> channel -> channel_impairments ->
    AWGN -> rx_impairments -> OFDM demodulator -> channel estimate ->
    LMMSE equalizer -> demap -> deinterleave -> decode.

    estimator brackets what a product receiver can do:
      "ls"       per-RE least squares on the pilots, linear interpolation in
                 time (`interpolation`), no smoothing: the floor, about 3 dB
                 behind perfect CSI at low SNR.
      "lmmse"    2D LMMSE filtering with the TDL's true frequency and time
                 covariance (genie statistics): the practical ceiling.
      "perfect"  the true frequency response of each OFDM symbol, averaged
                 over the symbol (see symbol_average_ofdm_channel).

    Es/N0 is per resource element, as in AwgnLink: the modulator's IFFT is
    unitary and every RE has unit energy, so time-domain noise of variance
    N0 per sample gives N0 per subcarrier after the FFT, and with
    channel="awgn" and perfect CSI the two links must give the same curve.

    channel: "awgn", or "tdl" (3GPP TR 38.901 TDL, model tdl_model, scaled to
    delay_spread_s, all users at the speed giving doppler_hz at carrier_hz).
    The TDL power delay profiles sum to one, so the mean channel gain is 1
    and Es/N0 is an average over the fading; it is not normalised per frame.

    Impairment hooks take fn(x, fs) (see harness/impairments.py): tx before
    the channel, channel after it and before the noise (oscillator effects,
    interference), rx after the noise (IQ imbalance, ADC).

    phase_track: common phase error correction (see derotate), lmmse only.
    "symbol" measures each symbol's phase on its own pilots (802.11 style;
    needs continual pilot subcarriers, noisy with few of them at low SNR).
    "linear" fits one phase ramp per frame (residual CFO) to all pilots; with
    block pilots alone it needs no extra overhead and is unambiguous up to
    1 / (2 x pilot symbol spacing), 1.9 kHz for symbols 2 and 11.
    """

    def __init__(self, num: Numerology, mcs: MCS, *, channel: str = "awgn", tdl_model: str = "D",
                 delay_spread_s: float = 100e-9, doppler_hz: float = 0.0, carrier_hz: float = 2.4e9,
                 estimator: str = "ls", interpolation: str = "lin", interleave: bool = True,
                 tx_impairments: Impairment | None = None, channel_impairments: Impairment | None = None,
                 rx_impairments: Impairment | None = None, device: str = "cpu", num_iter: int = 20,
                 compile: bool = True, phase_track: str | None = None):
        if channel not in ("awgn", "tdl"):
            raise ValueError(f"channel must be 'awgn' or 'tdl', not {channel!r}")
        if estimator not in ("ls", "lmmse", "perfect"):
            raise ValueError(f"estimator must be 'ls', 'lmmse' or 'perfect', not {estimator!r}")
        if phase_track not in (None, "symbol", "linear"):
            raise ValueError(f"phase_track must be None, 'symbol' or 'linear', not {phase_track!r}")
        if phase_track and estimator != "lmmse":
            raise ValueError("phase_track needs estimator='lmmse'")
        if phase_track == "symbol" and not num.continual_pilot_subcarriers:
            raise ValueError("phase_track='symbol' needs continual pilot subcarriers")
        self.phase_track = phase_track
        self.num, self.mcs, self.device, self.channel = num, mcs, device, channel
        self.estimator_kind = estimator
        self.tx_imp, self.ch_imp, self.rx_imp = tx_impairments, channel_impairments, rx_impairments
        self.fs = num.bandwidth_hz
        self.plan = plan_codewords(num, mcs)
        self.rg = resource_grid(num, device)
        assert self.rg.num_data_symbols == num.num_data_res, "resource grid and numerology disagree"

        # device= at construction, see AwgnLink
        self.source = BinarySource(device=device)
        self.encoder = LDPC5GEncoder(self.plan.k, self.plan.n, device=device)
        self.decoder = _compiled(LDPC5GDecoder(self.encoder, num_iter=num_iter, hard_out=True, device=device), compile)
        self.mapper = Mapper(mcs.constellation, mcs.bits_per_symbol, device=device)
        self.demapper = _compiled(Demapper("app", mcs.constellation, mcs.bits_per_symbol, device=device), compile)
        self.awgn = AWGN(device=device)
        self.rg_mapper = ResourceGridMapper(self.rg, device=device)
        self.modulator = OFDMModulator(num.cp_len, device=device)

        speed = doppler_hz * SPEED_OF_LIGHT / carrier_hz
        if channel == "tdl":
            self.tdl = TDL(tdl_model, delay_spread=delay_spread_s, carrier_frequency=carrier_hz,
                           min_speed=speed, max_speed=speed, device=device)
            max_delay = float(self.tdl.delays.max())
            self.l_min, self.l_max = time_lag_discrete_time_channel(self.fs, max(max_delay, 1 / self.fs))
            self.l_tot = self.l_max - self.l_min + 1
            self.apply_channel = ApplyTimeChannel(self.rg.num_time_samples, self.l_tot, device=device)
            # Path gains are generated at fs / tdl_decim and interpolated
            # linearly: at least 200 samples per Doppler cycle, so the
            # interpolation error is below 1e-4 while the generator's memory
            # (20 sinusoids x paths x samples x batch) drops by tdl_decim.
            self.tdl_decim = 64 if doppler_hz == 0 else int(min(64, max(1, self.fs / (200 * doppler_hz))))
        else:
            self.l_min = self.l_max = 0
            self.l_tot = 1

        self.demodulator = OFDMDemodulator(num.fft_size, self.l_min, num.cp_len, device=device)
        self.remove_nulled = RemoveNulledSubcarriers(self.rg, device=device)
        if estimator == "lmmse":
            # statistics of the configured TDL; on an AWGN channel they are a
            # mismatched but harmless prior
            cov_t = tdl_time_cov_mat(tdl_model, speed, carrier_hz, num.symbol_duration_s,
                                     num.num_ofdm_symbols, device=device)
            cov_f = tdl_freq_cov_mat(tdl_model, num.subcarrier_spacing_hz, num.fft_size,
                                     delay_spread_s, device=device)
            eff = torch.as_tensor(self.rg.effective_subcarrier_ind, device=cov_f.device)
            cov_f = cov_f[eff][:, eff]
            self.estimator = GenieLmmseEstimator(self.rg, cov_t, cov_f, device)
            if phase_track:
                self.block_estimator = GenieLmmseEstimator(self.rg, cov_t, cov_f, device,
                                                           use_symbols=num.pilot_ofdm_symbols)
                # every pilot, block and continual, at its place in the grid
                mask = self.rg.pilot_pattern.mask[0, 0].reshape(-1).bool()
                x_p = torch.zeros(mask.numel(), dtype=torch.complex64, device=device)
                x_p[torch.nonzero(mask).reshape(-1).to(device)] = self.rg.pilot_pattern.pilots[0, 0].to(device)
                self.track_pilots = x_p.reshape(num.num_ofdm_symbols, num.num_effective_subcarriers)
                self.track_anchor = float(np.mean(num.pilot_ofdm_symbols))
        else:
            self.estimator = LSChannelEstimator(self.rg, interpolation_type=interpolation, device=device)
        self.equalizer = _compiled(
            LMMSEEqualizer(self.rg, StreamManagement(np.ones([1, 1], int), 1), device=device), compile)

        # Fixed pseudo-random bit interleaver over the whole frame, so every
        # codeword sees all subcarriers and symbols and every bit position.
        frame_bits = num.num_data_res * mcs.bits_per_symbol
        perm = (torch.randperm(frame_bits, generator=torch.Generator().manual_seed(0))
                if interleave else torch.arange(frame_bits))
        self.perm = perm.to(device)
        self.inv_perm = torch.argsort(self.perm)

    def time_channel(self, batch: int) -> torch.Tensor:
        """Discrete-time TDL taps [batch, 1, 1, 1, 1, time, l_tot],
        h[t, l] = sum_p a_p(t) sinc(l - tau_p fs). Sionna's GenerateTimeChannel
        computes the same sum through a [paths, time, lags] tensor, about 13x
        the memory, which caps the batch on an 8 GB GPU."""
        n = self.rg.num_time_samples + self.l_tot - 1
        d = self.tdl_decim
        a_c, tau = self.tdl(batch, n // d + 2, self.fs / d)   # a [b,rx,rxa,tx,txa,p,t], tau [b,rx,tx,p]
        t = torch.arange(n, device=a_c.device) / d
        i0 = t.floor().long()
        w = (t - i0).to(a_c.dtype)
        a = a_c[..., i0] * (1 - w) + a_c[..., i0 + 1] * w
        lags = torch.arange(self.l_min, self.l_max + 1, device=tau.device, dtype=tau.dtype)
        s = torch.sinc(lags - tau[..., None] * self.fs)     # [b,rx,tx,p,l]
        s = s[:, :, None, :, None].to(a.dtype)              # [b,rx,1,tx,1,p,l]
        return torch.einsum("...pt,...pl->...tl", a, s)

    def derotate(self, y_f: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
        """Common phase error correction, as an 802.11 receiver does: estimate
        the channel from the block pilots alone, measure each symbol's phase
        on its pilots against it (z), turn that into a phase per symbol
        (phase_track) and rotate the symbol back. The full estimate (all
        pilots) then runs on the derotated grid."""
        h0, _ = self.block_estimator(y_f, no)                      # [b,1,1,1,1,S,F]
        y_eff = self.remove_nulled(y_f)                            # [b,1,1,S,F]
        z = (y_eff * (h0[:, :, :, 0, 0] * self.track_pilots).conj()).sum(-1)   # [b,1,1,S]
        if self.phase_track == "symbol":
            theta = torch.angle(z)
        else:
            # One phase ramp b (s - c) per frame. The block-pilot estimate h0
            # already carries the phase at c, the mean of the pilot symbols,
            # so only the slope is estimated: coarsely from neighbouring
            # symbols (to unwrap), then by least squares on the residual
            # phases. The phase error is common to the whole frame, so it
            # does not average out over codewords; fitting one parameter
            # from every continual pilot keeps it small.
            s = torch.arange(z.shape[-1], device=z.device, dtype=torch.float32) - self.track_anchor
            b0 = torch.angle((z[..., 1:] * z[..., :-1].conj()).sum(-1, keepdim=True))
            resid = torch.angle(z * torch.exp(-1j * b0 * s))
            w = z.abs()
            slope = b0 + (w * s * resid).sum(-1, keepdim=True) / (w * s * s).sum(-1, keepdim=True)
            theta = slope * s
        return y_f * torch.exp(-1j * theta)[..., None]

    def run(self, batch: int, esno_db: float) -> dict:
        """One batch of `batch` frames at Es/N0 = esno_db (per resource element)."""
        p, d = self.plan, self.device
        no = torch.tensor(10 ** (-esno_db / 10), device=d)
        b = self.source([batch, p.num_codewords, p.k])
        c = self.encoder(b).reshape(batch, -1)
        if p.padding_bits:
            c = torch.cat([c, torch.zeros(batch, p.padding_bits, dtype=c.dtype, device=d)], dim=-1)
        x = self.mapper(c[:, self.perm]).reshape(batch, 1, 1, -1)
        x_t = self.modulator(self.rg_mapper(x))             # [b, 1, 1, num_time_samples]
        if self.tx_imp:
            x_t = self.tx_imp(x_t, self.fs)
        if self.channel == "tdl":
            h_t = self.time_channel(batch)
            y = self.apply_channel(x_t, h_t)                # [b, 1, 1, num_time_samples + l_tot - 1]
        else:
            y = x_t
        if self.ch_imp:
            y = self.ch_imp(y, self.fs)
        y = self.awgn(y, no)
        if self.rx_imp:
            y = self.rx_imp(y, self.fs)
        y_f = self.demodulator(y)                           # [b, 1, 1, symbols, fft_size]

        if self.estimator_kind == "perfect":
            if self.channel == "tdl":
                h = self.remove_nulled(symbol_average_ofdm_channel(h_t, self.rg, self.l_min))
            else:
                h = torch.ones([batch, 1, 1, 1, 1, self.num.num_ofdm_symbols,
                                self.num.num_effective_subcarriers], dtype=y_f.dtype, device=d)
            err_var = torch.zeros(h.shape, dtype=no.dtype, device=d)
        else:
            if self.phase_track:
                y_f = self.derotate(y_f, no)
            h, err_var = self.estimator(y_f, no)
        x_hat, no_eff = self.equalizer(y_f, h, err_var, no)
        llr = self.demapper(x_hat, no_eff).reshape(batch, -1)[:, self.inv_perm]
        llr = llr[:, : p.num_codewords * p.n].reshape(batch, p.num_codewords, p.n)
        b_hat = self.decoder(llr)
        err = (b_hat != b)
        cw_err = err.any(dim=-1)
        return {
            "frames": batch,
            "frame_errors": int(cw_err.any(dim=-1).sum()),
            "codewords": batch * p.num_codewords,
            "codeword_errors": int(cw_err.sum()),
            "bits": int(err.numel()),
            "bit_errors": int(err.sum()),
        }
