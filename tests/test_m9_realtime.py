"""M9: the real-time receiver on the board (e020), and the Python side of
the RTL that M9 added: the decoder's word layout, the encoder program of
rtl/tx, the synchroniser on a stream (rtl/sync) and the whole chain's
model (rtl/onchip)."""
import dataclasses
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import modem as zm                               # noqa: E402
from harness import rtlcfg, twin                              # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords   # noqa: E402

MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, pilot_ofdm_symbols=(1, 6, 11))
CODES = [plan_codewords(num, m) for num in (NUM_10MHZ, P3) for m in MCS_TABLE]


def test_decoder_layout_round_trip():
    """Every LLR has its own (word, lane), inside the codeword's words; the
    output words hold exactly the k information bits."""
    for p in CODES:
        code = twin.ldpc_code(p.k, p.n)
        word, lane = rtlcfg.ldpc_llr_slot(code)
        slots = word * 16 + lane
        assert len(np.unique(slots)) == code.n
        assert word.min() >= 0 and word.max() < rtlcfg.ldpc_llr_words(code)
        assert p.num_codewords * rtlcfg.ldpc_llr_words(code) <= 1 << rtlcfg.LDPC_LLR_AW
        v = rtlcfg.ldpc_out_bits(code)
        assert sorted(v[v >= 0].tolist()) == list(range(code.k))


@pytest.mark.skipif(not zm.LIB.exists(), reason="build the modem first: make modem")
def test_encoder_program_equals_the_zig_encoder():
    rng = np.random.default_rng(5)
    for p in CODES:
        code = twin.ldpc_code(p.k, p.n)
        u = rng.integers(0, 2, p.k).astype(np.uint8)
        assert np.array_equal(rtlcfg.ldpc_encode_program(code, u), zm.ldpc_encode(u, p.n).astype(np.uint8))


def test_sync_stream_of_the_vectors():
    """The sync vectors' buffers back to back: one frame per buffer, the
    first exactly the vectors' (its base is the stream's start)."""
    sys.path.insert(0, str(ROOT / "rtl"))
    import types
    from tbutil import SYNC_FILES, load
    for name in SYNC_FILES:
        v = load(name)
        cfg = types.SimpleNamespace(**v["meta"]["sync_config"])
        # one more buffer behind: a frame is decided once its whole span is in
        r = np.concatenate([v["rx"][k] for k in (0, 1, 0, 1)]).astype(np.int64)
        out = twin.TwinSync(cfg).stream(r)
        assert len(out) >= 3, name
        assert np.array_equal(out[0]["frame"], v["frame"][0]), name
        assert out[0]["start"] == v["sync_start"][0]
        assert [o["start"] for o in out] == sorted(o["start"] for o in out)


@pytest.mark.skipif(not zm.LIB.exists(), reason="build the modem first: make modem")
def test_onchip_model_finds_and_decodes_every_frame():
    from harness.onchip import OnChipLink, cfo_inc_of
    link = OnChipLink(NUM_10MHZ, MCS["16qam-1/2"], cfo_inc=cfo_inc_of(-70e3), num_frames=2)
    r = link.run(5, 12.0)
    assert r["synced"] == 5 and r["missed"] == 0 and r["spurious"] == 0 and r["frame_errors"] == 0


def test_e020_real_time_on_the_board():
    """tools/board.py at 10 MS/s: every frame of every e018 point taken at
    the air's rate (none dropped, no sample lost), counters equal to the
    model, so the receiver is real time with e018's error rates."""
    path = ROOT / "results" / "e020_board_realtime.json"
    if not path.exists():
        pytest.skip("run tools/board.py all 10 with the Arty Z7-20 connected first")
    r = json.loads(path.read_text())
    assert r["pace_clocks_per_sample"] == 10 and r["fclk_mhz"] == 100.0
    assert {p["mcs"] for p in r["points"]} == set(MCS)
    assert all(p["match"] and p["no_drops"] for p in r["points"])
    assert r["dropped"] == 0 and r["lost"] == 0 and r["all_match"] and r["gate_met"]
    # long points: the air's rate (short ones include the pipeline's fill and drain)
    long = [p for p in r["points"] if p["nframes"] >= 20000]
    assert long and all(abs(p["board_frames_per_s"] - r["air_frames_per_s"]) < 2 for p in long)
