"""M7: the RTL's generated tables stay in step with the model. The blocks
themselves are tested bit exact against the golden vectors by cocotb
(make test-rtl)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import gen_rtl                                     # noqa: E402


def test_twiddle_rom_is_generated_from_the_twin():
    assert (ROOT / "rtl" / "fft256" / "fft256_twiddle.v").read_text() == gen_rtl.twiddle_rom(), \
        "rtl/fft256/fft256_twiddle.v is stale: make rtl-tables"


BLOCKS = ("fft256", "cordic", "stf_corr", "nco_rot", "ldpc_dec", "rx", "chan_emu", "loopback")


def test_every_block_has_a_cocotb_test():
    for block in BLOCKS:
        d = ROOT / "rtl" / block
        assert (d / "Makefile").exists() and list(d.glob("test_*.py")), block
    mk = (ROOT / "Makefile").read_text()
    assert all(b in mk.split("RTL_BLOCKS :=")[1].splitlines()[0].split() for b in BLOCKS)
