"""M0 gate: the Zig library loads through ctypes and its exports behave."""
import ctypes
import pathlib

import numpy as np
import pytest

LIB = pathlib.Path(__file__).resolve().parents[1] / "modem" / "zig-out" / "lib" / "libnestuaphy.dylib"


@pytest.fixture(scope="module")
def lib():
    if not LIB.exists():
        pytest.skip("build the modem first: make modem")
    l = ctypes.CDLL(str(LIB))
    l.nestuaphy_version.restype = ctypes.c_uint32
    l.nestuaphy_sat_add_q15.restype = ctypes.c_int16
    l.nestuaphy_sat_add_q15.argtypes = [ctypes.c_int16, ctypes.c_int16]
    l.nestuaphy_sum_q15.restype = ctypes.c_int32
    l.nestuaphy_sum_q15.argtypes = [ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t]
    return l


def test_version(lib):
    assert lib.nestuaphy_version() == (0 << 16) | 1


def test_sat_add(lib):
    assert lib.nestuaphy_sat_add_q15(30000, 10000) == 32767
    assert lib.nestuaphy_sat_add_q15(-30000, -10000) == -32768


def test_vector_sum_zero_copy(lib):
    rng = np.random.default_rng(0)
    x = rng.integers(-32768, 32767, size=100_003, dtype=np.int16)
    got = lib.nestuaphy_sum_q15(x.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), x.size)
    assert got == int(x.astype(np.int64).sum())
