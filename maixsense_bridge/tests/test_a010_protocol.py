from __future__ import annotations

import struct

import numpy as np
import pytest

from maixsense_bridge import protocol as P


def _codes(seed=0):
    rng = np.random.default_rng(seed)
    c = rng.integers(1, 255, size=(100, 100), dtype=np.uint8)
    c[3, 3:5] = (0x00, 0xFF)  # header pattern inside pixel data
    return c


def test_at_commands():
    assert P.at("FPS", 15) == b"AT+FPS=15\r"
    assert P.at("UNIT", query=True) == b"AT+UNIT?\r"
    assert P.at("") == b"AT\r"
    seq = P.init_sequence(unit=0, fps=10)
    assert seq[0] == b"AT+ISP=0\r" and seq[-1] == b"AT+ISP=1\r"
    assert b"AT+DISP=2\r" in seq and b"AT+UNIT=0\r" in seq and b"AT+FPS=10\r" in seq
    with pytest.raises(ValueError):
        P.init_sequence(fps=25)


def test_packet_layout():
    c = _codes()
    pkt = P.encode_frame(c, frame_id=1234)
    assert pkt[:2] == b"\x00\xff" and pkt[-1] == 0xDD
    assert struct.unpack("<H", pkt[2:4])[0] == 16 + 10000
    assert len(pkt) == 20 + 10000 + 2
    assert pkt[14] == 100 and pkt[15] == 100
    assert struct.unpack("<H", pkt[16:18])[0] == 1234
    assert pkt[-2] == sum(pkt[:-2]) & 0xFF


def test_parse_single_frame():
    c = _codes()
    fr = P.FrameParser().feed(P.encode_frame(c, frame_id=7))
    assert len(fr) == 1
    assert fr[0].frame_id == 7 and fr[0].rows == 100 and fr[0].cols == 100
    assert np.array_equal(fr[0].depth, c)


def test_garbage_before_header_and_between():
    c = _codes()
    stream = b"OK\r\n\x00\x13\xff\xdd" + P.encode_frame(c, 1) + b"\x00\x00junk" + P.encode_frame(c, 2)
    p = P.FrameParser()
    fr = p.feed(stream)
    assert [f.frame_id for f in fr] == [1, 2]
    assert p.stats.checksum_errors == 0


@pytest.mark.parametrize("chunk", [1, 7, 333, 4096])
def test_split_chunks(chunk):
    p = P.FrameParser()
    out = []
    for ch in P.fake_serial_stream(5, unit=0, garbage=b"\xde\xad\x00", chunk=chunk):
        out += p.feed(ch)
    assert [f.frame_id for f in out] == [0, 1, 2, 3, 4]


def test_bad_checksum_rejected_and_resync():
    c = _codes()
    p = P.FrameParser()
    fr = p.feed(P.encode_frame(c, 1, corrupt_checksum=True) + P.encode_frame(c, 2))
    assert [f.frame_id for f in fr] == [2]
    assert p.stats.checksum_errors >= 1


def test_bad_tail_rejected():
    pkt = bytearray(P.encode_frame(_codes(), 3))
    pkt[-1] = 0xCC
    p = P.FrameParser()
    assert p.feed(bytes(pkt)) == []
    assert p.stats.tail_errors >= 1


def test_truncated_frame_waits():
    pkt = P.encode_frame(_codes(), 9)
    p = P.FrameParser()
    assert p.feed(pkt[:5000]) == []
    assert len(p.feed(pkt[5000:])) == 1


def test_binned_and_ir_frames():
    c = np.full((50, 50), 80, np.uint8)
    f = P.FrameParser().feed(P.encode_frame(c, 1))[0]
    assert f.depth.shape == (50, 50)
    ir = np.full((25, 25), 9, np.uint8)
    f = P.FrameParser().feed(P.encode_frame(np.ones((25, 25), np.uint8), 2, output_mode=1, ir=ir))[0]
    assert f.ir is not None and np.array_equal(f.ir, ir)


def test_depth_unit_conversion():
    codes = np.array([0, 51, 102, 255], np.uint8)
    # unit 0: (p/5.1)^2 mm
    assert np.allclose(P.depth_mm(codes, 0), [0.0, 100.0, 400.0, 2500.0], atol=1e-3)
    # unit N: p*N mm
    assert np.allclose(P.depth_mm(codes, 4), [0, 204, 408, 1020])
    m = P.depth_meters(codes, 0)
    assert m.dtype == np.float32
    assert m[0] == 0.0 and m[3] == 0.0           # 0 / 255 invalid
    assert m[2] == pytest.approx(0.4, abs=1e-6)
    assert np.all(P.depth_meters(codes, 0, min_m=0.2)[:2] == 0)
    mm = np.array([200.0, 1000.0, 2000.0])
    assert np.allclose(P.depth_mm(P.mm_to_code(mm, 0), 0), mm, rtol=0.03)


def test_intrinsics_and_points():
    K = P.intrinsics_from_fov(100, 100, 70.0, 60.0)
    fx, fy, cx, cy = K
    assert fx == pytest.approx(50 / np.tan(np.radians(35)))
    assert fy == pytest.approx(50 / np.tan(np.radians(30)))
    d = np.zeros((100, 100), np.float32)
    d[0, 0] = 1.0
    d[50, 99] = 2.0
    pts = P.depth_to_points(d, K)
    assert pts.shape == (2, 3)
    assert pts[0, 0] < 0 and pts[0, 1] < 0 and pts[0, 2] == 1.0   # top-left: x<0, y<0
    assert pts[1, 0] > 0 and pts[1, 2] == 2.0


def test_parse_coeff():
    txt = '+COEFF=1\r\nOK\r\n{"fx": 19660800, "fy": 19660800, "u0": 13107200, "v0": 13107200}\r\n'
    assert P.parse_coeff(txt) == pytest.approx((75.0, 75.0, 50.0, 50.0))
    assert P.parse_coeff("garbage") is None
