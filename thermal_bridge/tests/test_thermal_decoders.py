from __future__ import annotations

import numpy as np
import pytest

from thermal_bridge import decoders as D


def _scene():
    t = np.full((192, 256), 20.0, np.float32)
    t[50:80, 100:120] = 35.5
    t[0, 0] = -10.0
    return t


def test_infiray_roundtrip_bottom_half():
    t = _scene()
    buf = D.encode_infiray(t)
    assert len(buf) == 256 * 384 * 2
    out = D.decode("infiray_p2", buf, 256, 192)
    assert out.dtype == np.float32 and out.shape == (192, 256)
    assert np.allclose(out, t, atol=1.0 / 64 + 1e-4)


def test_infiray_known_raw_value():
    # raw/64 - 273.15 : 300.0 K -> 26.85 C -> raw = 19200
    raw = np.full((192, 256), 19200, "<u2")
    top = np.zeros(256 * 192 * 2, np.uint8).tobytes()
    out = D.decode("tc001", top + raw.tobytes(), 256, 192)
    assert np.allclose(out, 26.85, atol=1e-4)


def test_infiray_accepts_opencv_shapes():
    t = _scene()
    buf = np.frombuffer(D.encode_infiray(t), np.uint8)
    for shaped in (buf.reshape(1, -1), buf.reshape(384, 256, 2)):
        assert np.allclose(D.decode("infiray_p2", shaped, 256, 192), t, atol=0.02)


def test_infiray_top_half_option():
    t = _scene()
    buf = D.encode_infiray(t, thermal_half="top")
    out = D.decode("infiray_p2", buf, 256, 192, {"thermal_half": "top"})
    assert np.allclose(out, t, atol=0.02)


def test_infiray_short_buffer_raises():
    with pytest.raises(ValueError):
        D.decode("infiray_p2", b"\x00" * 1000, 256, 192)


def test_raw_y16_linear():
    raw = np.arange(160 * 120, dtype="<u2").reshape(120, 160)
    out = D.decode("raw_y16", raw.tobytes(), 160, 120, {"scale": 0.01, "offset": -5.0})
    assert np.allclose(out, raw * 0.01 - 5.0, atol=1e-3)
    t = _scene()
    assert np.allclose(D.decode("raw_y16", D.encode_y16(t), 256, 192), t, atol=0.02)


def test_grey8_relative_from_yuyv_and_grey():
    y = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (192, 1))
    yuyv = np.stack([y, np.full_like(y, 128)], axis=-1).tobytes()
    out = D.decode("grey8_relative", yuyv, 256, 192, {"t_min": 10.0, "t_max": 40.0})
    assert out[0, 0] == pytest.approx(10.0) and out[0, -1] == pytest.approx(40.0)
    out2 = D.decode("grey8_relative", y, 256, 192, {"t_min": 10.0, "t_max": 40.0})
    assert np.allclose(out, out2)


def test_unknown_decoder():
    with pytest.raises(ValueError):
        D.get_decoder("nope")


def test_frame_stats_hotspot():
    t = _scene()
    t[10, 200] = 50.0
    s = D.frame_stats(t)
    assert s["max"] == pytest.approx(50.0)
    assert s["hotspot"]["x"] == 200 and s["hotspot"]["y"] == 10
    assert s["min"] == pytest.approx(-10.0)
