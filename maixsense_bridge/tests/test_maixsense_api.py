from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from maixsense_bridge import maixsense_bridge as MB  # noqa: E402
from maixsense_bridge import protocol as P  # noqa: E402


@pytest.fixture()
def bridge():
    b = MB.MaixSenseBridge(mock=True)
    assert b.mock_step(t=b.t_start + 1.0) == 1
    assert b.mock_step(t=b.t_start + 1.1) == 1
    return b


def test_health(bridge):
    h = TestClient(MB.create_app(bridge)).get("/health").json()
    assert h["ok"] and h["frames"] == 2 and h["checksum_errors"] == 0
    assert h["fps"] == pytest.approx(10.0, rel=0.01)
    assert h["resolution"] == [100, 100]


def test_frame_npy(bridge):
    r = TestClient(MB.create_app(bridge)).get("/frame.npy")
    assert r.status_code == 200 and float(r.headers["X-Capture-Time"]) > 0
    d = np.load(io.BytesIO(r.content))
    assert d.shape == (100, 100) and d.dtype == np.float32
    valid = d[d > 0]
    assert valid.size > 5000 and 0.2 < valid.min() and valid.max() < 2.6


def test_png_and_points(bridge):
    c = TestClient(MB.create_app(bridge))
    r = c.get("/frame.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    r = c.get("/points.npy")
    pts = np.load(io.BytesIO(r.content))
    assert pts.ndim == 2 and pts.shape[1] == 3 and int(r.headers["X-Points"]) == len(pts)
    assert np.all(pts[:, 2] > 0)


def test_feed_fake_serial_stream_with_corruption():
    b = MB.MaixSenseBridge(mock=True)
    bad = P.encode_frame(np.full((100, 100), 100, np.uint8), frame_id=99, corrupt_checksum=True)
    for i, ch in enumerate(P.fake_serial_stream(3, garbage=b"AT+ISP=1\r\nOK\r\n" + bad, chunk=1000)):
        b.feed(ch, t=1000.0 + i)
    h = b.health()
    assert h["frames"] == 3 and h["checksum_errors"] >= 1 and h["last_frame_id"] == 2


def test_no_frame_503():
    c = TestClient(MB.create_app(MB.MaixSenseBridge(mock=True)))
    assert c.get("/frame.npy").status_code == 503
