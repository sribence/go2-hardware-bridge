from __future__ import annotations

import io
import os

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from thermal_bridge import thermal_bridge as TB  # noqa: E402


@pytest.fixture()
def client():
    cfg = TB.load_config(TB.DEFAULT_CONFIG)
    bridge = TB.ThermalBridge(cfg, mock=True)
    for w in bridge.workers.values():
        assert w.step()
    return TestClient(TB.create_app(bridge)), bridge


def test_config_has_both_front_cams():
    cfg = TB.load_config(TB.DEFAULT_CONFIG)
    assert {"th_front_narrow", "th_front_wide"} <= set(cfg)
    assert cfg["th_front_narrow"]["hfov_deg"] < cfg["th_front_wide"]["hfov_deg"]


def test_cams_and_health(client):
    c, _ = client
    r = c.get("/cams")
    assert r.status_code == 200
    assert {x["id"] for x in r.json()} == {"th_front_narrow", "th_front_wide"}
    h = c.get("/health").json()
    assert h["ok"] and h["mock"]


def test_frame_npy(client):
    c, _ = client
    r = c.get("/cams/th_front_narrow/frame.npy")
    assert r.status_code == 200
    assert float(r.headers["X-Capture-Time"]) > 0
    a = np.load(io.BytesIO(r.content))
    assert a.dtype == np.float32 and a.shape == (192, 256)
    assert 15.0 < float(np.median(a)) < 25.0       # ~20 C room
    assert 33.5 < float(a.max()) < 36.5            # person


def test_png_and_stats(client):
    c, _ = client
    r = c.get("/cams/th_front_wide/frame.png")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert float(r.headers["X-Temp-Max"]) > float(r.headers["X-Temp-Min"])
    s = c.get("/cams/th_front_wide/stats").json()
    assert s["max"] > 33.0 and "hotspot" in s


def test_narrow_sees_bigger_person(client):
    c, _ = client
    n = np.load(io.BytesIO(c.get("/cams/th_front_narrow/frame.npy").content))
    w = np.load(io.BytesIO(c.get("/cams/th_front_wide/frame.npy").content))
    assert (n > 30).sum() > (w > 30).sum()


def test_unknown_cam_404(client):
    c, _ = client
    assert c.get("/cams/nope/frame.npy").status_code == 404


def test_real_mode_missing_device_does_not_crash():
    cfg = TB.load_config(TB.DEFAULT_CONFIG)
    cfg["th_front_wide"]["device"] = "/dev/v4l/by-id/does-not-exist"
    w = TB.CamWorker("th_front_wide", cfg["th_front_wide"], mock=False)
    assert w.step() is False and "not found" in w.last_error
    c = TestClient(TB.create_app(TB.ThermalBridge({"th_front_wide": cfg["th_front_wide"]})))
    assert c.get("/cams/th_front_wide/frame.npy").status_code == 503
