"""End-to-end against tools/a010_fake_device.py on a pty: AT handshake, COEFF, streaming,
binning, hot-unplug/replug through a stable symlink (like udev's /dev/maixsense)."""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pytest

pytest.importorskip("serial")
if not hasattr(os, "openpty"):
    pytest.skip("no pty support", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from a010_fake_device import FakeA010  # noqa: E402

from maixsense_bridge import maixsense_bridge as MB  # noqa: E402


def _wait(cond, timeout):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def rig(tmp_path):
    link = str(tmp_path / "maixsense")
    dev = FakeA010(link=link, fx=74.0, fy=73.0, u0=49.0, v0=50.0, garbage_every=5).start()
    made = []

    def make(**kw):
        b = MB.MaixSenseBridge(device=link, fps=19, reconnect_max_s=0.3, stall_s=2.0, **kw)
        b.start()
        made.append(b)
        return b

    yield dev, make
    for b in made:
        b.stop()
    dev.stop()


def test_pty_handshake_and_stream(rig):
    dev, make = rig
    b = make()
    assert _wait(lambda: b.health()["frames"] >= 20, 5.0), b.health()
    h = b.health()
    assert dev.commands[:9] == ["AT+ISP=0", "AT+DISP=1", "AT+ISP=1", "AT", "AT+COEFF?",
                                "AT+BINN=1", "AT+UNIT=0", "AT+FPS=19", "AT+DISP=2"]
    assert h["intrinsics_source"] == "AT+COEFF?"
    assert h["intrinsics"] == pytest.approx({"fx": 74.0, "fy": 73.0, "cx": 49.0, "cy": 50.0})
    assert h["checksum_errors"] == 0 and h["resolution"] == [100, 100]
    assert 12.0 < h["fps"] < 25.0


@pytest.mark.parametrize("binn,shape", [(2, (50, 50)), (4, (25, 25))])
def test_pty_binning_api(rig, binn, shape):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    dev, make = rig
    b = make(binn=binn)
    assert _wait(lambda: b.health()["frames"] >= 3, 5.0), b.health()
    assert dev.state["BINN"] == binn
    c = TestClient(MB.create_app(b))
    r = c.get("/frame.npy")
    assert r.headers["X-Resolution"] == "%dx%d" % shape
    assert np.load(io.BytesIO(r.content)).shape == shape
    h = c.get("/health").json()
    assert h["resolution"] == list(shape)
    s = shape[1] / 100.0
    assert h["intrinsics"]["fx"] == pytest.approx(74.0 * s)
    assert h["intrinsics"]["cx"] == pytest.approx((49.0 + 0.5) * s - 0.5)
    pts = np.load(io.BytesIO(c.get("/points.npy").content))
    assert pts.shape[1] == 3 and len(pts) > 0


def test_pty_hot_unplug_replug(rig):
    dev, make = rig
    b = make()
    assert _wait(lambda: b.health()["frames"] >= 5, 5.0)
    dev.unplug()
    assert _wait(lambda: not b.connected, 2.0)
    time.sleep(0.5)                       # device absent: open() fails, bridge keeps retrying
    assert not os.path.exists(dev.link)
    n0 = b.health()["frames"]
    dev.replug()                          # new pty, same symlink
    t0 = time.time()
    assert _wait(lambda: b.connected and b.health()["frames"] > n0 + 3, 6.0), b.health()
    h = b.health()
    assert h["reconnects"] >= 1 and h["ok"]
    assert time.time() - t0 < 6.0
    # firmware was power-cycled: the bridge re-sent the full handshake
    assert dev.commands.count("AT+DISP=2") >= 2


def test_pty_stall_recovers(rig):
    dev, make = rig
    b = make()
    assert _wait(lambda: b.health()["frames"] >= 3, 5.0)
    dev.state["ISP"] = 0                  # sensor silently stops streaming
    n0 = b.health()["frames"]
    assert _wait(lambda: b.health()["frames"] > n0 + 3, 8.0), b.health()  # stall -> reopen -> ISP=1
    assert b.health()["opens"] >= 2
