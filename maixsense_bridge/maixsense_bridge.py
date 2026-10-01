#!/usr/bin/env python3
"""maixsense_bridge -- Sipeed MaixSense A010 ToF (USB CDC serial) -> HTTP (FastAPI, :9121).

OmniVision 360 contract (go2-brain-logic/mission_control/omni/CONTRACTS.md, F; cam_id tof_rear):
  GET /frame.npy   (rows x cols, 100x100) float32 metres, 0 = invalid; header X-Capture-Time
  GET /frame.png   turbo-colorized depth
  GET /points.npy  (N,3) float32 points in OpenCV cam frame (z fwd, x right, y down)
  GET /health      fps, checksum errors, last frame age, intrinsics

Run:  python3 maixsense_bridge.py [--mock] [--device /dev/serial/by-id/...] [--port 9121]
Config via env: MAIXSENSE_DEVICE, MAIXSENSE_BAUD, MAIXSENSE_UNIT, MAIXSENSE_FPS,
MAIXSENSE_HFOV, MAIXSENSE_VFOV, MAIXSENSE_USE_COEFF, MAIXSENSE_MIN_M, MAIXSENSE_MAX_M.
Read-only sensor bridge: never sends anything to the robot.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # package import (pytest) vs. script run
    from . import protocol as P
except ImportError:  # pragma: no cover
    import protocol as P  # type: ignore

log = logging.getLogger("maixsense_bridge")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class MaixSenseBridge:
    def __init__(self, device: str = "/dev/ttyUSB0", baud: int = 115200, unit: int = 0, fps: int = 15,
                 hfov_deg: float = P.DEFAULT_HFOV_DEG, vfov_deg: float = P.DEFAULT_VFOV_DEG,
                 use_coeff: bool = True, min_m: float = 0.0, max_m: float = 10.0, mock: bool = False):
        self.device = device
        self.baud = baud
        self.unit = unit
        self.fps_cfg = fps
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.use_coeff = use_coeff
        self.min_m = min_m
        self.max_m = max_m
        self.mock = mock

        self.parser = P.FrameParser()
        self.lock = threading.Lock()
        self.depth_m: Optional[np.ndarray] = None
        self.last_frame: Optional[P.A010Frame] = None
        self.t_capture = 0.0
        self.fps = 0.0
        self.coeff: Optional[Tuple[float, float, float, float]] = None  # from AT+COEFF? (100x100)
        self.connected = False
        self.reconnects = 0
        self.last_error = ""
        self.t_start = time.time()
        self._ser = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fid: Optional[int] = None
        self._t_open = 0.0

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        target = self._run_mock if self.mock else self._run_serial
        self._thread = threading.Thread(target=target, name="maixsense", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close()

    # ---- frame handling ------------------------------------------------
    def intrinsics(self, rows: int = 100, cols: int = 100) -> Tuple[float, float, float, float]:
        if self.coeff is not None:
            s = cols / 100.0  # coeff is for the native 100x100 grid; scale for BINN
            fx, fy, u0, v0 = self.coeff
            return fx * s, fy * s, u0 * s, v0 * s
        return P.intrinsics_from_fov(cols, rows, self.hfov_deg, self.vfov_deg)

    def feed(self, data: bytes, t: Optional[float] = None) -> int:
        """Feed raw serial bytes; publishes the newest valid frame. Returns #frames."""
        frames = self.parser.feed(data)
        if not frames:
            return 0
        t = time.time() if t is None else t
        for f in frames:
            if f.frame_id == self._last_fid:
                continue  # [C] duplicate frame ids are dropped
            self._last_fid = f.frame_id
            d = P.depth_meters(f.depth, self.unit, min_m=self.min_m, max_m=self.max_m)
            with self.lock:
                if self.t_capture:
                    dt = t - self.t_capture
                    if dt > 0:
                        self.fps = 1.0 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
                self.depth_m = d
                self.last_frame = f
                self.t_capture = t
        return len(frames)

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            return self.depth_m, self.t_capture

    def health(self) -> Dict[str, Any]:
        st = self.parser.stats
        age = (time.time() - self.t_capture) if self.t_capture else None
        f = self.last_frame
        rows, cols = (f.rows, f.cols) if f else (100, 100)
        return {
            "ok": age is not None and age < 2.0,
            "mock": self.mock,
            "device": self.device,
            "connected": self.connected,
            "unit": self.unit,
            "fps": round(self.fps, 2),
            "fps_cfg": self.fps_cfg,
            "frames": st.frames,
            "checksum_errors": st.checksum_errors,
            "tail_errors": st.tail_errors,
            "length_errors": st.length_errors,
            "bytes_dropped": st.bytes_dropped,
            "last_frame_age_s": None if age is None else round(age, 3),
            "last_frame_id": f.frame_id if f else None,
            "resolution": [rows, cols],
            "sensor_temp": f.sensor_temp if f else None,
            "error_code": f.error_code if f else None,
            "intrinsics": dict(zip(("fx", "fy", "cx", "cy"), self.intrinsics(rows, cols))),
            "intrinsics_source": "AT+COEFF?" if self.coeff else "fov %.0fx%.0f deg" % (self.hfov_deg, self.vfov_deg),
            "reconnects": self.reconnects,
            "last_error": self.last_error,
        }

    # ---- mock ----------------------------------------------------------
    def mock_step(self, t: Optional[float] = None) -> int:
        t = time.time() if t is None else t
        fid = (self.parser.stats.frames + 1) & 0x0FFF
        codes = P.mm_to_code(P.synth_depth_m(t - self.t_start) * 1000.0, self.unit)
        self.connected = True
        return self.feed(P.encode_frame(codes, frame_id=fid), t)

    def _run_mock(self) -> None:
        period = 1.0 / max(1, self.fps_cfg)
        while not self._stop.is_set():
            self.mock_step()
            self._stop.wait(period)

    # ---- serial --------------------------------------------------------
    def _close(self) -> None:
        self.connected = False
        if self._ser is not None:
            try:
                self._ser.write(P.at("ISP", 0))
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def _cmd(self, cmd: bytes, wait: float = 0.15) -> bytes:
        self._ser.write(cmd)
        self._ser.flush()
        time.sleep(wait)
        return self._ser.read(self._ser.in_waiting or 0)

    def _open(self) -> None:
        import serial  # pyserial, lazy so tests need no serial stack

        self.reconnects += 1
        ser = serial.Serial(self.device, self.baud, timeout=0.1)
        self._ser = ser
        self._cmd(P.at("ISP", 0), 0.3)  # stop streaming first
        ser.reset_input_buffer()
        if self.use_coeff and self.coeff is None:
            ser.write(P.at("COEFF", query=True))
            deadline, resp = time.time() + 1.5, b""
            while time.time() < deadline and b"}" not in resp:
                resp += ser.read(512)
            self.coeff = P.parse_coeff(resp.decode("ascii", "ignore"))
            log.info("AT+COEFF? -> %s", self.coeff)
        for c in P.init_sequence(unit=self.unit, fps=self.fps_cfg, disp=P.DISP_USB)[1:]:
            self._cmd(c)
        self.parser.reset()
        self._t_open = time.time()
        self.connected = True
        self.last_error = ""
        log.info("opened %s @ %d (unit=%d fps=%d)", self.device, self.baud, self.unit, self.fps_cfg)

    def _run_serial(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                if self._ser is None:
                    self._open()
                    backoff = 0.5
                data = self._ser.read(max(1, min(8192, self._ser.in_waiting or 1)))
                if data:
                    self.feed(data)
                elif time.time() - max(self.t_capture, self._t_open) > 5.0:
                    raise IOError("no valid frames for 5 s")
            except Exception as e:  # unplugged, permission, timeout...
                self.last_error = str(e)
                log.warning("serial error: %s (retry in %.1fs)", e, backoff)
                self._close()
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, 10.0)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def colorize(depth_m: np.ndarray, max_m: float = 2.5) -> bytes:
    import cv2

    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    norm = np.clip(depth_m / max_m * 255.0, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(norm, cmap)
    img[depth_m <= 0] = 0
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return png.tobytes()


def create_app(bridge: MaixSenseBridge):
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="maixsense_bridge", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
                       expose_headers=["X-Capture-Time", "X-Points"])

    def _latest():
        d, t = bridge.latest()
        if d is None:
            raise HTTPException(503, "no frame yet (%s)" % (bridge.last_error or "starting"))
        return d, t

    def _npy(arr: np.ndarray, t: float, extra: Optional[Dict[str, str]] = None):
        bio = io.BytesIO()
        np.save(bio, arr.astype(np.float32), allow_pickle=False)
        h = {"X-Capture-Time": "%.6f" % t, "Cache-Control": "no-store"}
        h.update(extra or {})
        return Response(bio.getvalue(), media_type="application/octet-stream", headers=h)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return bridge.health()

    @app.get("/frame.npy")
    def frame_npy():
        d, t = _latest()
        return _npy(d, t)

    @app.get("/frame.png")
    def frame_png(max_m: float = 2.5):
        d, t = _latest()
        return Response(colorize(d, max_m), media_type="image/png",
                        headers={"X-Capture-Time": "%.6f" % t, "Cache-Control": "no-store"})

    @app.get("/points.npy")
    def points_npy():
        d, t = _latest()
        pts = P.depth_to_points(d, bridge.intrinsics(d.shape[0], d.shape[1]))
        return _npy(pts, t, {"X-Points": str(len(pts))})

    return app


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", default=_env("MAIXSENSE_MOCK", "0") == "1")
    ap.add_argument("--device", default=_env("MAIXSENSE_DEVICE", "/dev/ttyUSB0"))
    ap.add_argument("--baud", type=int, default=int(_env("MAIXSENSE_BAUD", "115200")))
    ap.add_argument("--unit", type=int, default=int(_env("MAIXSENSE_UNIT", "0")))
    ap.add_argument("--fps", type=int, default=int(_env("MAIXSENSE_FPS", "15")))
    ap.add_argument("--hfov", type=float, default=float(_env("MAIXSENSE_HFOV", str(P.DEFAULT_HFOV_DEG))))
    ap.add_argument("--vfov", type=float, default=float(_env("MAIXSENSE_VFOV", str(P.DEFAULT_VFOV_DEG))))
    ap.add_argument("--no-coeff", action="store_true", default=_env("MAIXSENSE_USE_COEFF", "1") == "0",
                    help="do not query AT+COEFF? intrinsics; use FOV-based pinhole")
    ap.add_argument("--min-m", type=float, default=float(_env("MAIXSENSE_MIN_M", "0.0")))
    ap.add_argument("--max-m", type=float, default=float(_env("MAIXSENSE_MAX_M", "10.0")))
    ap.add_argument("--host", default=_env("MAIXSENSE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(_env("MAIXSENSE_PORT", "9121")))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bridge = MaixSenseBridge(a.device, a.baud, a.unit, a.fps, a.hfov, a.vfov, not a.no_coeff,
                             a.min_m, a.max_m, mock=a.mock)
    bridge.start()
    import uvicorn

    try:
        uvicorn.run(create_app(bridge), host=a.host, port=a.port, log_level="warning")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
