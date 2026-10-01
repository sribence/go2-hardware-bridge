#!/usr/bin/env python3
"""thermal_bridge -- USB (UVC) thermal cameras -> HTTP (FastAPI, :9120).

OmniVision 360 contract (go2-brain-logic/mission_control/omni/CONTRACTS.md, F):
  GET /cams                      camera list + status
  GET /cams/{id}/frame.npy       float32 degC (np.save bytes), header X-Capture-Time
  GET /cams/{id}/frame.png       inferno-colorized, headers X-Temp-Min / X-Temp-Max
  GET /cams/{id}/stats           min / max / mean / hotspot
  GET /health

Run:  python3 thermal_bridge.py [--mock] [--config thermal_cams.yaml] [--port 9120]
Read-only sensor bridge: never sends anything to the robot.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:  # package import (pytest) vs. script run
    from . import decoders
except ImportError:  # pragma: no cover
    import decoders  # type: ignore

log = logging.getLogger("thermal_bridge")

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thermal_cams.yaml")


def load_config(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cams = cfg.get("cameras") or {}
    out = {}
    for cam_id, c in cams.items():
        c = dict(c or {})
        c.setdefault("decoder", "infiray_p2")
        c.setdefault("width", 256)
        c.setdefault("height", 192)
        c.setdefault("capture_width", c["width"])
        c.setdefault("capture_height", c["height"] * 2 if c["decoder"] in ("infiray_p2", "tc001") else c["height"])
        c.setdefault("fourcc", "YUYV")
        c.setdefault("fps", 25)
        c.setdefault("hfov_deg", 50.0)
        c.setdefault("decoder_params", {})
        decoders.get_decoder(c["decoder"])  # validate early
        out[str(cam_id)] = c
    return out


# --------------------------------------------------------------------------
# Mock scene
# --------------------------------------------------------------------------
def synth_scene(t: float, width: int, height: int, hfov_deg: float,
                offset_px: float = 0.0, seed: int = 0) -> np.ndarray:
    """~20 degC room with a walking 34-36 degC person.

    Person is 0.5 m x 1.7 m, walking back and forth (2.5 .. 7 m distance);
    its apparent size follows the lens focal length, so the narrow-FOV camera
    sees a bigger blob than the wide one.
    """
    rng = np.random.default_rng(seed + int(t * 25))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img = 19.5 + 1.5 * (yy / height) + 0.3 * np.sin(xx / 23.0)  # floor warmer
    img += rng.normal(0.0, 0.08, size=img.shape).astype(np.float32)

    f_px = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    dist = 4.75 + 2.25 * math.sin(t * 0.4)
    lateral = 1.2 * math.sin(t * 0.7)  # metres, + right
    cx = width / 2.0 + f_px * lateral / dist + offset_px
    cy = height / 2.0 + f_px * 0.2 / dist  # camera slightly above hip
    h_px = f_px * 1.7 / dist
    w_px = f_px * 0.5 / dist
    # torso ellipse
    torso = ((xx - cx) / (w_px / 2.0)) ** 2 + ((yy - (cy + 0.05 * h_px)) / (0.35 * h_px)) ** 2 <= 1.0
    # head circle
    head = ((xx - cx) ** 2 + (yy - (cy - 0.40 * h_px)) ** 2) <= (0.08 * h_px) ** 2
    # legs
    legs = (np.abs(xx - cx) <= 0.35 * w_px) & (yy >= cy + 0.3 * h_px) & (yy <= cy + 0.5 * h_px)
    img = np.where(torso | legs, 34.0 + 0.5 * np.sin(yy / 5.0), img)
    img = np.where(head, 35.8, img)
    return img.astype(np.float32)


# --------------------------------------------------------------------------
# Per-camera worker
# --------------------------------------------------------------------------
class CamWorker:
    def __init__(self, cam_id: str, cfg: Dict[str, Any], mock: bool = False):
        self.cam_id = cam_id
        self.cfg = cfg
        self.mock = mock
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.t_capture = 0.0
        self.frames = 0
        self.decode_errors = 0
        self.reopen_count = 0
        self.last_error = ""
        self.connected = False
        self._cap = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_ema = 0.0
        self._t0 = time.time()

    # ---- public --------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="thermal-" + self.cam_id, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release()

    def latest(self):
        with self.lock:
            return (None if self.frame is None else self.frame, self.t_capture)

    def status(self) -> Dict[str, Any]:
        age = (time.time() - self.t_capture) if self.t_capture else None
        return {
            "id": self.cam_id,
            "device": self.cfg.get("device"),
            "decoder": self.cfg["decoder"],
            "width": self.cfg["width"],
            "height": self.cfg["height"],
            "hfov_deg": self.cfg.get("hfov_deg"),
            "fov_note": self.cfg.get("fov_note", ""),
            "mock": self.mock,
            "connected": self.connected,
            "frames": self.frames,
            "fps": round(self._fps_ema, 2),
            "last_frame_age_s": None if age is None else round(age, 3),
            "decode_errors": self.decode_errors,
            "reopen_count": self.reopen_count,
            "last_error": self.last_error,
        }

    def step(self) -> bool:
        """Grab + decode one frame. Returns True on success."""
        if self.mock:
            t = time.time()
            temp = synth_scene(t - self._t0, self.cfg["width"], self.cfg["height"],
                               float(self.cfg.get("hfov_deg", 50.0)),
                               float(self.cfg.get("mock_offset_px", 0)))
            dec = self.cfg["decoder"]
            if dec in ("infiray_p2", "tc001"):
                buf = decoders.encode_infiray(temp, self.cfg["decoder_params"].get("thermal_half", "bottom"))
                temp = decoders.decode(dec, buf, self.cfg["width"], self.cfg["height"], self.cfg["decoder_params"])
            elif dec == "raw_y16":
                p = self.cfg["decoder_params"]
                buf = decoders.encode_y16(temp, p.get("scale", 1 / 64.0), p.get("offset", -decoders.KELVIN_OFFSET))
                temp = decoders.decode(dec, buf, self.cfg["width"], self.cfg["height"], p)
            self.connected = True
            self._publish(temp, t)
            return True

        if self._cap is None and not self._open():
            return False
        ok, raw = self._cap.read()
        t = time.time()
        if not ok or raw is None:
            self.last_error = "read failed"
            self._release()
            return False
        try:
            temp = decoders.decode(self.cfg["decoder"], raw, self.cfg["width"], self.cfg["height"],
                                   self.cfg["decoder_params"])
        except Exception as e:  # wrong stream format etc.
            self.decode_errors += 1
            self.last_error = "decode: %s" % e
            return False
        self._publish(temp, t)
        return True

    # ---- internals -----------------------------------------------------
    def _publish(self, temp: np.ndarray, t: float) -> None:
        with self.lock:
            if self.t_capture:
                dt = t - self.t_capture
                if dt > 0:
                    self._fps_ema = 1.0 / dt if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 / dt
            self.frame = temp
            self.t_capture = t
            self.frames += 1

    def _open(self) -> bool:
        import cv2  # lazy: tests / mock do not need a camera stack

        dev = self.cfg.get("device")
        self.reopen_count += 1
        if not dev or not os.path.exists(str(dev)):
            self.last_error = "device not found: %s" % dev
            return False
        cap = cv2.VideoCapture(os.path.realpath(str(dev)), cv2.CAP_V4L2)
        if not cap.isOpened():
            self.last_error = "open failed: %s" % dev
            return False
        fourcc = str(self.cfg.get("fourcc", "YUYV"))[:4].ljust(4)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg["capture_width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg["capture_height"]))
        cap.set(cv2.CAP_PROP_FPS, float(self.cfg.get("fps", 25)))
        # Raw YUYV bytes: the bottom half holds 16-bit temperatures and must
        # not go through OpenCV's YUV->BGR conversion.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._cap = cap
        self.connected = True
        self.last_error = ""
        log.info("%s: opened %s", self.cam_id, dev)
        return True

    def _release(self) -> None:
        self.connected = False if not self.mock else self.connected
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _run(self) -> None:
        backoff = 0.5
        period = 1.0 / max(1.0, float(self.cfg.get("fps", 25)))
        while not self._stop.is_set():
            ok = self.step()
            if ok:
                backoff = 0.5
                if self.mock:
                    self._stop.wait(period)
            else:
                # reopen with exponential backoff (max 10 s)
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, 10.0)


class ThermalBridge:
    def __init__(self, cams_cfg: Dict[str, Dict[str, Any]], mock: bool = False):
        self.mock = mock
        self.workers: Dict[str, CamWorker] = {k: CamWorker(k, v, mock) for k, v in cams_cfg.items()}
        self.t_start = time.time()

    def start(self) -> None:
        for w in self.workers.values():
            w.start()

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def colorize(temp: np.ndarray, t_min: Optional[float] = None, t_max: Optional[float] = None) -> bytes:
    import cv2

    lo = float(np.nanmin(temp)) if t_min is None else float(t_min)
    hi = float(np.nanmax(temp)) if t_max is None else float(t_max)
    norm = np.clip((temp - lo) / max(hi - lo, 1e-3) * 255.0, 0, 255)
    img = cv2.applyColorMap(np.nan_to_num(norm).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return png.tobytes()


def create_app(bridge: ThermalBridge):
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="thermal_bridge", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
                       expose_headers=["X-Capture-Time", "X-Temp-Min", "X-Temp-Max"])

    def _frame(cam_id: str):
        w = bridge.workers.get(cam_id)
        if w is None:
            raise HTTPException(404, "unknown camera %r" % cam_id)
        frame, t = w.latest()
        if frame is None:
            raise HTTPException(503, "no frame yet from %s (%s)" % (cam_id, w.last_error or "starting"))
        return frame, t

    @app.get("/health")
    def health() -> Dict[str, Any]:
        st = [w.status() for w in bridge.workers.values()]
        ok = all(s["last_frame_age_s"] is not None and s["last_frame_age_s"] < 2.0 for s in st)
        return {"ok": ok, "mock": bridge.mock, "uptime_s": round(time.time() - bridge.t_start, 1),
                "cams": {s["id"]: {k: s[k] for k in ("connected", "fps", "last_frame_age_s", "frames",
                                                     "decode_errors", "last_error")} for s in st}}

    @app.get("/cams")
    def cams() -> List[Dict[str, Any]]:
        return [w.status() for w in bridge.workers.values()]

    @app.get("/cams/{cam_id}/frame.npy")
    def frame_npy(cam_id: str):
        frame, t = _frame(cam_id)
        bio = io.BytesIO()
        np.save(bio, frame.astype(np.float32), allow_pickle=False)
        return Response(bio.getvalue(), media_type="application/octet-stream",
                        headers={"X-Capture-Time": "%.6f" % t, "Cache-Control": "no-store"})

    @app.get("/cams/{cam_id}/frame.png")
    def frame_png(cam_id: str, t_min: Optional[float] = None, t_max: Optional[float] = None):
        frame, t = _frame(cam_id)
        lo = float(np.nanmin(frame)) if t_min is None else t_min
        hi = float(np.nanmax(frame)) if t_max is None else t_max
        return Response(colorize(frame, lo, hi), media_type="image/png",
                        headers={"X-Capture-Time": "%.6f" % t, "X-Temp-Min": "%.2f" % lo,
                                 "X-Temp-Max": "%.2f" % hi, "Cache-Control": "no-store"})

    @app.get("/cams/{cam_id}/stats")
    def stats(cam_id: str) -> Dict[str, Any]:
        frame, t = _frame(cam_id)
        s = decoders.frame_stats(frame)
        s["t"] = t
        s["id"] = cam_id
        return s

    return app


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", default=os.environ.get("THERMAL_MOCK", "0") == "1")
    ap.add_argument("--config", default=os.environ.get("THERMAL_CONFIG", DEFAULT_CONFIG))
    ap.add_argument("--host", default=os.environ.get("THERMAL_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("THERMAL_PORT", "9120")))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bridge = ThermalBridge(load_config(args.config), mock=args.mock)
    bridge.start()
    import uvicorn

    try:
        uvicorn.run(create_app(bridge), host=args.host, port=args.port, log_level="warning")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
