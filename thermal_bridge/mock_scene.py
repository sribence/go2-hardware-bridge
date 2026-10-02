"""Synthetic thermal scene + fake MLX90640 USB device (mock mode, tests, tools/).

``FakeMlxDevice`` emulates our mlx90640_usb firmware byte-for-byte: it emits
MLXF v1 packets for both sensors, answers the text commands (RATE / EMIS /
INFO?) with '#' JSON lines, so ``--mock`` and tools/mlx_fake_device.py go
through exactly the same parser as real hardware.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # package import (pytest) vs. script run
    from . import mlx90640_protocol as MP
except ImportError:  # pragma: no cover
    import mlx90640_protocol as MP  # type: ignore


def synth_scene(t: float, width: int, height: int, hfov_deg: float,
                offset_px: float = 0.0, seed: int = 0, vfov_deg: Optional[float] = None,
                supersample: int = 1) -> np.ndarray:
    """~20 degC room with a walking 34-36 degC person.

    Person is 0.5 m x 1.7 m, walking back and forth (2.5 .. 7 m distance);
    its apparent size follows the lens focal length, so the narrow-FOV camera
    sees a bigger blob than the wide one. ``vfov_deg`` gives a separate
    vertical focal length (non-square sensor FOV, e.g. MLX90640 55x35 deg);
    ``supersample`` > 1 box-filters a finer render (mixed pixels, like a real
    low-resolution bolometer).
    """
    ss = max(1, int(supersample))
    w, h = width * ss, height * ss
    rng = np.random.default_rng(seed + int(t * 25))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 19.5 + 1.5 * (yy / h) + 0.3 * np.sin(xx / (23.0 * ss))  # floor warmer

    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx if vfov_deg is None else (h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    dist = 4.75 + 2.25 * math.sin(t * 0.4)
    lateral = 1.2 * math.sin(t * 0.7)  # metres, + right
    cx = w / 2.0 + fx * lateral / dist + offset_px * ss
    cy = h / 2.0 + fy * 0.2 / dist  # camera slightly above hip
    h_px = fy * 1.7 / dist
    w_px = fx * 0.5 / dist
    # torso ellipse
    torso = ((xx - cx) / (w_px / 2.0)) ** 2 + ((yy - (cy + 0.05 * h_px)) / (0.35 * h_px)) ** 2 <= 1.0
    # head circle (ellipse when fx != fy)
    head = ((xx - cx) / (0.08 * h_px * fx / fy)) ** 2 + ((yy - (cy - 0.40 * h_px)) / (0.08 * h_px)) ** 2 <= 1.0
    # legs
    legs = (np.abs(xx - cx) <= 0.35 * w_px) & (yy >= cy + 0.3 * h_px) & (yy <= cy + 0.5 * h_px)
    img = np.where(torso | legs, 34.0 + 0.5 * np.sin(yy / (5.0 * ss)), img)
    img = np.where(head, 35.8, img)
    if ss > 1:
        img = img.reshape(height, ss, width, ss).mean(axis=(1, 3))
    img = img + rng.normal(0.0, 0.08, size=img.shape)
    return img.astype(np.float32)


class FakeSensor(object):
    def __init__(self, sensor_id: int, fov_deg: Sequence[float], offset_px: float = 0.0,
                 present: bool = True):
        self.sensor_id = sensor_id
        self.fov_deg = (float(fov_deg[0]), float(fov_deg[1]))
        self.offset_px = offset_px
        self.present = present
        self.seq = 0
        self.frames = 0


# Our hardware plan: narrow MLX90640ESF-BAA (55x35), wide MLX90640ESF-BAB (110x75).
DEFAULT_SENSORS = ((0, (55.0, 35.0)), (1, (110.0, 75.0)))


class FakeMlxDevice(object):
    """Byte-level emulation of the mlx90640_usb firmware."""

    def __init__(self, sensors: Sequence = DEFAULT_SENSORS, rate_hz: float = 8.0,
                 emissivity: float = 0.95, ta: float = 27.5, t0: float = 0.0):
        self.sensors: Dict[int, FakeSensor] = {}
        for s in sensors:
            fs = s if isinstance(s, FakeSensor) else FakeSensor(s[0], s[1])
            self.sensors[fs.sensor_id] = fs
        self.rate_hz = rate_hz
        self.emissivity = emissivity
        self.ta = ta
        self.t0 = t0
        self._rx = bytearray()
        self._pending_text: List[bytes] = []

    @property
    def frame_period_s(self) -> float:
        """Full frame = 2 chess subpages."""
        return 2.0 / self.rate_hz

    def frame(self, sensor_id: int, t: float) -> np.ndarray:
        s = self.sensors[sensor_id]
        return synth_scene(t - self.t0, MP.COLS, MP.ROWS, s.fov_deg[0], s.offset_px,
                           seed=sensor_id * 1000, vfov_deg=s.fov_deg[1], supersample=4)

    def packet(self, sensor_id: int, t: float) -> bytes:
        s = self.sensors[sensor_id]
        if not s.present:
            return b""
        pkt = MP.encode_packet(sensor_id, s.seq, int((t - self.t0) * 1000), self.ta, self.frame(sensor_id, t))
        s.seq = (s.seq + 1) & 0xFFFF
        s.frames += 1
        return pkt

    def step(self, t: float) -> bytes:
        """Bytes the device emits for one frame period: pending text, then all sensors."""
        out = b"".join(self._pending_text)
        self._pending_text = []
        for sid in sorted(self.sensors):
            out += self.packet(sid, t)
        return out

    # ---- host -> device text commands -------------------------------------
    def info(self) -> Dict:
        return {"fw": "mlx90640_usb", "version": "fake", "proto": MP.VERSION, "board": "fake",
                "rate_hz": self.rate_hz, "emissivity": self.emissivity, "mode": "chess",
                "i2c_hz": 1000000,
                "sensors": [{"id": s.sensor_id, "name": "narrow" if s.sensor_id == 0 else "wide",
                             "state": "ok" if s.present else "missing", "serial": "FAKE%08d" % s.sensor_id,
                             "frames": s.frames, "errors": 0, "ta": self.ta}
                            for s in self.sensors.values()]}

    def _reply(self, obj: Dict) -> None:
        self._pending_text.append(b"#" + json.dumps(obj, separators=(",", ":")).encode() + b"\n")

    def write(self, data: bytes) -> None:
        """Feed host->device bytes; complete lines are executed, replies queued."""
        self._rx += data
        while b"\n" in self._rx:
            line, _, rest = bytes(self._rx).partition(b"\n")
            self._rx = bytearray(rest)
            self.command(line.decode("ascii", "replace").strip())

    def command(self, line: str) -> None:
        if not line:
            return
        cmd, _, arg = line.partition(" ")
        if cmd == "INFO?":
            self._reply(self.info())
        elif cmd == "RATE":
            try:
                hz = float(arg)
                MP.cmd_rate(hz)
                self.rate_hz = hz
                self._reply({"ok": True, "cmd": "RATE", "detail": arg})
            except ValueError:
                self._reply({"ok": False, "cmd": "RATE", "detail": "allowed: 0.5 1 2 4 8 16 32 64"})
        elif cmd == "EMIS":
            try:
                e = float(arg)
                MP.cmd_emissivity(e)
                self.emissivity = e
                self._reply({"ok": True, "cmd": "EMIS", "detail": arg})
            except ValueError:
                self._reply({"ok": False, "cmd": "EMIS", "detail": "range 0.1 .. 1.0"})
        else:
            self._reply({"ok": False, "cmd": cmd, "detail": "unknown command"})
