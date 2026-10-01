"""Pure decoders: raw UVC thermal frame bytes -> float32 degrees Celsius.

All functions are side-effect free and unit-tested (tests/test_thermal_decoders.py).

Supported decoders (select with ``decoder:`` in thermal_cams.yaml):

* ``infiray_p2`` / ``tc001`` -- InfiRay P2 Pro / TOPDON TC001 class modules.
  The UVC stream is 256x384 YUYV (2 bytes/pixel). One half (default: bottom,
  rows 192..383) is NOT an image but raw little-endian uint16 temperature
  per pixel: ``degC = raw / 64 - 273.15``. The other half is the vendor's
  8-bit AGC preview image (ignored here).
* ``raw_y16`` -- generic Y16 sensor (e.g. Lepton/Boson radiometric, or any
  module exposing a 16-bit stream): ``degC = raw * scale + offset``.
* ``grey8_relative`` -- fallback for modules that only output an AGC'd 8-bit
  image (no radiometry). Grey 0..255 is mapped linearly to
  ``[t_min, t_max]``. Values are RELATIVE, not absolute temperatures.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

KELVIN_OFFSET = 273.15


def _as_bytes_array(buf: Any) -> np.ndarray:
    """Flatten whatever cv2/V4L2 returned (1xN, HxWx2, bytes...) to uint8[]."""
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(buf), dtype=np.uint8)
    arr = np.asarray(buf)
    if arr.dtype != np.uint8:
        arr = arr.view(np.uint8) if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr).view(np.uint8)
    return arr.reshape(-1)


def decode_infiray(buf: Any, width: int = 256, height: int = 192,
                   thermal_half: str = "bottom", **_: Any) -> np.ndarray:
    """InfiRay P2 Pro / TC001: 2*height rows of YUYV, one half raw uint16 temp."""
    data = _as_bytes_array(buf)
    half = width * height * 2
    if data.size < 2 * half:
        raise ValueError("infiray frame too short: %d bytes, expected %d" % (data.size, 2 * half))
    seg = data[half:2 * half] if thermal_half == "bottom" else data[:half]
    raw = seg.view("<u2").reshape(height, width)
    return (raw.astype(np.float32) / 64.0 - KELVIN_OFFSET).astype(np.float32)


def decode_raw_y16(buf: Any, width: int = 256, height: int = 192,
                   scale: float = 1.0 / 64.0, offset: float = -KELVIN_OFFSET,
                   **_: Any) -> np.ndarray:
    """Generic little-endian Y16: degC = raw * scale + offset."""
    data = _as_bytes_array(buf)
    n = width * height * 2
    if data.size < n:
        raise ValueError("y16 frame too short: %d bytes, expected %d" % (data.size, n))
    raw = data[:n].view("<u2").reshape(height, width)
    return (raw.astype(np.float32) * np.float32(scale) + np.float32(offset)).astype(np.float32)


def decode_grey8_relative(buf: Any, width: int = 256, height: int = 192,
                          t_min: float = 15.0, t_max: float = 40.0,
                          **_: Any) -> np.ndarray:
    """8-bit grey (or YUYV luma, or BGR) -> linear [t_min, t_max] degC."""
    arr = np.asarray(buf) if not isinstance(buf, (bytes, bytearray, memoryview)) else None
    if arr is not None and arr.ndim == 3 and arr.shape[2] == 3:
        grey = arr.mean(axis=2)  # already decoded BGR image
    elif arr is not None and arr.ndim == 2 and arr.shape == (height, width):
        grey = arr
    else:
        data = _as_bytes_array(buf)
        n = width * height
        if data.size >= 2 * n:
            grey = data[: 2 * n : 2]  # YUYV: Y on even bytes
        elif data.size >= n:
            grey = data[:n]
        else:
            raise ValueError("grey8 frame too short: %d bytes" % data.size)
        grey = grey.reshape(height, width)
    g = np.asarray(grey, dtype=np.float32) / 255.0
    return (np.float32(t_min) + g * np.float32(t_max - t_min)).astype(np.float32)


DECODERS: Dict[str, Callable[..., np.ndarray]] = {
    "infiray_p2": decode_infiray,
    "tc001": decode_infiray,  # same 256x384 YUYV + raw-temperature layout
    "raw_y16": decode_raw_y16,
    "grey8_relative": decode_grey8_relative,
}


def get_decoder(name: str) -> Callable[..., np.ndarray]:
    try:
        return DECODERS[name]
    except KeyError:
        raise ValueError("unknown thermal decoder %r (known: %s)" % (name, ", ".join(sorted(DECODERS))))


def decode(name: str, buf: Any, width: int, height: int,
           params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    return get_decoder(name)(buf, width=width, height=height, **(params or {}))


# --------------------------------------------------------------------------
# Encoders (inverse) -- used by --mock mode and unit tests only.
# --------------------------------------------------------------------------
def celsius_to_raw(temp_c: np.ndarray) -> np.ndarray:
    return np.clip(np.round((np.asarray(temp_c, np.float64) + KELVIN_OFFSET) * 64.0), 0, 65535).astype("<u2")


def encode_infiray(temp_c: np.ndarray, thermal_half: str = "bottom") -> bytes:
    """Build a synthetic 256x384 YUYV InfiRay buffer from a HxW degC image."""
    h, w = temp_c.shape
    raw = celsius_to_raw(temp_c).tobytes()
    t = np.asarray(temp_c, np.float32)
    lo, hi = float(t.min()), float(t.max())
    y = np.clip((t - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    yuyv = np.empty((h, w, 2), np.uint8)
    yuyv[..., 0] = y
    yuyv[..., 1] = 128
    img = yuyv.tobytes()
    return img + raw if thermal_half == "bottom" else raw + img


def encode_y16(temp_c: np.ndarray, scale: float = 1.0 / 64.0, offset: float = -KELVIN_OFFSET) -> bytes:
    raw = np.clip(np.round((np.asarray(temp_c, np.float64) - offset) / scale), 0, 65535).astype("<u2")
    return raw.tobytes()


def frame_stats(temp_c: np.ndarray) -> Dict[str, Any]:
    """min/max/mean + hotspot (x=col, y=row) of a degC image (NaN-safe)."""
    t = np.asarray(temp_c, np.float32)
    finite = np.isfinite(t)
    if not finite.any():
        return {"min": None, "max": None, "mean": None, "hotspot": None}
    tt = np.where(finite, t, -np.inf)
    idx = int(np.argmax(tt))
    row, col = divmod(idx, t.shape[1])
    return {
        "min": float(np.nanmin(t)),
        "max": float(t[row, col]),
        "mean": float(np.nanmean(t)),
        "hotspot": {"x": int(col), "y": int(row), "temp": float(t[row, col])},
        "width": int(t.shape[1]),
        "height": int(t.shape[0]),
    }
