"""Sipeed MaixSense A010 ToF -- pure protocol helpers (no I/O).

Sources (see README "Protocol provenance"):
  [W] https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a010/at_command_en.html
      (raw: github.com/sipeed/sipeed_wiki docs/hardware/en/maixsense/maixsense-a010/at_command_en.md)
  [C] github.com/sipeed/sipeed_wiki docs/hardware/zh/maixsense/maixsense-a010/code.md (Python examples)
  [R] github.com/sipeed/MaixSense_ROS sipeed_tof_ms_a010_ros/ros2/src/frame_struct.h, frame_handle.cc, main.cc

Packet (all multi-byte fields little-endian) [W][R]:
  off  size  field
   0    2    header 0x00 0xFF
   2    2    frame_data_len = 16 + payload_len  (bytes after this field, excl. checksum+tail)
   4    1    reserved1 (0xFF)
   5    1    output_mode (0 depth only, 1 depth+IR)
   6    1    sensor_temp
   7    1    driver_temp
   8    4    exposure_time
  12    1    error_code
  13    1    reserved2 (0x00)
  14    1    resolution_rows
  15    1    resolution_cols
  16    2    frame_id (12-bit, 0..4095)
  18    1    isp_version
  19    1    reserved3 (0xFF)
  20    N    payload: rows*cols uint8 depth codes (row-major)
  20+N  1    checksum = sum(bytes[0 : 20+N]) & 0xFF
  21+N  1    tail 0xDD
Depth [W]: UNIT=0 -> d_mm = (p/5.1)^2 ; UNIT=k (1..10) -> d_mm = p*k.
"""
from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

HEADER = b"\x00\xFF"
TAIL = 0xDD
HEAD_SIZE = 20          # header(2) + len(2) + metadata(16)
META_SIZE = 16
MAX_PAYLOAD = 100 * 100 * 2  # depth+IR worst case; [R] caps depth-only at 100*100

# AT+BAUD index -> baud rate [W]
BAUD_TABLE = {0: 9600, 1: 57600, 2: 115200, 3: 230400, 4: 460800, 5: 921600,
              6: 1000000, 7: 2000000, 8: 3000000}
# AT+DISP values [W]
DISP_OFF, DISP_LCD, DISP_USB, DISP_LCD_USB, DISP_UART = 0, 1, 2, 3, 4

DEFAULT_HFOV_DEG = 70.0  # ASSUMED (contract / vendor listings), not in wiki pages fetched
DEFAULT_VFOV_DEG = 60.0


# --------------------------------------------------------------------------
# AT commands
# --------------------------------------------------------------------------
def at(cmd: str, value: Optional[object] = None, query: bool = False) -> bytes:
    """Build an AT command. at('FPS', 15) -> b'AT+FPS=15\\r'; at('UNIT', query=True) -> b'AT+UNIT?\\r'."""
    cmd = cmd.upper().lstrip("+")
    if cmd.startswith("AT+"):
        cmd = cmd[3:]
    if query:
        return ("AT+%s?\r" % cmd).encode("ascii")
    if value is None:
        return ("AT+%s\r" % cmd).encode("ascii") if cmd else b"AT\r"
    return ("AT+%s=%s\r" % (cmd, value)).encode("ascii")


def init_sequence(unit: int = 0, fps: int = 15, disp: int = DISP_USB, binn: int = 1) -> List[bytes]:
    """stop ISP -> configure -> start ISP. Values validated against [W] ranges."""
    if not 0 <= unit <= 10:
        raise ValueError("UNIT must be 0..10")
    if not 1 <= fps <= 19:
        raise ValueError("FPS must be 1..19")
    if not 0 <= disp <= 7:
        raise ValueError("DISP must be 0..7")
    if binn not in (1, 2, 4):
        raise ValueError("BINN must be 1, 2 or 4")
    return [
        at("ISP", 0),       # stop: ISP off, IR emitter off
        at("BINN", binn),
        at("UNIT", unit),
        at("FPS", fps),
        at("DISP", disp),   # 2 = USB output only (LCD off saves bandwidth)
        at("ISP", 1),       # start: frames follow after ~1-2 s
    ]


def parse_coeff(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse the JSON returned by AT+COEFF? -> (fx, fy, u0, v0) in pixels.

    [R] main.cc: values are u14p18 fixed point -> divide by 262144.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return tuple(float(d[k]) / 262144.0 for k in ("fx", "fy", "u0", "v0"))  # type: ignore
    except (ValueError, KeyError, TypeError):
        return None


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
@dataclass
class A010Frame:
    frame_id: int
    rows: int
    cols: int
    output_mode: int
    sensor_temp: int
    driver_temp: int
    exposure_time: int
    error_code: int
    isp_version: int
    depth: np.ndarray            # (rows, cols) uint8 depth codes
    ir: Optional[np.ndarray] = None  # (rows, cols) uint8 when output_mode == 1


def checksum(data: bytes) -> int:
    return sum(data) & 0xFF


def encode_frame(depth: np.ndarray, frame_id: int = 0, output_mode: int = 0,
                 sensor_temp: int = 30, driver_temp: int = 32, exposure_time: int = 1000,
                 error_code: int = 0, isp_version: int = 1, ir: Optional[np.ndarray] = None,
                 corrupt_checksum: bool = False) -> bytes:
    """Build a byte-exact A010 packet (fake-serial generator for mock + tests)."""
    d = np.ascontiguousarray(depth, dtype=np.uint8)
    rows, cols = d.shape
    payload = d.tobytes()
    if output_mode == 1:
        payload += np.ascontiguousarray(ir if ir is not None else np.zeros_like(d), np.uint8).tobytes()
    meta = struct.pack("<BBBBIBBBBHBB", 0xFF, output_mode, sensor_temp & 0xFF, driver_temp & 0xFF,
                       exposure_time & 0xFFFFFFFF, error_code & 0xFF, 0x00, rows, cols,
                       frame_id & 0x0FFF, isp_version & 0xFF, 0xFF)
    assert len(meta) == META_SIZE
    body = HEADER + struct.pack("<H", META_SIZE + len(payload)) + meta + payload
    cs = checksum(body)
    if corrupt_checksum:
        cs = (cs + 1) & 0xFF
    return body + bytes((cs, TAIL))


@dataclass
class ParserStats:
    frames: int = 0
    checksum_errors: int = 0
    tail_errors: int = 0
    length_errors: int = 0
    bytes_dropped: int = 0


class FrameParser:
    """Streaming parser: feed() arbitrary chunks, get complete validated frames.

    Resyncs on the 0x00 0xFF header; a corrupt packet only skips one byte so
    a real header hidden inside garbage/pixel data is still found.
    """

    def __init__(self, max_buffer: int = 1 << 16):
        self.buf = bytearray()
        self.stats = ParserStats()
        self.max_buffer = max_buffer

    def reset(self) -> None:
        self.buf.clear()

    def _drop(self, n: int) -> None:
        del self.buf[:n]
        self.stats.bytes_dropped += n

    def feed(self, chunk: bytes) -> List[A010Frame]:
        self.buf += chunk
        out: List[A010Frame] = []
        while True:
            idx = self.buf.find(HEADER)
            if idx < 0:
                # keep a trailing 0x00 (may be the first header byte)
                keep = 1 if self.buf[-1:] == b"\x00" else 0
                self._drop(len(self.buf) - keep)
                break
            if idx > 0:
                self._drop(idx)
            if len(self.buf) < 4:
                break
            data_len = struct.unpack_from("<H", self.buf, 2)[0]
            payload_len = data_len - META_SIZE
            if payload_len <= 0 or payload_len > MAX_PAYLOAD:
                self.stats.length_errors += 1
                self._drop(1)
                continue
            total = 4 + data_len + 2
            if len(self.buf) < total:
                if len(self.buf) > self.max_buffer:
                    self._drop(1)
                    continue
                break
            pkt = bytes(self.buf[:total])
            if pkt[-1] != TAIL:
                self.stats.tail_errors += 1
                self._drop(1)
                continue
            if checksum(pkt[:-2]) != pkt[-2]:
                self.stats.checksum_errors += 1
                self._drop(1)
                continue
            frame = self._decode(pkt, payload_len)
            if frame is None:
                self.stats.length_errors += 1
                self._drop(1)
                continue
            del self.buf[:total]
            self.stats.frames += 1
            out.append(frame)
        return out

    @staticmethod
    def _decode(pkt: bytes, payload_len: int) -> Optional[A010Frame]:
        (_r1, mode, s_temp, d_temp, exp, err, _r2, rows, cols, fid, isp, _r3) = struct.unpack_from(
            "<BBBBIBBBBHBB", pkt, 4)
        n = rows * cols
        if n == 0 or payload_len < n:
            return None
        payload = np.frombuffer(pkt, dtype=np.uint8, count=payload_len, offset=HEAD_SIZE)
        depth = payload[:n].reshape(rows, cols).copy()
        ir = payload[n:2 * n].reshape(rows, cols).copy() if (mode == 1 and payload_len >= 2 * n) else None
        return A010Frame(frame_id=fid & 0x0FFF, rows=rows, cols=cols, output_mode=mode,
                         sensor_temp=s_temp, driver_temp=d_temp, exposure_time=exp,
                         error_code=err, isp_version=isp, depth=depth, ir=ir)


# --------------------------------------------------------------------------
# Depth conversion / geometry
# --------------------------------------------------------------------------
def depth_mm(codes: np.ndarray, unit: int) -> np.ndarray:
    """uint8 codes -> millimetres (float32) per AT+UNIT [W]."""
    p = np.asarray(codes, dtype=np.float32)
    if unit == 0:
        return (p / 5.1) ** 2
    return p * float(unit)


def mm_to_code(mm: np.ndarray, unit: int) -> np.ndarray:
    """Inverse of depth_mm (for mock/tests)."""
    mm = np.asarray(mm, dtype=np.float64)
    p = 5.1 * np.sqrt(np.clip(mm, 0, None)) if unit == 0 else mm / float(unit)
    return np.clip(np.round(p), 0, 255).astype(np.uint8)


def depth_meters(codes: np.ndarray, unit: int, invalid_codes: Sequence[int] = (0, 255),
                 min_m: float = 0.0, max_m: float = float("inf")) -> np.ndarray:
    """uint8 codes -> float32 metres, 0.0 = invalid.

    ASSUMPTION: code 0 (no return / too close) and 255 (saturated / out of
    range) are treated as invalid; not stated in the vendor docs.
    """
    codes = np.asarray(codes)
    m = depth_mm(codes, unit) / 1000.0
    bad = np.isin(codes, np.asarray(list(invalid_codes), dtype=codes.dtype)) | (m < min_m) | (m > max_m)
    return np.where(bad, 0.0, m).astype(np.float32)


def intrinsics_from_fov(width: int = 100, height: int = 100, hfov_deg: float = DEFAULT_HFOV_DEG,
                        vfov_deg: float = DEFAULT_VFOV_DEG) -> Tuple[float, float, float, float]:
    """Pinhole (fx, fy, cx, cy) from FOV (cx, cy at pixel-centre convention)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fx, fy, (width - 1) / 2.0, (height - 1) / 2.0


def depth_to_points(depth_m: np.ndarray, K: Tuple[float, float, float, float]) -> np.ndarray:
    """(H,W) metres -> (N,3) float32 points in OpenCV cam frame (z fwd, x right, y down).

    The A010 value is used as z (as in [R] main.cc point-cloud code); invalid
    (0) pixels are dropped.
    """
    fx, fy, cx, cy = K
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth_m.astype(np.float32)
    ok = z > 0
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x[ok], y[ok], z[ok]], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Mock scene + fake serial stream
# --------------------------------------------------------------------------
def synth_depth_m(t: float, width: int = 100, height: int = 100) -> np.ndarray:
    """Rear view: back wall ~2.3 m, floor, and a person walking 0.8..2.0 m away."""
    v, u = np.mgrid[0:height, 0:width].astype(np.float32)
    d = np.full((height, width), 2.3, np.float32)
    # floor (lower part of image gets closer the lower the row)
    floor = v > height * 0.55
    d = np.where(floor, np.clip(0.35 * height / np.maximum(v - height * 0.45, 1.0), 0.3, 2.3), d)
    dist = 1.4 + 0.6 * math.sin(t * 0.5)
    cx = width / 2.0 + 25.0 * math.sin(t * 0.8)
    w_px = 40.0 / dist
    person = (np.abs(u - cx) < w_px / 2.0) & (v > height * 0.1) & (v < height * 0.95)
    d = np.where(person, dist, d)
    return d


def fake_serial_stream(n_frames: int, unit: int = 0, garbage: bytes = b"",
                       chunk: int = 0, t0: float = 0.0, dt: float = 0.1,
                       noise_bytes: bytes = b"", first_id: int = 0) -> Iterator[bytes]:
    """Byte stream identical to the real device: optional leading garbage,
    then n_frames packets, optionally split into `chunk`-sized pieces."""
    data = bytearray(garbage)
    for i in range(n_frames):
        codes = mm_to_code(synth_depth_m(t0 + i * dt) * 1000.0, unit)
        data += encode_frame(codes, frame_id=first_id + i)
        data += noise_bytes
    if chunk <= 0:
        yield bytes(data)
        return
    for k in range(0, len(data), chunk):
        yield bytes(data[k:k + chunk])
