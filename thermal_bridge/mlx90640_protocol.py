"""MLXF v1 -- binary MLX90640 frame protocol (our firmware -> host over USB CDC).

Pure module (numpy only), unit-tested in tests/test_mlx90640_protocol.py.
Firmware side: firmware/mlx90640_usb/src/mlxf_protocol.h (keep in sync).

Packet, all fields little-endian::

    off  size  field
     0    4    magic b"MLXF"
     4    1    version = 1
     5    1    sensor_id   (0 = narrow, 1 = wide)
     6    2    seq         (u16 per sensor, wraps)
     8    4    mcu_millis  (u32)
    12    4    ta          (f32, sensor ambient degC)
    16    2    n = 768
    18   2n    int16 temperatures, centi-degC, row-major 24 x 32 (sensor order);
               -32768 = invalid pixel (-> NaN)
  18+2n   2    CRC16-CCITT (poly 0x1021, init 0xFFFF, no reflection, no xorout)
               over bytes [4, 18+2n) = everything after the magic

Text lines from the MCU (command replies, INFO JSON) start with ``#`` and end
with ``\\n``. They are written between packets only. The parser resyncs on the
magic and validates header + CRC, so binary payload bytes that look like text
(or like a magic) cannot desynchronise the stream.
"""
from __future__ import annotations

import binascii
import struct
from typing import Any, Callable, Dict, List, Optional

import numpy as np

MAGIC = b"MLXF"
VERSION = 1
ROWS, COLS = 24, 32
NPIX = ROWS * COLS
HEADER = struct.Struct("<4sBBHIfH")  # magic, version, sensor_id, seq, millis, ta, n
HEADER_LEN = HEADER.size  # 18
PACKET_LEN = HEADER_LEN + 2 * NPIX + 2  # 1556
INVALID_CENTI = -32768
MAX_TEXT_LINE = 1024
MAX_SENSOR_ID = 7


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC16-CCITT (a.k.a. CCITT-FALSE): poly 0x1021, init 0xFFFF. '123456789' -> 0x29B1."""
    return binascii.crc_hqx(bytes(data), crc)


def centi_to_celsius(raw: np.ndarray) -> np.ndarray:
    """int16 centi-degC -> float32 degC, INVALID_CENTI -> NaN."""
    raw = np.asarray(raw, dtype=np.int16)
    out = raw.astype(np.float32) / np.float32(100.0)
    out[raw == INVALID_CENTI] = np.nan
    return out


def celsius_to_centi(temp_c: np.ndarray) -> np.ndarray:
    t = np.asarray(temp_c, dtype=np.float64)
    c = np.clip(np.round(np.nan_to_num(t, nan=0.0) * 100.0), -32767, 32767).astype("<i2")
    c[np.isnan(t)] = INVALID_CENTI
    return c


def encode_packet(sensor_id: int, seq: int, mcu_millis: int, ta: float, temp_c: np.ndarray) -> bytes:
    """Build a byte-exact MLXF v1 packet (used by the mock, the fake device and tests)."""
    t = np.asarray(temp_c, dtype=np.float32).reshape(-1)
    if t.size != NPIX:
        raise ValueError("need %d temperatures, got %d" % (NPIX, t.size))
    body = HEADER.pack(MAGIC, VERSION, sensor_id & 0xFF, seq & 0xFFFF, mcu_millis & 0xFFFFFFFF,
                       float(ta), NPIX) + celsius_to_centi(t).tobytes()
    return body + struct.pack("<H", crc16_ccitt(body[4:]))


class MlxPacket(object):
    __slots__ = ("sensor_id", "seq", "mcu_millis", "ta", "temp_c")

    def __init__(self, sensor_id: int, seq: int, mcu_millis: int, ta: float, temp_c: np.ndarray):
        self.sensor_id = sensor_id
        self.seq = seq
        self.mcu_millis = mcu_millis
        self.ta = ta
        self.temp_c = temp_c  # float32 (24, 32) degC, sensor orientation

    def __repr__(self) -> str:  # pragma: no cover
        return "MlxPacket(sensor=%d seq=%d ta=%.2f)" % (self.sensor_id, self.seq, self.ta)


def decode_packet(buf: bytes) -> MlxPacket:
    """Decode exactly one complete, CRC-valid packet (raises ValueError otherwise)."""
    if len(buf) < PACKET_LEN:
        raise ValueError("short packet: %d bytes" % len(buf))
    magic, ver, sid, seq, millis, ta, n = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or ver != VERSION or n != NPIX:
        raise ValueError("bad header")
    end = HEADER_LEN + 2 * n
    (crc,) = struct.unpack_from("<H", buf, end)
    if crc != crc16_ccitt(bytes(buf[4:end])):
        raise ValueError("crc mismatch")
    raw = np.frombuffer(bytes(buf[HEADER_LEN:end]), dtype="<i2")
    return MlxPacket(sid, seq, millis, ta, centi_to_celsius(raw).reshape(ROWS, COLS))


class ParserStats(object):
    def __init__(self) -> None:
        self.packets = 0
        self.crc_errors = 0
        self.header_errors = 0
        self.bytes_dropped = 0
        self.text_lines = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


class StreamParser(object):
    """Incremental MLXF parser: arbitrary chunking, garbage, '#' text lines.

    ``feed(chunk)`` returns the complete packets found; text lines (without the
    leading '#') go to ``on_text`` if given, and the last 32 are kept in
    ``self.text``.
    """

    def __init__(self, on_text: Optional[Callable[[str], Any]] = None, max_buffer: int = 65536):
        self.buf = bytearray()
        self.stats = ParserStats()
        self.on_text = on_text
        self.text: List[str] = []
        self.max_buffer = max_buffer

    def reset(self) -> None:
        self.buf = bytearray()

    def _text(self, line: bytes) -> None:
        s = line.decode("utf-8", "replace").rstrip("\r")
        self.stats.text_lines += 1
        self.text = (self.text + [s])[-32:]
        if self.on_text is not None:
            self.on_text(s)

    def _drop(self, n: int) -> None:
        del self.buf[:n]
        self.stats.bytes_dropped += n

    def feed(self, data: bytes) -> List[MlxPacket]:
        self.buf += data
        if len(self.buf) > self.max_buffer:  # never happens with a sane stream
            self._drop(len(self.buf) - self.max_buffer)
        out: List[MlxPacket] = []
        buf = self.buf
        while buf:
            if buf[0] == 0x23:  # '#': text line, ends at '\n' (or where a magic starts)
                nl = buf.find(b"\n")
                mg = buf.find(MAGIC)
                if mg != -1 and (nl == -1 or mg < nl):
                    self._text(bytes(buf[1:mg]))
                    del buf[:mg]
                    continue
                if nl == -1:
                    if len(buf) > MAX_TEXT_LINE:
                        self._drop(1)
                        continue
                    break  # wait for the rest of the line
                self._text(bytes(buf[1:nl]))
                del buf[:nl + 1]
                continue
            if buf[:4] == MAGIC:
                if len(buf) < HEADER_LEN:
                    break
                _, ver, sid, _seq, _ms, _ta, n = HEADER.unpack_from(buf, 0)
                if ver != VERSION or n != NPIX or sid > MAX_SENSOR_ID:
                    self.stats.header_errors += 1
                    self._drop(1)
                    continue
                if len(buf) < PACKET_LEN:
                    break
                try:
                    pkt = decode_packet(bytes(buf[:PACKET_LEN]))
                except ValueError:
                    self.stats.crc_errors += 1
                    self._drop(1)
                    continue
                del buf[:PACKET_LEN]
                self.stats.packets += 1
                out.append(pkt)
                continue
            if len(buf) < 4 and MAGIC.startswith(bytes(buf)):
                break  # possible partial magic at the end
            # garbage: skip to the next magic or text line start
            cands = [i for i in (buf.find(MAGIC, 1), buf.find(b"#", 1)) if i != -1]
            if cands:
                self._drop(min(cands))
            else:
                self._drop(max(1, len(buf) - 3))  # keep a possible partial magic
        return out


# --------------------------------------------------------------------------
# Orientation (applied on the host, per camera)
# --------------------------------------------------------------------------
def orient(frame: np.ndarray, flip_h: bool = False, flip_v: bool = False, rotate: int = 0) -> np.ndarray:
    """Sensor order -> image order: flips first, then clockwise rotation (0/90/180/270).

    90/270 swap the shape to (32, 24). Returns a C-contiguous float32 copy.
    """
    a = frame
    if flip_h:
        a = a[:, ::-1]
    if flip_v:
        a = a[::-1, :]
    rot = int(rotate) % 360
    if rot not in (0, 90, 180, 270):
        raise ValueError("rotate must be 0/90/180/270, got %r" % rotate)
    if rot:
        a = np.rot90(a, k=-rot // 90)
    return np.ascontiguousarray(a, dtype=np.float32)


# --------------------------------------------------------------------------
# Text commands (host -> MCU)
# --------------------------------------------------------------------------
REFRESH_RATES_HZ = (0.5, 1, 2, 4, 8, 16, 32, 64)


def cmd_rate(hz: float) -> bytes:
    if not any(abs(hz - r) < 1e-6 for r in REFRESH_RATES_HZ):
        raise ValueError("refresh rate must be one of %s" % (REFRESH_RATES_HZ,))
    return ("RATE %g\n" % hz).encode("ascii")


def cmd_emissivity(e: float) -> bytes:
    if not 0.1 <= e <= 1.0:
        raise ValueError("emissivity must be 0.1 .. 1.0")
    return ("EMIS %.3f\n" % e).encode("ascii")


CMD_INFO = b"INFO?\n"
