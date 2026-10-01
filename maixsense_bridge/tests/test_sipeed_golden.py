"""Golden tests: Sipeed's own A010 host code, transliterated line by line to Python, must give
exactly the same results as our protocol.py on the same byte streams.

Vendor sources (pinned):
  ROS  = github.com/sipeed/MaixSense_ROS @34c2facb sipeed_tof_ms_a010_ros/ros2/src/{frame_handle.cc,main.cc}
  HOST = github.com/sipeed/sipeed_wiki docs/hardware/zh/maixsense/maixsense-a010/code.md (tof_main_host.py)
  TOOL = github.com/sipeed/MetaSense-ComTool @1ba26875 COMTool/plugins/gragh_widgets.py (Gragh_MetaSenseLite)
No sample captures exist in these repos, so streams are built with protocol.encode_frame.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from maixsense_bridge import protocol as P


# ----------------------------------------------------------------------------------------------
# Vendor transliterations
# ----------------------------------------------------------------------------------------------
class RosHandleProcess(object):
    """ros2/src/frame_handle.cc handle_process() incl. its quirks (needs 1 extra byte after the
    tail before a frame is released, payload cap 100*100, uint8 checksum accumulation)."""
    FRAME_HEAD_SIZE, FRAME_HEAD_DATA_SIZE, CS, END = 20, 16, 1, 1

    def __init__(self):
        self.v = bytearray()

    def __call__(self, s: bytes):
        v = self.v
        v += s
        if len(v) < 2:
            return None
        while True:
            # __find_header: find 0xFF from [1:] with previous byte 0x00
            it = 0
            while True:
                try:
                    it = v.index(0xFF, it + 1)
                except ValueError:
                    del v[:-1]                     # keep last element (may be sflag_l)
                    return None
                if v[it - 1] == 0x00:
                    break
            if it - 1 != 0:
                del v[:it - 1]
                it = 1
            if len(v) < 20:                         # sizeof(frame_t)
                return None
            data_len = struct.unpack_from("<H", v, 2)[0]
            payload_len = (data_len - 16) & 0xFFFFFFFF  # uint32 wraparound like the C code
            if payload_len > 100 * 100:
                # ros2 does `goto __find_header` without consuming anything, so it re-finds the
                # same header forever (vendor bug: hangs on a 00 FF lookalike with a big length).
                # ros1 frame_handle.cc pops bytes instead; we drop the 0x00 so the search moves on.
                del v[:it]
                continue
            if self.FRAME_HEAD_SIZE + payload_len + self.CS + self.END + 1 > len(v):
                return None
            cs = 0
            for i in range(self.FRAME_HEAD_SIZE + payload_len):
                cs = (cs + v[i]) & 0xFF
            if cs != v[20 + payload_len] or v[21 + payload_len] != 0xDD:
                del v[:it]                          # vector(it, end) -> drops the 0x00
                continue
            frame = bytes(v[:20 + payload_len])
            del v[:it + 20 + payload_len + 1 + 1 - 1]
            return frame


def host_relay(stream: bytes, last_frameid: int = 0):
    """code.md tof_main_host.py relay_thread() body (lines ~106-198)."""
    out = []
    buf = bytearray(stream)
    while True:
        idx = buf.find(b"\x00\xFF")
        if idx < 0:
            break
        if idx > 0:
            del buf[:idx]
        if len(buf) < 4:
            break
        dataLen = struct.unpack("<H", buf[2:4])[0]
        frameLen = 2 + 2 + dataLen + 2
        if len(buf) < frameLen:
            break
        frame = bytes(buf[:frameLen])
        del buf[:frameLen]
        if frame[-1] not in (0xCC, 0xDD) or frame[-2] != sum(frame[:-2]) & 0xFF:
            continue
        resR, resC = frame[14], frame[15]
        frameid = struct.unpack("<H", frame[16:18])[0]
        data_end = 20 + dataLen - 16
        if data_end > len(frame) - 2:
            continue
        if frameid == last_frameid:
            continue
        last_frameid = frameid
        payload = frame[20:data_end]
        if len(payload) != resR * resC:             # processor_thread size check
            continue
        out.append((frameid, resR, resC, payload))
    return out


def comtool_decode(stream: bytes):
    """TOOL Gragh_MetaSenseLite.decodeData (L969-1046), called until it stops yielding."""
    raw = stream
    out = []
    while True:
        idx = raw.find(b"\x00\xFF")
        if idx < 0:
            break
        raw = raw[idx:]
        if len(raw) < 4:
            break
        dataLen = struct.unpack("H", raw[2:4])[0]
        frameLen = 2 + 2 + dataLen + 2
        frameDataLen = dataLen - 16
        if len(raw) < frameLen:
            break
        frame = raw[:frameLen]
        raw = raw[frameLen:]
        if frame[-1] != 0xDD and frame[-2] != sum(frame[:frameLen - 2]) % 256:  # (sic: 'and')
            continue
        fid = struct.unpack("H", frame[16:18])[0]
        res = (frame[14], frame[15])
        data = bytes(frame[20:20 + frameDataLen])
        out.append((fid, res, data))
    return out


def comtool_distance_mm(val: int, unit: int) -> float:
    """TOOL compute_real_distance (L1100-1106) without the 8-sample averaging."""
    if unit != 0:
        return val * unit
    r = int(val) / 5.1
    return r * r


def ros_coeff(reply1: bytes, reply2: bytes):
    """ROS main.cc L69-94: first read must equal '+COEFF=1\\r\\nOK\\r\\n', JSON in the next read,
    cJSON valueint / 262144.0f."""
    assert reply1.decode() == "+COEFF=1\r\nOK\r\n"
    d = json.loads(reply2.decode())
    return tuple(float(np.float32(int(d[k])) / np.float32(262144.0)) for k in ("fx", "fy", "u0", "v0"))


def ros_cloud(depth_u8: np.ndarray, K):
    """ROS main.cc L190-200 (float32 math). Returns (rows*cols, 3) incl. zero points."""
    fox, foy, u0, v0 = (np.float32(k) for k in K)
    rows, cols = depth_u8.shape
    pts = np.zeros((rows * cols, 3), np.float32)
    n = 0
    for j in range(rows):
        for i in range(cols):
            cx = (np.float32(i) - u0) / fox
            cy = (np.float32(j) - v0) / foy
            dst = np.float32(depth_u8[j, i]) / np.float32(1000)
            pts[n] = (dst * cx, dst * cy, dst)
            n += 1
    return pts


# ----------------------------------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------------------------------
def _stream(seed=1, binns=(1, 2, 4, 1), corrupt=(2,), garbage=True):
    rng = np.random.default_rng(seed)
    data = bytearray(b"+COEFF=1\r\nOK\r\n" if garbage else b"")
    frames = []
    for k, b in enumerate(binns):
        rows, cols = P.BINN_SHAPE[b]
        c = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
        c[0, :2] = (0x00, 0xFF)                      # header lookalike inside the payload
        pkt = P.encode_frame(c, frame_id=100 + k, corrupt_checksum=(k in corrupt),
                             sensor_temp=int(rng.integers(0, 255)), exposure_time=int(rng.integers(0, 1 << 20)))
        if k not in corrupt:
            frames.append((100 + k, rows, cols, c.tobytes()))
        data += pkt
        if garbage:
            data += b"\x00\x13\xff\xddOK\r\n"
    return bytes(data), frames


def _ours(stream, chunk=0):
    p = P.FrameParser()
    out = []
    chunks = [stream] if chunk <= 0 else [stream[i:i + chunk] for i in range(0, len(stream), chunk)]
    for ch in chunks:
        out += p.feed(ch)
    return [(f.frame_id, f.rows, f.cols, f.depth.tobytes()) for f in out], p


@pytest.mark.parametrize("chunk", [0, 1, 97, 4096])
def test_golden_vs_ros_handle_process(chunk):
    stream, expected = _stream()
    stream += b"\x00"   # vendor needs one byte after the last tail before releasing it
    ros = RosHandleProcess()
    got_ros = []
    chunks = [stream] if chunk <= 0 else [stream[i:i + chunk] for i in range(0, len(stream), chunk)]
    for ch in chunks:
        f = ros(ch)
        while f is not None:                         # drain like timer_callback's goto _more
            got_ros.append(f)
            f = ros(b"")
    ros_frames = [(struct.unpack_from("<H", f, 16)[0] & 0x0FFF, f[14], f[15], f[20:]) for f in got_ros]
    ours, p = _ours(stream, chunk)
    assert ros_frames == ours == expected
    assert p.stats.checksum_errors >= 1


def test_golden_header_fields_vs_ros_struct():
    c = np.arange(2500, dtype=np.uint32).astype(np.uint8).reshape(50, 50)
    pkt = P.encode_frame(c, frame_id=4095, sensor_temp=41, driver_temp=43, exposure_time=0x00ABCDEF,
                         error_code=3, isp_version=0x23)
    # frame_head_t, __attribute__((packed)), little-endian (frame_struct.h L17-32)
    (beg, dlen, r1, mode, st, dt, e0, e1, e2, e3, err, r2, rows, cols, fid, isp, r3) = struct.unpack_from(
        "<HHBBBB4BBBBBHBB", pkt, 0)
    assert beg == 0xFF00 and dlen == 16 + 2500 and (r1, r2, r3) == (0xFF, 0x00, 0xFF)
    assert (mode, st, dt, err, rows, cols, fid, isp) == (0, 41, 43, 3, 50, 50, 4095, 0x23)
    f = P.FrameParser().feed(pkt)[0]
    assert (f.sensor_temp, f.driver_temp, f.error_code, f.isp_version, f.frame_id) == (41, 43, 3, 0x23, 4095)
    assert f.exposure_time == e0 | e1 << 8 | e2 << 16 | e3 << 24


def test_golden_vs_host_example_and_cc_tail():
    stream, expected = _stream(seed=3)
    rows = np.full((25, 25), 77, np.uint8)
    cc = P.encode_frame(rows, frame_id=200, tail=0xCC)     # UART/SPI-style tail accepted by HOST
    stream += cc
    expected = expected + [(200, 25, 25, rows.tobytes())]
    assert host_relay(stream) == expected
    ours, p = _ours(stream)
    assert ours == expected
    assert P.FrameParser(tails=(0xDD,)).feed(cc) == []       # ROS/ComTool-strict mode


def test_golden_vs_comtool_on_clean_stream():
    stream, expected = _stream(seed=5, corrupt=(), garbage=False)
    tool = [(fid, res[0], res[1], data) for fid, res, data in comtool_decode(stream)]
    ours, _ = _ours(stream)
    assert tool == ours == expected


@pytest.mark.parametrize("unit", [0, 1, 4, 10])
def test_golden_depth_conversion_vs_comtool(unit):
    codes = np.arange(256, dtype=np.uint8)
    ref = np.array([comtool_distance_mm(int(v), unit) for v in codes])
    assert np.allclose(P.depth_mm(codes, unit), ref, rtol=1e-6, atol=1e-3)


def test_golden_coeff_vs_ros():
    ack, js = P.encode_coeff_reply(75.123, 74.9, 49.37, 50.81)
    ref = ros_coeff(ack, js)
    assert P.parse_coeff((ack + js).decode()) == ref          # one buffer (our read path)
    assert P.parse_coeff(js.decode()) == ref                  # JSON alone
    assert ref == pytest.approx((75.123, 74.9, 49.37, 50.81), abs=1e-5)
    # vendor-looking full LensCoeff_t JSON with distortion fields
    full = '{"cali_mode":0,"fx":19005235,"fy":19005235,"u0":12976128,"v0":13107200,"k1":-1,"k2":0,"k3":0,"k4_p1":0,"k5_p2":0,"skew":0}'
    assert P.parse_coeff(full) == ros_coeff(ack, full.encode())


def test_golden_point_cloud_vs_ros():
    rng = np.random.default_rng(7)
    codes = rng.integers(1, 255, size=(100, 100), dtype=np.uint8)
    K = (75.1, 74.8, 49.3, 50.2)
    ref = ros_cloud(codes, K)
    # vendor feeds raw code/1000 as metres (ignores UNIT): give ours the same depth to compare geometry
    ours = P.depth_to_points(codes.astype(np.float32) / np.float32(1000), K)
    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, rtol=1e-5, atol=1e-7)
    # z is the A010 value itself (z-depth), not range along the ray
    assert np.array_equal(ours[:, 2], ref[:, 2])
