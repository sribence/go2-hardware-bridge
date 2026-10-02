#!/usr/bin/env python3
"""Fake Sipeed MaixSense A010 on a pseudo-terminal (no hardware needed).

Behaves like the firmware as seen by Sipeed's own host code (MaixSense_ROS main.cc / node.cc,
MetaSense-ComTool gragh_widgets.py, sipeed_wiki code.md):
  * AT commands end with '\\r'; no echo.  `AT` -> 'OK\\r\\n'; `AT+X=v` -> 'OK\\r\\n' (out of range ->
    'ERROR\\r\\n', ASSUMED); `AT+X?` -> '+X=v\\r\\nOK\\r\\n'.
  * `AT+COEFF?` -> '+COEFF=1\\r\\nOK\\r\\n', then (separate write, ~20 ms later) a JSON object with
    u14p18 fixed-point fx, fy, u0, v0 (the vendor ROS node reads them in two reads).
  * Streams 0x00 0xFF ... 0xDD packets while ISP=1 and DISP has the USB bit (2), at AT+FPS,
    resolution per AT+BINN, depth codes per AT+UNIT.  Power-on state: ISP=1, DISP=1 (LCD only,
    ASSUMED), BINN=1, UNIT=0, FPS=15 (ComTool default label).
  * unplug()/replug(): closes the pty master (host read fails like a USB unplug) and removes the
    symlink; replug creates a new pty and re-points the symlink (like udev's /dev/maixsense).

CLI:  python3 tools/a010_fake_device.py --link /tmp/maixsense [--unplug-every 10 --replug-after 2]
Then: python3 maixsense_bridge/maixsense_bridge.py --device /tmp/maixsense
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import tty
from typing import Dict, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from maixsense_bridge import protocol as P  # noqa: E402


class FakeA010(object):
    def __init__(self, link: Optional[str] = None, fx: float = 75.0, fy: float = 75.0,
                 u0: float = 49.5, v0: float = 49.5, coeff_delay_s: float = 0.02,
                 garbage_every: int = 0, corrupt_every: int = 0):
        self.link = link
        self.coeff = (fx, fy, u0, v0)
        self.coeff_delay_s = coeff_delay_s
        self.garbage_every = garbage_every
        self.corrupt_every = corrupt_every
        self.state: Dict[str, int] = {"ISP": 1, "DISP": P.DISP_LCD, "BINN": 1, "UNIT": 0, "FPS": 15,
                                      "BAUD": 2, "ANTIMMI": 0, "AE": 1, "EV": 0}
        self.master = -1
        self.slave = -1
        self.slave_path = ""
        self.frames_sent = 0
        self.commands = []          # every command line received (for tests)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._fid = 0
        self.t0 = time.time()

    # ---- pty lifecycle -------------------------------------------------
    def plug(self) -> str:
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        self.slave_path = os.ttyname(self.slave)
        if self.link:
            tmp = self.link + ".tmp"
            if os.path.lexists(tmp):
                os.unlink(tmp)
            os.symlink(self.slave_path, tmp)
            os.replace(tmp, self.link)
        return self.slave_path

    def unplug(self) -> None:
        with self._lock:
            if self.link and os.path.lexists(self.link):
                os.unlink(self.link)
            for fd in (self.master, self.slave):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            self.master = self.slave = -1
            # power cycle: firmware state back to defaults
            self.state.update({"ISP": 1, "DISP": P.DISP_LCD})

    def replug(self) -> str:
        with self._lock:
            return self.plug()

    def start(self) -> "FakeA010":
        if self.master < 0:
            self.plug()
        for target in (self._rx_loop, self._tx_loop):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
        self.unplug()

    # ---- I/O -------------------------------------------------------------
    def _write(self, data: bytes) -> None:
        fd = self.master
        if fd < 0:
            return
        try:
            os.write(fd, data)
        except OSError:
            pass

    def _rx_loop(self) -> None:
        import select

        buf = b""
        while not self._stop.is_set():
            fd = self.master
            if fd < 0:
                time.sleep(0.01)
                buf = b""
                continue
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                data = os.read(fd, 1024)
            except (OSError, ValueError):   # slave side closed / fd swapped by replug
                time.sleep(0.01)
                continue
            buf += data
            while b"\r" in buf:
                line, buf = buf.split(b"\r", 1)
                line = line.strip(b"\n ").decode("ascii", "ignore")
                if line:
                    self._handle(line)

    def _handle(self, line: str) -> None:
        self.commands.append(line)
        up = line.upper()
        if up == "AT":
            return self._write(b"OK\r\n")
        if not up.startswith("AT+"):
            return self._write(b"ERROR\r\n")
        body = up[3:]
        if body == "COEFF?":
            ack, js = P.encode_coeff_reply(*self.coeff)
            self._write(ack)
            time.sleep(self.coeff_delay_s)
            return self._write(js)
        if body.endswith("=?"):
            name = body[:-2]
            rng = P.AT_RANGES.get(name)
            if rng is None:
                return self._write(b"ERROR\r\n")
            return self._write(("+%s=%s\r\nOK\r\n" % (name, ",".join(str(v) for v in rng))).encode())
        if body.endswith("?"):
            name = body[:-1]
            if name not in self.state:
                return self._write(b"ERROR\r\n")
            return self._write(("+%s=%d\r\nOK\r\n" % (name, self.state[name])).encode())
        if "=" in body:
            name, val = body.split("=", 1)
            try:
                v = int(val)
            except ValueError:
                return self._write(b"ERROR\r\n")
            if name not in P.AT_RANGES or v not in P.AT_RANGES[name]:
                return self._write(b"ERROR\r\n")
            self.state[name] = v
            return self._write(b"OK\r\n")
        return self._write(b"ERROR\r\n")

    def _tx_loop(self) -> None:
        next_t = time.time()
        while not self._stop.is_set():
            st = self.state
            period = 1.0 / max(1, st["FPS"])
            now = time.time()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t = max(next_t + period, now)
            if self.master < 0 or not st["ISP"] or not (st["DISP"] & P.DISP_USB):
                continue
            rows, cols = P.BINN_SHAPE[st["BINN"]]
            codes = P.mm_to_code(P.synth_depth_m(now - self.t0, cols, rows) * 1000.0, st["UNIT"])
            self.frames_sent += 1
            corrupt = bool(self.corrupt_every) and self.frames_sent % self.corrupt_every == 0
            pkt = P.encode_frame(codes, frame_id=self._fid, corrupt_checksum=corrupt)
            self._fid = (self._fid + 1) & 0x0FFF
            if self.garbage_every and self.frames_sent % self.garbage_every == 0:
                pkt = b"\x00\x13\xff\xdd" + pkt
            self._write(pkt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", default="/tmp/maixsense", help="stable symlink to the pty slave")
    ap.add_argument("--unplug-every", type=float, default=0.0, help="simulate unplug every N s (0 = never)")
    ap.add_argument("--replug-after", type=float, default=2.0)
    ap.add_argument("--corrupt-every", type=int, default=0)
    ap.add_argument("--garbage-every", type=int, default=0)
    a = ap.parse_args()
    dev = FakeA010(link=a.link, corrupt_every=a.corrupt_every, garbage_every=a.garbage_every).start()
    print("fake A010 on %s -> %s" % (a.link, dev.slave_path), flush=True)
    try:
        while True:
            if a.unplug_every > 0:
                time.sleep(a.unplug_every)
                dev.unplug()
                print("unplugged", flush=True)
                time.sleep(a.replug_after)
                print("replugged -> %s" % dev.replug(), flush=True)
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        dev.stop()


if __name__ == "__main__":
    main()
