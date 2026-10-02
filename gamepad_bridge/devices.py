"""Input devices: python-evdev backend (lazy import), fake devices for tests,
and a scripted virtual pad for --mock. All expose the same small interface:

  backend.list() -> [PadInfo]            (gamepad candidates + siblings)
  backend.open(info) -> PadDevice
  PadDevice.fileno() / read() -> [(type, code, value)]  (OSError = gone)
  PadDevice.rumble(strong, weak, ms) -> bool,  .close()
"""
from __future__ import annotations

import collections
import errno
import os
import threading
import time
from typing import Deque, Dict, List, Optional, Sequence, Tuple

try:
    from .profiles import (CODES, EV_ABS, EV_FF, EV_KEY, EV_SYN, FF_RUMBLE, SYN_REPORT, AbsInfo,
                           PadInfo, is_gamepad)
except ImportError:
    from profiles import (CODES, EV_ABS, EV_FF, EV_KEY, EV_SYN, FF_RUMBLE, SYN_REPORT, AbsInfo,
                          PadInfo, is_gamepad)

Event = Tuple[int, int, int]


# --------------------------------------------------------------------------- evdev
class EvdevPad:
    def __init__(self, info: PadInfo):
        import evdev                                   # lazy: tests / mock never need it
        self._evdev = evdev
        self.info = info
        self.dev = evdev.InputDevice(info.path)
        self._effect_id: Optional[int] = None
        self.can_rumble = FF_RUMBLE in info.caps.get(EV_FF, set())

    def fileno(self) -> int:
        return self.dev.fd

    def read(self) -> List[Event]:
        try:
            return [(e.type, e.code, e.value) for e in self.dev.read()]
        except BlockingIOError:
            return []
        # OSError(ENODEV) on unplug / BT drop propagates to the caller

    def rumble(self, strong: float, weak: float, ms: int) -> bool:
        if not self.can_rumble:
            return False
        try:
            ff = self._evdev.ff
            eff = ff.Effect(FF_RUMBLE, -1, 0, ff.Trigger(0, 0), ff.Replay(int(ms), 0),
                            ff.EffectType(ff_rumble_effect=ff.Rumble(
                                strong_magnitude=int(0xFFFF * strong), weak_magnitude=int(0xFFFF * weak))))
            if self._effect_id is not None:
                try:
                    self.dev.erase_effect(self._effect_id)
                except OSError:
                    pass
            self._effect_id = self.dev.upload_effect(eff)
            self.dev.write(EV_FF, self._effect_id, 1)
            return True
        except Exception:
            self.can_rumble = False                    # driver refused: don't retry
            return False

    def close(self) -> None:
        try:
            self.dev.close()
        except Exception:
            pass


class EvdevBackend:
    def list(self) -> List[PadInfo]:
        import evdev
        out = []
        for path in evdev.list_devices():
            try:
                d = evdev.InputDevice(path)
            except OSError:
                continue
            try:
                caps_raw = d.capabilities(absinfo=True)
                caps: Dict[int, set] = {}
                absinfo: Dict[int, AbsInfo] = {}
                for t, lst in caps_raw.items():
                    s = set()
                    for item in lst:
                        if isinstance(item, tuple):
                            c, ai = item
                            s.add(c)
                            if t == EV_ABS:
                                absinfo[c] = AbsInfo(ai.min, ai.max, ai.flat, ai.fuzz, ai.value)
                        else:
                            s.add(item)
                    caps[t] = s
                out.append(PadInfo(name=d.name or "", vendor=d.info.vendor, product=d.info.product,
                                   path=path, uniq=d.uniq or "", caps=caps, absinfo=absinfo))
            finally:
                d.close()
        return out

    def open(self, info: PadInfo) -> EvdevPad:
        return EvdevPad(info)


# --------------------------------------------------------------------------- fakes
class FakePad:
    """uinput-free stand-in for an evdev device: push events, read them back,
    `unplug()` makes the next read raise OSError(ENODEV) like a BT drop."""

    def __init__(self, info: PadInfo):
        self.info = info
        self.queue: Deque[Event] = collections.deque()
        self.gone = False
        self.closed = False
        self.rumbles: List[Tuple[float, float, int]] = []
        self.can_rumble = FF_RUMBLE in info.caps.get(EV_FF, set())
        self.lock = threading.Lock()

    def push(self, *events: Event, syn: bool = True) -> None:
        with self.lock:
            self.queue.extend(events)
            if syn:
                self.queue.append((EV_SYN, SYN_REPORT, 0))

    def fileno(self) -> int:
        return -1

    def read(self) -> List[Event]:
        if self.gone:
            raise OSError(errno.ENODEV, "No such device")
        with self.lock:
            out = list(self.queue)
            self.queue.clear()
        return out

    def rumble(self, strong: float, weak: float, ms: int) -> bool:
        if not self.can_rumble:
            return False
        self.rumbles.append((strong, weak, ms))
        return True

    def unplug(self) -> None:
        self.gone = True

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Hotplug simulation: add()/remove() PadInfos; open() returns FakePad."""

    def __init__(self, infos: Sequence[PadInfo] = ()):
        self.infos: List[PadInfo] = list(infos)
        self.pads: Dict[str, FakePad] = {}

    def add(self, info: PadInfo) -> None:
        self.infos.append(info)

    def remove(self, path: str) -> None:
        self.infos = [i for i in self.infos if i.path != path]
        if path in self.pads:
            self.pads[path].unplug()

    def list(self) -> List[PadInfo]:
        return list(self.infos)

    def open(self, info: PadInfo) -> FakePad:
        p = FakePad(info)
        self.pads[info.path] = p
        return p


def make_info(name: str, vendor: int = 0, product: int = 0, path: str = "/dev/input/event99",
              abs_ranges: Optional[Dict[str, Tuple[int, int]]] = None, keys: Sequence[str] = (),
              uniq: str = "", rumble: bool = False, flat: int = 0) -> PadInfo:
    """Synthetic PadInfo (absinfo + caps) for tests and the mock pad."""
    absinfo: Dict[int, AbsInfo] = {}
    for n, (lo, hi) in (abs_ranges or {}).items():
        c = CODES[n]
        rest = (lo + hi) // 2 if n not in ("ABS_Z", "ABS_RZ", "ABS_GAS", "ABS_BRAKE") or lo < 0 else lo
        absinfo[c] = AbsInfo(lo, hi, flat if n in ("ABS_X", "ABS_Y", "ABS_RX", "ABS_RY") else 0, 0, rest)
    caps = {EV_ABS: set(absinfo), EV_KEY: {CODES[k] for k in keys}}
    if rumble:
        caps[EV_FF] = {FF_RUMBLE}
    return PadInfo(name=name, vendor=vendor, product=product, path=path, uniq=uniq, caps=caps, absinfo=absinfo)


XBOX_KEYS = ("BTN_SOUTH", "BTN_EAST", "BTN_X", "BTN_Y", "BTN_TL", "BTN_TR", "BTN_SELECT", "BTN_START",
             "BTN_MODE", "BTN_THUMBL", "BTN_THUMBR")
XBOX_ABS = {"ABS_X": (-32768, 32767), "ABS_Y": (-32768, 32767), "ABS_RX": (-32768, 32767),
            "ABS_RY": (-32768, 32767), "ABS_Z": (0, 1023), "ABS_RZ": (0, 1023),
            "ABS_HAT0X": (-1, 1), "ABS_HAT0Y": (-1, 1)}


def mock_xbox_info(path: str = "/dev/input/mock0") -> PadInfo:
    return make_info("Xbox Wireless Controller", 0x045E, 0x0B13, path, XBOX_ABS, XBOX_KEYS,
                     uniq="aa:bb:cc:dd:ee:ff", rumble=True)


# --------------------------------------------------------------------------- scripted mock
def _stick(v: float) -> int:
    return int(max(-1.0, min(1.0, v)) * 32767)


B = CODES
# (t_offset_s, events). Logical "up" = negative raw ABS_Y.
DEFAULT_SCRIPT: List[Tuple[float, List[Event]]] = [
    (1.0, [(EV_KEY, B["BTN_TR"], 1)]),                                   # dead-man on
    (1.5, [(EV_ABS, B["ABS_Y"], _stick(-0.6))]),                         # forward
    (2.0, [(EV_ABS, B["ABS_RX"], _stick(-0.5))]),                        # + turn left
    (3.0, [(EV_ABS, B["ABS_RX"], 0)]),
    (3.5, [(EV_ABS, B["ABS_Y"], 0)]),                                    # stick released -> /stop
    (4.0, [(EV_KEY, B["BTN_TR"], 0)]),                                   # dead-man off
    (4.5, [(EV_ABS, B["ABS_Y"], _stick(-0.8))]),                         # no RB: denied
    (5.0, [(EV_ABS, B["ABS_Y"], 0)]),
    (5.5, [(EV_KEY, B["BTN_X"], 1)]), (5.6, [(EV_KEY, B["BTN_X"], 0)]),  # capture
    (6.0, [(EV_ABS, B["ABS_RZ"], 1023)]), (6.2, [(EV_ABS, B["ABS_RZ"], 0)]),   # level up (no-op at top)
    (6.5, [(EV_ABS, B["ABS_Z"], 1023)]), (6.7, [(EV_ABS, B["ABS_Z"], 0)]),     # level down -> stealth
    (7.0, [(EV_KEY, B["BTN_TR"], 1), (EV_ABS, B["ABS_X"], _stick(0.7))]),     # strafe right
    (8.0, [(EV_KEY, B["BTN_EAST"], 1)]), (8.1, [(EV_KEY, B["BTN_EAST"], 0)]), # B = stop (latched)
    (8.5, [(EV_ABS, B["ABS_X"], 0), (EV_KEY, B["BTN_TR"], 0)]),
    (9.0, [(EV_KEY, B["BTN_START"], 1), (EV_KEY, B["BTN_SELECT"], 1)]),      # E-STOP combo
    (11.5, [(EV_KEY, B["BTN_START"], 0), (EV_KEY, B["BTN_SELECT"], 0)]),
    (12.0, [(EV_ABS, B["ABS_RZ"], 1023)]), (12.2, [(EV_ABS, B["ABS_RZ"], 0)]), # back to normal
]


class ScriptedPadFeeder(threading.Thread):
    """Plays a script into a FakePad; loops every `period_s`. The virtual pad
    also emits SYN every 50 ms (like a pad with an IMU stream) so a held stick
    stays 'live'."""

    def __init__(self, backend: FakeBackend, info: PadInfo, script=None, period_s: float = 14.0,
                 clock=time.time):
        super().__init__(daemon=True, name="mockpad")
        self.backend, self.info = backend, info
        self.script = script or DEFAULT_SCRIPT
        self.period_s = period_s
        self.clock = clock
        self.stop_ev = threading.Event()

    def run(self) -> None:
        while not self.stop_ev.is_set():
            t0 = self.clock()
            i = 0
            while not self.stop_ev.is_set() and self.clock() - t0 < self.period_s:
                pad = self.backend.pads.get(self.info.path)
                el = self.clock() - t0
                while i < len(self.script) and self.script[i][0] <= el:
                    if pad:
                        pad.push(*self.script[i][1])
                    i += 1
                if pad:
                    pad.push(syn=True)
                self.stop_ev.wait(0.05)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")
