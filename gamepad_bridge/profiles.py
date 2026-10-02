"""Pad profiles + normalization: raw evdev events -> logical PadState.

Pure (no evdev import). Code names are the real Linux input-event-codes.h names;
the table below holds their numeric values so tests run without python-evdev.

Logical layout (positional, Xbox naming, like the browser "standard" mapping):
  sticks lx, ly, rx, ry in -1..1 (right / UP positive), triggers lt, rt in 0..1,
  dpad_x, dpad_y in {-1, 0, 1} (right / up positive),
  buttons a (south), b (east), x (west), y (north), lb, rb, start, select, home, ls, rs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

EV_SYN, EV_KEY, EV_ABS, EV_FF = 0x00, 0x01, 0x03, 0x15
SYN_REPORT, SYN_DROPPED = 0, 3
FF_RUMBLE = 0x50

CODES: Dict[str, int] = {
    # absolute axes
    "ABS_X": 0x00, "ABS_Y": 0x01, "ABS_Z": 0x02, "ABS_RX": 0x03, "ABS_RY": 0x04, "ABS_RZ": 0x05,
    "ABS_THROTTLE": 0x06, "ABS_RUDDER": 0x07, "ABS_WHEEL": 0x08, "ABS_GAS": 0x09, "ABS_BRAKE": 0x0A,
    "ABS_HAT0X": 0x10, "ABS_HAT0Y": 0x11,
    # joystick buttons (cheap generic HID pads)
    "BTN_TRIGGER": 0x120, "BTN_THUMB": 0x121, "BTN_THUMB2": 0x122, "BTN_TOP": 0x123, "BTN_TOP2": 0x124,
    "BTN_PINKIE": 0x125, "BTN_BASE": 0x126, "BTN_BASE2": 0x127, "BTN_BASE3": 0x128, "BTN_BASE4": 0x129,
    "BTN_BASE5": 0x12A, "BTN_BASE6": 0x12B,
    # gamepad buttons
    "BTN_SOUTH": 0x130, "BTN_A": 0x130, "BTN_GAMEPAD": 0x130,
    "BTN_EAST": 0x131, "BTN_B": 0x131, "BTN_C": 0x132,
    "BTN_NORTH": 0x133, "BTN_X": 0x133, "BTN_WEST": 0x134, "BTN_Y": 0x134, "BTN_Z": 0x135,
    "BTN_TL": 0x136, "BTN_TR": 0x137, "BTN_TL2": 0x138, "BTN_TR2": 0x139,
    "BTN_SELECT": 0x13A, "BTN_START": 0x13B, "BTN_MODE": 0x13C, "BTN_THUMBL": 0x13D, "BTN_THUMBR": 0x13E,
    "BTN_DPAD_UP": 0x220, "BTN_DPAD_DOWN": 0x221, "BTN_DPAD_LEFT": 0x222, "BTN_DPAD_RIGHT": 0x223,
    # non-gamepad (used to reject touchpads / mice)
    "BTN_LEFT": 0x110, "BTN_TOUCH": 0x14A, "BTN_TOOL_FINGER": 0x145,
}

STICKS = ("lx", "ly", "rx", "ry")
TRIGGERS = ("lt", "rt")
BUTTONS = ("a", "b", "x", "y", "lb", "rb", "start", "select", "home", "ls", "rs")
DPAD_BUTTONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def code(c: Union[str, int]) -> int:
    if isinstance(c, int):
        return c
    s = str(c).strip()
    if s in CODES:
        return CODES[s]
    try:
        return int(s, 0)
    except ValueError:
        raise ValueError("unknown evdev code %r" % c)


@dataclass
class AbsInfo:
    min: int = -32768
    max: int = 32767
    flat: int = 0
    fuzz: int = 0
    value: int = 0


Caps = Dict[int, Set[int]]          # {EV_KEY: {codes}, EV_ABS: {codes}, EV_FF: {...}}

GAMEPAD_KEYS = {CODES["BTN_SOUTH"], CODES["BTN_EAST"], CODES["BTN_TRIGGER"], CODES["BTN_THUMB"]}


def is_gamepad(caps: Caps) -> bool:
    """Sticks (ABS_X+ABS_Y) + a gamepad/joystick button. Rejects DS4/DualSense
    touchpads (BTN_TOUCH, no BTN_SOUTH) and motion-sensor nodes (no keys)."""
    keys = caps.get(EV_KEY, set())
    absc = caps.get(EV_ABS, set())
    if not {CODES["ABS_X"], CODES["ABS_Y"]} <= absc:
        return False
    if CODES["BTN_TOUCH"] in keys and CODES["BTN_SOUTH"] not in keys:
        return False
    return bool(keys & GAMEPAD_KEYS)


@dataclass
class PadInfo:
    name: str = ""
    vendor: int = 0
    product: int = 0
    path: str = ""
    uniq: str = ""
    caps: Caps = field(default_factory=dict)
    absinfo: Dict[int, AbsInfo] = field(default_factory=dict)

    @property
    def ids(self) -> str:
        return "%04x:%04x" % (self.vendor, self.product)


class Profile:
    def __init__(self, name: str, d: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None):
        dd = dict(defaults or {})
        self.name = name
        self.label = d.get("label", name)
        m = d.get("match", {}) or {}
        self.names = [re.compile(p, re.I) for p in m.get("names", [])]
        self.ids = [str(i).lower() for i in m.get("ids", [])]
        self.requires = [code(c) for c in m.get("requires", [])]
        self.excludes = [code(c) for c in m.get("excludes", [])]
        self.any_gamepad = bool(m.get("any", False))
        self.deadzone = float(d.get("deadzone", dd.get("deadzone", 0.12)))
        self.trigger_deadzone = float(d.get("trigger_deadzone", dd.get("trigger_deadzone", 0.05)))
        # axes: logical -> (code, invert)
        self.axes: Dict[str, Tuple[int, bool]] = {}
        for k, v in (d.get("axes") or {}).items():
            if isinstance(v, dict):
                self.axes[k] = (code(v["code"]), bool(v.get("invert", False)))
            else:
                self.axes[k] = (code(v), k in ("ly", "ry"))      # evdev Y is down-positive
        self.trig_abs: Dict[str, int] = {}
        self.trig_btn: Dict[str, int] = {}
        for k, v in (d.get("triggers") or {}).items():
            v = v if isinstance(v, dict) else {"abs": v}
            if v.get("abs") is not None:
                self.trig_abs[k] = code(v["abs"])
            if v.get("btn") is not None:
                self.trig_btn[k] = code(v["btn"])
        self.buttons: Dict[int, str] = {code(v): k for k, v in (d.get("buttons") or {}).items()}
        dp = d.get("dpad", {}) or {}
        self.hat = (code(dp.get("hat_x", "ABS_HAT0X")), code(dp.get("hat_y", "ABS_HAT0Y")))
        self.dpad_btn: Dict[int, str] = {code(dp.get(k, "BTN_DPAD_" + k.upper())): k for k in DPAD_BUTTONS}

    def score(self, info: PadInfo) -> int:
        """>0 = matches (higher = more specific), 0 = no match."""
        keys = info.caps.get(EV_KEY, set()) | info.caps.get(EV_ABS, set())
        if any(c not in keys for c in self.requires) or any(c in keys for c in self.excludes):
            return 0
        if self.ids and info.ids in self.ids:
            return 3
        if self.names and any(p.search(info.name or "") for p in self.names):
            return 2
        if self.any_gamepad and is_gamepad(info.caps):
            return 1
        return 0


def build_profiles(doc: Dict[str, Any]) -> List[Profile]:
    defaults = doc.get("defaults", {}) or {}
    return [Profile(n, d, defaults) for n, d in (doc.get("profiles") or {}).items()]


def select_profile(profiles: List[Profile], info: PadInfo, forced: Optional[str] = None) -> Optional[Profile]:
    if forced:
        for p in profiles:
            if p.name == forced:
                return p
    best, best_s = None, 0
    for p in profiles:                       # list order breaks ties (YAML order = priority)
        s = p.score(info)
        if s > best_s:
            best, best_s = p, s
    return best


@dataclass
class PadState:
    axes: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in STICKS + TRIGGERS})
    buttons: Dict[str, bool] = field(default_factory=lambda: {k: False for k in BUTTONS})
    dpad_x: int = 0
    dpad_y: int = 0
    last_event_t: float = 0.0

    def copy(self) -> "PadState":
        return PadState(dict(self.axes), dict(self.buttons), self.dpad_x, self.dpad_y, self.last_event_t)

    def as_dict(self) -> Dict[str, Any]:
        return {"axes": {k: round(v, 3) for k, v in self.axes.items()},
                "buttons": {k: v for k, v in self.buttons.items() if v},
                "dpad": [self.dpad_x, self.dpad_y], "last_event_t": self.last_event_t}


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Normalizer:
    """Feeds raw (type, code, value) events into a PadState using a Profile and
    the device's absinfo (ranges differ per driver: 0..255, +-32767, 0..1023...)."""

    def __init__(self, profile: Profile, absinfo: Dict[int, AbsInfo]):
        self.p = profile
        self.absinfo = absinfo
        self.state = PadState()
        self._abs_map: Dict[int, List[Tuple[str, str, bool]]] = {}
        for k, (c, inv) in profile.axes.items():
            self._abs_map.setdefault(c, []).append(("stick", k, inv))
        for k, c in profile.trig_abs.items():
            self._abs_map.setdefault(c, []).append(("trig", k, False))
        self._trig_btn = {c: k for k, c in profile.trig_btn.items()}
        self._dpad_btns: Dict[str, bool] = {k: False for k in DPAD_BUTTONS}
        for c, ai in absinfo.items():            # initial values (sticks resting off-centre)
            if c in self._abs_map:
                self._abs(c, ai.value)

    def stick_value(self, c: int, raw: int, invert: bool) -> float:
        ai = self.absinfo.get(c) or AbsInfo()
        half = (ai.max - ai.min) / 2.0
        if half <= 0:
            return 0.0
        v = _clip((raw - (ai.max + ai.min) / 2.0) / half, -1.0, 1.0)
        # absinfo.flat is the driver's own dead band (in raw units)
        if abs(v) <= (ai.flat / half if ai.flat else 0.0):
            v = 0.0
        return -v if invert else v

    def trigger_value(self, c: int, raw: int) -> float:
        ai = self.absinfo.get(c) or AbsInfo(0, 255)
        span = float(ai.max - ai.min)
        if span <= 0:
            return 0.0
        v = _clip((raw - ai.min) / span, 0.0, 1.0)
        return 0.0 if v < self.p.trigger_deadzone else v

    def _abs(self, c: int, value: int) -> bool:
        changed = False
        for kind, k, inv in self._abs_map.get(c, ()):
            v = self.stick_value(c, value, inv) if kind == "stick" else self.trigger_value(c, value)
            if self.state.axes.get(k) != v:
                self.state.axes[k] = v
                changed = True
        if c == self.p.hat[0]:
            self.state.dpad_x = (value > 0) - (value < 0)
            changed = True
        elif c == self.p.hat[1]:
            self.state.dpad_y = (value < 0) - (value > 0)      # HAT0Y -1 = up
            changed = True
        return changed

    def feed(self, etype: int, ecode: int, value: int, t: float) -> bool:
        """Apply one event. Any event (incl. SYN) counts as input liveness."""
        self.state.last_event_t = t
        if etype == EV_ABS:
            return self._abs(ecode, value)
        if etype != EV_KEY or value == 2:                        # 2 = autorepeat
            return False
        on = bool(value)
        if ecode in self.p.buttons:
            self.state.buttons[self.p.buttons[ecode]] = on
            return True
        if ecode in self._trig_btn:
            self.state.axes[self._trig_btn[ecode]] = 1.0 if on else 0.0
            return True
        if ecode in self.p.dpad_btn:
            self._dpad_btns[self.p.dpad_btn[ecode]] = on
            self.state.dpad_x = int(self._dpad_btns["right"]) - int(self._dpad_btns["left"])
            self.state.dpad_y = int(self._dpad_btns["up"]) - int(self._dpad_btns["down"])
            return True
        return False

    def feed_many(self, events: Iterable[Tuple[int, int, int]], t: float) -> None:
        for e in events:
            self.feed(e[0], e[1], e[2], t)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
