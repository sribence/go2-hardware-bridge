"""Pure gamepad control logic (no I/O): PadState + time -> list of Commands.

Mapping (mission/CONTRACT.md 9.4; actions remappable in config `control.actions`):
  RB held         dead-man: REQUIRED for any motion; POST /deadman 10 Hz while held
  left stick      vx (up = forward), vy (left = +)        } scaled by speed level,
  right stick X   vyaw (left = +, CCW)                    } POST /move 20 Hz
  LT / RT         speed level down / up (stealth, normal[, sprint if allow_sprint])
  B               /stop now (motion latched off until RB is re-pressed)
  Y               stand/sit toggle -> /action/{stand|sit}
  A               mission pause/resume
  X               capture (omni)
  Start+Select    held estop_hold_s -> E-STOP (latched until RB re-pressed)
Fail-safe: dead-man release, disconnect, read error, or no input for
stale_input_s while moving -> exactly ONE stop, then silence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from .profiles import PadState
except ImportError:                       # run as a script
    from profiles import PadState

DEFAULT_LEVELS = {
    "stealth": (0.25, 0.15, 0.5),
    "normal": (0.6, 0.3, 1.0),
    "sprint": (1.5, 0.4, 1.5),
}
ACTIONS = ("stop", "stand_sit", "mission_toggle", "capture", "none")


@dataclass
class ControlConfig:
    deadzone: float = 0.12
    move_hz: float = 20.0
    deadman_hz: float = 10.0
    stale_input_s: float = 0.5
    deadman_stale_s: float = 3.0
    estop_hold_s: float = 2.0
    trigger_on: float = 0.6
    trigger_off: float = 0.3
    default_level: str = "normal"
    allow_sprint: bool = False
    stand_requires_deadman: bool = True
    assume_standing: bool = True
    levels: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: dict(DEFAULT_LEVELS))
    actions: Dict[str, str] = field(default_factory=lambda: {
        "a": "mission_toggle", "b": "stop", "x": "capture", "y": "stand_sit"})
    client_id: str = "gamepad"
    source: str = "gamepad"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]], deadzone: Optional[float] = None) -> "ControlConfig":
        c = cls()
        d = dict(d or {})
        lv = d.pop("levels", None)
        if lv:
            c.levels = dict(c.levels)
            for k, v in lv.items():
                c.levels[k] = (float(v["vx"]), float(v["vy"]), float(v["vyaw"])) if isinstance(v, dict) \
                    else tuple(float(x) for x in v)
        acts = d.pop("actions", None)
        if acts:
            for k, v in acts.items():
                if v not in ACTIONS:
                    raise ValueError("unknown action %r for button %r" % (v, k))
            c.actions = dict(acts)
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, type(getattr(c, k))(v))
        if deadzone is not None:
            c.deadzone = float(deadzone)
        return c

    def level_names(self) -> List[str]:
        names = [n for n in ("stealth", "normal") if n in self.levels]
        if self.allow_sprint and "sprint" in self.levels:
            names.append("sprint")
        return names


@dataclass
class Command:
    kind: str                       # move|stop|deadman|deadman_release|action|mission_toggle|capture|estop|rumble
    body: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return "Command(%s, %s)" % (self.kind, self.body)


def radial_deadzone(x: float, y: float, dz: float) -> Tuple[float, float]:
    """Circular deadzone with rescaling so output starts at 0 at the edge and
    reaches 1 at full deflection (no jump)."""
    m = math.hypot(x, y)
    if m <= dz or dz >= 1.0:
        return 0.0, 0.0
    s = min(1.0, (m - dz) / (1.0 - dz)) / m
    return x * s, y * s


def axis_deadzone(v: float, dz: float) -> float:
    if abs(v) <= dz or dz >= 1.0:
        return 0.0
    return math.copysign(min(1.0, (abs(v) - dz) / (1.0 - dz)), v)


def stick_to_twist(state: PadState, level: Tuple[float, float, float], dz: float) -> Tuple[float, float, float]:
    """Left stick -> (vx, vy), right stick X -> vyaw. Logical axes are
    right/up-positive; robot frame is x forward, y left, yaw CCW."""
    lx, ly = radial_deadzone(state.axes.get("lx", 0.0), state.axes.get("ly", 0.0), dz)
    rx = axis_deadzone(state.axes.get("rx", 0.0), dz)
    vx = ly * level[0]
    vy = -lx * level[1]
    vyaw = -rx * level[2]
    return (round(vx, 4) + 0.0, round(vy, 4) + 0.0, round(vyaw, 4) + 0.0)


class Controller:
    def __init__(self, cfg: Optional[ControlConfig] = None):
        self.cfg = cfg or ControlConfig()
        names = self.cfg.level_names()
        self.level = self.cfg.default_level if self.cfg.default_level in names else names[-1]
        self.standing = self.cfg.assume_standing
        self.moving = False                 # a /move went out since the last /stop
        self.deadman = False                # RB considered held (and pad live)
        self.latched = False                # B / E-STOP: no motion until RB re-pressed
        self.prev_buttons: Dict[str, bool] = {}
        self.trig_armed = {"lt": True, "rt": True}
        self.combo_since: Optional[float] = None
        self.estop_fired = False
        self.last_move_t = -1e9
        self.last_beat_t = -1e9
        self.last_cmd: Optional[Command] = None
        self.last_reason = ""
        self.denied = False

    # -- helpers ---------------------------------------------------------------
    def _emit(self, out: List[Command], cmd: Command) -> None:
        out.append(cmd)
        if cmd.kind not in ("rumble", "deadman"):
            self.last_cmd = cmd

    def _stop(self, out: List[Command], reason: str) -> None:
        """At most one stop per motion episode."""
        if any(c.kind in ("stop", "estop") for c in out):
            self.moving = False
            return
        self._emit(out, Command("stop", {"source": self.cfg.source, "reason": reason}))
        self.moving = False
        self.last_reason = reason

    def _release_deadman(self, out: List[Command]) -> None:
        if self.deadman:
            self._emit(out, Command("deadman_release", {"client_id": self.cfg.client_id, "release": True}))
        self.deadman = False

    def disconnect(self, now: float, reason: str = "disconnect") -> List[Command]:
        out: List[Command] = []
        if self.moving:
            self._stop(out, reason)
        self._release_deadman(out)
        self.prev_buttons = {}
        self.combo_since = None
        self.estop_fired = False
        self.trig_armed = {"lt": True, "rt": True}
        return out

    def _change_level(self, step: int, out: List[Command]) -> None:
        names = self.cfg.level_names()
        i = names.index(self.level) if self.level in names else 0
        j = max(0, min(len(names) - 1, i + step))
        if j != i:
            self.level = names[j]
            self._emit(out, Command("rumble", {"pattern": "level", "level": self.level}))

    # -- main ----------------------------------------------------------------
    def tick(self, now: float, state: Optional[PadState], connected: bool = True,
             error: Optional[str] = None) -> List[Command]:
        if not connected or state is None or error:
            return self.disconnect(now, error or "disconnect")
        cfg = self.cfg
        out: List[Command] = []
        age = now - state.last_event_t
        btn = state.buttons
        pressed = {k for k, v in btn.items() if v and not self.prev_buttons.get(k)}
        self.prev_buttons = dict(btn)

        # E-STOP combo (Start+Select held)
        if btn.get("start") and btn.get("select"):
            if self.combo_since is None:
                self.combo_since = now
            elif not self.estop_fired and now - self.combo_since >= cfg.estop_hold_s:
                self.estop_fired = True
                self.latched = True
                self.moving = False
                self._emit(out, Command("estop", {"source": cfg.source}))
                self._emit(out, Command("rumble", {"pattern": "estop"}))
                self.last_reason = "estop"
        else:
            self.combo_since = None
            self.estop_fired = False

        # speed level: LT down / RT up (edge with hysteresis)
        for k, step in (("lt", -1), ("rt", +1)):
            v = state.axes.get(k, 0.0)
            if self.trig_armed[k] and v >= cfg.trigger_on:
                self.trig_armed[k] = False
                self._change_level(step, out)
            elif v <= cfg.trigger_off:
                self.trig_armed[k] = True

        # dead-man (RB). Counts only while the pad is live.
        rb = bool(btn.get("rb")) and age <= cfg.deadman_stale_s
        if rb and not self.deadman:
            self.deadman = True
            self.latched = False
            self.last_beat_t = -1e9
            self._emit(out, Command("rumble", {"pattern": "engage"}))
        elif not rb and self.deadman:
            if self.moving:
                self._stop(out, "deadman released" if not btn.get("rb") else "input stale")
            self._release_deadman(out)
        if self.deadman and now - self.last_beat_t >= 1.0 / cfg.deadman_hz - 1e-6:
            self.last_beat_t = now
            self._emit(out, Command("deadman", {"client_id": cfg.client_id}))

        # one-shot buttons
        for b in sorted(pressed):
            act = cfg.actions.get(b, "none")
            if act == "stop":
                self.latched = True
                self._stop(out, "button stop")
            elif act == "stand_sit":
                name = "sit" if self.standing else "stand"
                if name == "stand" and cfg.stand_requires_deadman and not self.deadman:
                    self._emit(out, Command("rumble", {"pattern": "deny"}))
                    self.last_reason = "stand denied: hold dead-man (RB)"
                else:
                    self.standing = not self.standing
                    self._emit(out, Command("action", {"name": name, "source": cfg.source}))
            elif act == "mission_toggle":
                self._emit(out, Command("mission_toggle", {"source": cfg.source}))
            elif act == "capture":
                self._emit(out, Command("capture", {"cams": "all", "mode": "max", "source": cfg.source}))

        # motion
        lv = cfg.levels[self.level]
        vx, vy, vyaw = stick_to_twist(state, lv, cfg.deadzone)
        active = (vx, vy, vyaw) != (0.0, 0.0, 0.0)
        fresh = age <= cfg.stale_input_s
        if self.deadman and not self.latched and active and fresh:
            if now - self.last_move_t >= 1.0 / cfg.move_hz - 1e-6:
                self.last_move_t = now
                self.moving = True
                self._emit(out, Command("move", {"vx": vx, "vy": vy, "vyaw": vyaw,
                                                 "source": cfg.source, "profile": self.level}))
        elif self.moving:
            self._stop(out, "input stale" if (active and not fresh) else "stick released")
        # denial feedback: stick pushed without dead-man (once per push)
        if active and not self.deadman:
            if not self.denied:
                self.denied = True
                self._emit(out, Command("rumble", {"pattern": "deny"}))
                self.last_reason = "motion denied: hold dead-man (RB)"
        elif not active:
            self.denied = False
        return out

    def status(self) -> Dict[str, Any]:
        return {"level": self.level, "levels": self.cfg.level_names(), "deadman": self.deadman,
                "moving": self.moving, "latched": self.latched, "standing_assumed": self.standing,
                "estop_combo_held": self.combo_since is not None,
                "last_command": None if self.last_cmd is None else
                {"kind": self.last_cmd.kind, "body": self.last_cmd.body},
                "last_reason": self.last_reason}
