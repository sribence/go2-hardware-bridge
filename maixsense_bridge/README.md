# maixsense_bridge — MaixSense A010 ToF → HTTP (:9121)

Rear depth camera `tof_rear` for OmniVision 360 (contract §0 / §F). Sipeed MaixSense A010
(100×100, 8-bit depth) over its Type-C virtual serial port (`/dev/ttyUSBx`, FTDI-style, see below). Read-only, never commands the robot.

## Run
```bash
pip install -r requirements.txt
python3 maixsense_bridge.py --mock [--binn 2]
python3 maixsense_bridge.py --device /dev/maixsense        # udev symlink, see 99-maixsense.rules
docker compose up -d --build                              # bind-mounts /dev, survives replug
# no hardware: fake A010 on a pty (AT replies + frames), optional unplug/replug cycling
python3 ../tools/a010_fake_device.py --link /tmp/maixsense --unplug-every 10 &
python3 maixsense_bridge.py --device /tmp/maixsense
```
Env / flags: `MAIXSENSE_PORT` (9121), `MAIXSENSE_DEVICE` (/dev/maixsense), `MAIXSENSE_BAUD` (115200; irrelevant
on USB), `MAIXSENSE_UNIT` (0), `MAIXSENSE_FPS` (15, 1..19), `MAIXSENSE_BINN` (1/2/4), `MAIXSENSE_HFOV/VFOV` (70/60),
`MAIXSENSE_USE_COEFF` (1 = query `AT+COEFF?` intrinsics), `MAIXSENSE_MIN_M/MAX_M`, `MAIXSENSE_RECONNECT_MAX_S` (5),
`MAIXSENSE_MOCK=1`.

## Endpoints
| GET | returns |
|---|---|
| `/frame.npy` | (rows,cols) float32 metres, 0 = invalid; 100×100 / 50×50 / 25×25 for BINN 1/2/4. Headers `X-Capture-Time`, `X-Resolution: <rows>x<cols>`, `X-Frame-Id` |
| `/frame.png` | turbo colormap (0..`max_m`, default 2.5 m), invalid = black |
| `/points.npy` | (N,3) float32, OpenCV cam frame (z fwd, x right, y down), invalid dropped, headers `X-Points`, `X-Resolution` |
| `/health` | fps (3 s window), frames, checksum/tail/length errors, bytes_dropped, last_frame_age_s, resolution, binn, intrinsics (+source), opens/reconnects, last_error |

omni (`HttpSource`) only reads the body; `perception360.tof_points_base` already rescales any (h,w) to the rig's
100×100 K, so binned frames work without consumer changes.

## Serial sequence (matches vendor ROS2 `main.cc` L37-105)
open (8N1, baud ignored on USB) → `AT+ISP=0` (drain) → `AT+DISP=1` (USB stream off, drain) → `AT+ISP=1` (drain)
→ `AT` must answer `OK\r\n` (else "not this port", e.g. the other FTDI interface) → `AT+COEFF?` (`+COEFF=1\r\nOK\r\n`
then JSON) → `AT+BINN=b`, `AT+UNIT=u`, `AT+FPS=f` (each `OK`) → `AT+DISP=2` (USB stream on) → streaming parse.
Vendor differences: ROS ends with `DISP=3` (LCD+USB) — we use `DISP=2` like the vendor host example ("FASTER IF YOU
DO NOT NEED LCD"); ROS never sets BINN/FPS. Reconnect with backoff (0.5 s → `RECONNECT_MAX_S`) on read error
(unplug), missing device, or no valid frame for 5 s (stall); each reopen repeats the full sequence.

## Protocol provenance
Pinned sources:
[ROS] https://github.com/sipeed/MaixSense_ROS @ `34c2facb` (`sipeed_tof_ms_a010_ros/{ros1,ros2}/src`) ·
[TOOL] https://github.com/sipeed/MetaSense-ComTool @ `1ba26875` (`COMTool/plugins/gragh_widgets.py`, class
`Gragh_MetaSenseLite` = the official A010 PC viewer) ·
[AT] https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/en/maixsense/maixsense-a010/at_command_en.md ·
[HOST] https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/zh/maixsense/maixsense-a010/code.md (`tof_main_host.py` USB example) ·
[DOC] https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/en/maixsense/maixsense-a010/maixsense-a010.md.
No raw captures exist in these repos; `tests/test_sipeed_golden.py` transliterates the vendor parsers
(ROS `handle_process`, HOST `relay_thread`, TOOL `decodeData`, TOOL `compute_real_distance`, ROS COEFF + point cloud)
and asserts byte-identical results with ours.

| item | ours | source (file:line) | status |
|---|---|---|---|
| header `00 FF`, LE u16 len = 16 + payload, 16 B meta, payload, u8 sum, tail | `protocol.py` | [ROS] ros2/src/frame_struct.h:9-32, frame_handle.cc:57-76; [AT] at_command_en.md:143-151; [TOOL] gragh_widgets.py:969-1016 | verified, golden |
| meta: res 0xFF, mode, sensor/driver temp, exposure[4], err, 0x00, rows@14, cols@15, frame_id@16 (12 bit), isp_ver, 0xFF | `_decode` | [ROS] frame_struct.h:17-32; [TOOL] gragh_widgets.py:972-979; [HOST] code.md:163-165 | verified, golden |
| checksum = sum(all bytes before it) & 0xFF | `checksum` | [ROS] frame_handle.cc:70-75; [HOST] code.md:153 | verified, golden |
| tail 0xDD; **0xCC also accepted** (fixed: was rejected) | `TAILS` | [HOST] code.md:25,155 (`ALLOWED_TAILS=(0xCC,0xDD)`); [ROS] frame_handle.cc:76 / [TOOL] :1016 accept 0xDD only; MaixPy UART example code.md:349 uses 0xCC | verified |
| payload must be rows·cols (**fixed**: was ≥) | `_decode` | [HOST] code.md:241; [ROS] frame_handle.cc:57 caps at 100·100 | verified |
| duplicate frame_id dropped | `feed` | [HOST] code.md:183 | verified |
| BINN 1/2/4 → 100²/50²/25² (**added** `MAIXSENSE_BINN`, X-Resolution) | `BINN_SHAPE` | [AT] at_command_en.md:41-45; [TOOL] gragh_widgets.py:885 (`1<<i`) | verified |
| UNIT=0 → (p/5.1)² mm, UNIT=k → p·k mm, k ≤ 10 | `depth_mm` | [AT] at_command_en.md:105-108,153-159; [TOOL] gragh_widgets.py:1100-1106, slider max 10 :916 | verified, golden (wiki also says "1...9" at :108) |
| FPS 1..19 | `config_sequence` | [AT] at_command_en.md:121-123 ([TOOL] slider allows 20, :925) | verified |
| DISP bitmask 1=LCD 2=USB 4=UART | `DISP_*` | [AT] at_command_en.md:60-69; [TOOL] gragh_widgets.py:870-872 | verified |
| handshake ISP=0/DISP=1/ISP=1, `AT`→`OK\r\n` (**added**) | `handshake_sequence` | [ROS] ros2/src/main.cc:37-66; ros1 node.cc:57-66 | verified, pty test |
| `AT+COEFF?` → `+COEFF=1\r\nOK\r\n` + JSON `fx,fy,u0,v0` u14p18 (÷262144, cJSON valueint, float32) | `parse_coeff` | [ROS] ros2/src/main.cc:69-94; ros1 node.cc:68-89; frame_struct.h:39-52 (LensCoeff_t) | verified, golden |
| baud irrelevant on USB; vendor uses 115200 (ROS) / 921600 (HOST) | `--baud` | [ROS] serial.cc:49, ros1 msa010.hpp:212; [HOST] code.md:21; [DOC] maixsense-a010.md:75 "choose any high baud rate" | verified |
| point cloud: x=d·(i−u0)/fx, y=d·(j−v0)/fy, **z=d (z-depth, not ray range)**, integer pixel index | `depth_to_points` | [ROS] ros2/src/main.cc:194-199 (ros1 node.cc:168-176 = same in ROS axes y fwd, z up) | verified, golden. Vendor uses raw code/1000 and ignores UNIT (vendor bug); we apply UNIT first |
| other AT: ANTIMMI −1..41, AE, EV (not used) | `AT_RANGES` | [AT] at_command_en.md:125-141; [TOOL] gragh_widgets.py:880,939 | verified |
| Linux node `/dev/ttyUSBx`, Win7 needs FTDI driver | udev rule | [DOC] maixsense-a010.md:23,69 | verified text |

**Assumed / not in any Sipeed source**
- USB VID:PID: not stated anywhere. Inferred 0403:6010 (BL702 emulating FTDI FT2232: ttyUSB + "FTDI driver"; ROS1 lib
  defaults to `/dev/ttyUSB1`, msa010.hpp:13, i.e. two interfaces). `99-maixsense.rules` has a TODO with the
  `udevadm info -a -n /dev/ttyUSB0` command to confirm; the `AT`→`OK` probe rejects a wrong interface at runtime.
- `AT+COEFF?` intrinsics refer to the 100×100 grid; for BINN 2/4 they are rescaled (`scale_intrinsics`, pixel-centre convention).
- Codes 0 and 255 = invalid (no return / saturated).
- Depth+IR (`output_mode=1`) payload = depth block then IR block (no vendor parser handles mode 1; ROS caps payload at 100·100).
- FOV 70°×60° (contract value) — used only if `AT+COEFF?` fails.
- Invalid AT value → `ERROR\r\n`; power-on DISP state (fake device uses 1=LCD). Neither affects the bridge.
- ROS2 `handle_process` hangs on a `00 FF` lookalike with length > 10016 (`goto __find_header` without consuming);
  our parser and the golden transliteration skip one byte instead.

## USB robustness
- `99-maixsense.rules` → `/dev/maixsense` (VID:PID, interface 00; `maixsense1` for interface 01).
- docker-compose bind-mounts `/dev` read-only + `device_cgroup_rules: c 188:* rmw`, so the symlink follows replugs
  (a static `devices:` entry would pin the old node).
- `tests/test_a010_pty.py`: handshake, COEFF, BINN 2/4 through the API, hot-unplug (pty + symlink removed) → replug,
  silent stall (ISP off) → reopen.

## Tests
`cd go2-hardware-bridge && python3 -m pytest maixsense_bridge -q`
