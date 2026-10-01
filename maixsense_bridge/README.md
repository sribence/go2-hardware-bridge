# maixsense_bridge — MaixSense A010 ToF → HTTP (:9121)

Rear depth camera `tof_rear` for OmniVision 360 (contract §0 / §F). Sipeed MaixSense A010
(100×100, 8-bit depth) over USB CDC serial. Read-only, never commands the robot.

## Run
```bash
pip install -r requirements.txt
python3 maixsense_bridge.py --mock
python3 maixsense_bridge.py --device /dev/serial/by-id/usb-...A010...   # or /dev/ttyUSB0
MAIXSENSE_HOST_DEVICE=/dev/serial/by-id/usb-... docker compose up -d --build
```
Env / flags: `MAIXSENSE_PORT` (9121), `MAIXSENSE_DEVICE`, `MAIXSENSE_BAUD` (115200; ignored by CDC),
`MAIXSENSE_UNIT` (0), `MAIXSENSE_FPS` (15, 1..19), `MAIXSENSE_HFOV/VFOV` (70/60),
`MAIXSENSE_USE_COEFF` (1 = query `AT+COEFF?` intrinsics), `MAIXSENSE_MIN_M/MAX_M`, `MAIXSENSE_MOCK=1`.

## Endpoints
| GET | returns |
|---|---|
| `/frame.npy` | (100,100) float32 metres, 0 = invalid, header `X-Capture-Time` |
| `/frame.png` | turbo colormap (0..`max_m`, default 2.5 m), invalid = black |
| `/points.npy` | (N,3) float32, OpenCV cam frame (z fwd, x right, y down), invalid dropped, header `X-Points` |
| `/health` | fps, frames, checksum/tail/length errors, bytes_dropped, last_frame_age_s, intrinsics (+source) |

Serial thread: `AT+ISP=0` (stop) → `AT+COEFF?` (optional) → `AT+BINN=1`, `AT+UNIT=u`, `AT+FPS=f`,
`AT+DISP=2` (USB only) → `AT+ISP=1` (start), then streaming parse. Reconnect with backoff
(0.5 → 10 s) on error or no valid frame for 5 s. Mock mode builds byte-exact packets and pushes them
through the same parser (`protocol.fake_serial_stream` is the test generator).

## Protocol provenance
Sources: [W] AT command page https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a010/at_command_en.html
(read via its source https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/en/maixsense/maixsense-a010/at_command_en.md — wiki.sipeed.com itself was blocked from here),
[C] https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/zh/maixsense/maixsense-a010/code.md,
[R] https://github.com/sipeed/MaixSense_ROS (`sipeed_tof_ms_a010_ros/ros2/src/frame_struct.h`, `frame_handle.cc`, `main.cc`).

**Verified from docs/vendor code**
- AT syntax `AT+X=v\r`, `AT+X?\r`; ranges: ISP 0/1, BINN 1/2/4, DISP 0..7 (2 = USB), BAUD index 0..8, UNIT 0..10, FPS 1..19 [W].
- Packet: `00 FF`, LE uint16 length = 16 + payload (excludes checksum+tail), 16 metadata bytes, pixels, checksum = low 8 bits of sum of all previous bytes, tail `0xDD` [W][C][R].
- Metadata layout (reserved 0xFF, output_mode, sensor/driver temp, exposure[4], error_code, reserved, rows@14, cols@15, frame_id@16 LE 12-bit, isp_version, reserved) [R]; rows/cols/frame_id offsets also in [C].
- Depth: UNIT=0 → `(p/5.1)²` mm, UNIT=k → `p·k` mm [W].
- `AT+COEFF?` returns JSON `fx, fy, u0, v0` in u14p18 fixed point (÷262144) [R]; vendor ROS uses value as z.

**Assumed (not found in fetched docs)**
- FOV 70°×60° (contract value / vendor listings) — used only if `AT+COEFF?` fails.
- Codes 0 and 255 = invalid (no return / saturated).
- Depth+IR (`output_mode=1`) payload = depth block then IR block.
- `/dev/ttyUSB0` default and 115200 baud (vendor ROS uses 115200; CDC ignores it).
- Some vendor samples also accept tail `0xCC` (UART/SPI variant); this parser accepts only `0xDD`.

## Tests
`cd go2-hardware-bridge && python3 -m pytest maixsense_bridge -q`
