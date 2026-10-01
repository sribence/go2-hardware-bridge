# thermal_bridge — USB thermal cameras → HTTP (:9120)

Two front UVC thermal modules (`th_front_narrow`, `th_front_wide`) for OmniVision 360
(contract: `go2-brain-logic/mission_control/omni/CONTRACTS.md` §0 and §F). Read-only, never commands the robot.

## Run
```bash
pip install -r requirements.txt
python3 thermal_bridge.py --mock                 # synthetic scene, no hardware
python3 thermal_bridge.py                        # real cams from thermal_cams.yaml
docker compose up -d --build                     # devices from docker-compose.yml
```
Env: `THERMAL_PORT` (9120), `THERMAL_CONFIG`, `THERMAL_MOCK=1`, `THERMAL_HOST`.
Docker mock without the cameras plugged in: remove the `devices:` block (Docker refuses missing devices).

## Endpoints
| GET | returns |
|---|---|
| `/cams` | config + status per camera (fps, frames, last_frame_age_s, errors) |
| `/cams/{id}/frame.npy` | float32 °C, `np.save` bytes, header `X-Capture-Time` (unix s) |
| `/cams/{id}/frame.png` | inferno colormap, headers `X-Temp-Min`, `X-Temp-Max`; query `t_min`,`t_max` fix the scale |
| `/cams/{id}/stats` | `min`, `max`, `mean`, `hotspot {x, y, temp}` |
| `/health` | `ok` = every camera delivered a frame < 2 s ago |

## Config (`thermal_cams.yaml`)
Per camera: `device` (stable `/dev/v4l/by-id/...`; if both modules report the same serial use
`/dev/v4l/by-path/...`), `decoder`, `width/height` (decoded image), `capture_width/height` + `fourcc`
(V4L2 request), `hfov_deg`, `fov_note`, `decoder_params`. Device names in the YAML are placeholders.

## Decoders (`decoders.py`, pure + unit-tested)
| name | input | °C |
|---|---|---|
| `infiray_p2`, `tc001` | 256×384 YUYV; `thermal_half` (default `bottom`) = raw LE uint16 | `raw/64 − 273.15` |
| `raw_y16` | W×H LE uint16 | `raw*scale + offset` |
| `grey8_relative` | 8-bit grey / YUYV luma / BGR | `t_min + g/255·(t_max−t_min)` — **relative only** |

Capture: `cv2.VideoCapture(dev, CAP_V4L2)` + `CAP_PROP_CONVERT_RGB=0` → raw YUYV bytes; on failure the
device is reopened with exponential backoff (0.5 → 10 s). Requires an OpenCV whose V4L2 backend honours
`CONVERT_RGB=0` (OpenCV ≥ 4.2); if a build returns BGR anyway, `decode_errors` grows on `/health`.

## Notes / assumptions
- Layout of P2 Pro / TC001 (bottom half = temperature, `raw/64 − 273.15`) is the community-documented
  layout of these modules; verify with a known reference (e.g. a hand ≈ 34 °C) on first install.
- Flat-field correction (NUC/shutter) is done autonomously by the modules; triggering it is not implemented.
- Mock: ~20 °C room, a 34–36 °C person walking at 2.5–7 m; blob size follows `hfov_deg`, so the narrow cam
  shows a larger person than the wide one.

## Tests
`cd go2-hardware-bridge && python3 -m pytest thermal_bridge -q`
