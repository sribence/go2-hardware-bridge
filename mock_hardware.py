"""
Mock Hardware Publisher for Unitree Go2 Hardware Bridge
Publishes dummy camera frames and IMU/state telemetry for testing without real hardware.
"""

import time
import json

def run_mock_hardware():
    print("[MOCK HARDWARE] Starting synthetic hardware bridge publisher...")
    frame_count = 0
    try:
        while True:
            frame_count += 1
            telemetry = {
                "timestamp": time.time(),
                "frame": frame_count,
                "battery_percentage": 95,
                "mode": "standby",
                "imu": {"pitch": 0.01, "roll": -0.02, "yaw": 1.25}
            }
            print(f"[MOCK HARDWARE] Published frame #{frame_count} - Telemetry: {json.dumps(telemetry)}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[MOCK HARDWARE] Stopped.")

if __name__ == "__main__":
    run_mock_hardware()
