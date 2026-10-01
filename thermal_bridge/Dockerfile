# thermal_bridge -- UVC thermal cameras (InfiRay P2 Pro / TC001 class) -> HTTP :9120
# Read-only sensor bridge; does not touch the robot or the native system.
FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 v4l-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY decoders.py thermal_bridge.py thermal_cams.yaml ./

ENV THERMAL_PORT=9120 THERMAL_CONFIG=/app/thermal_cams.yaml
EXPOSE 9120
CMD ["python", "thermal_bridge.py"]
