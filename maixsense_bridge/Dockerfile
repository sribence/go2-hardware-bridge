# maixsense_bridge -- Sipeed MaixSense A010 ToF (USB CDC serial) -> HTTP :9121
# Read-only sensor bridge; does not touch the robot or the native system.
FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY protocol.py maixsense_bridge.py ./

ENV MAIXSENSE_PORT=9121 MAIXSENSE_DEVICE=/dev/maixsense
EXPOSE 9121
CMD ["python", "maixsense_bridge.py"]
