# vector-faces: local household face recognition for Vector (2026-10-15, hermes)
# YuNet (detect) + SFace (embed) via OpenCV DNN, CPU only. No cloud, no telemetry.
# Same base as vector-curious so wirepod_vector_sdk 0.8.1 behaves identically
# (enroll.py runs in here via `docker exec`).
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/root \
    FACES_IN_CONTAINER=1 \
    OPENCV_LOG_LEVEL=ERROR

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 711 /root

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
# code/models/people live on a bind mount at /app (see run.sh); this copy is
# only a fallback so the image is self-describing.
COPY app/ /app/app/
COPY enroll.py /app/enroll.py

EXPOSE 8096
CMD ["python3", "/app/app/app.py"]
