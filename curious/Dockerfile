FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# code/config/state/log live on a bind mount at /app (see run command); the
# COPY below is a fallback baked into the image so the container is usable
# even before the bind mount is populated on first build.
COPY curious.py /app/curious.py

CMD ["python3", "/app/curious.py"]
