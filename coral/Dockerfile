# vector-coral: Google Coral USB Edge TPU object-detection service
# Debian bullseye ships Python 3.9, which is the last CPython version
# pycoral/tflite-runtime officially published wheels for. The host's
# system Python (3.14) is far too new for pycoral, hence this container.

FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Debian 11 (bullseye) is EOL -- deb.debian.org 404s on it now. Repoint at
# the archive so apt still works, and disable Release Valid-Until checks
# (archived Release files are intentionally frozen/expired).
RUN echo "deb http://archive.debian.org/debian bullseye main" > /etc/apt/sources.list \
    && echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until

# --- OS deps + Coral apt repo (modern signed-by keyring, not deprecated apt-key) ---
# Note: base image's pre-baked gpgv is newer than archive.debian.org's
# main-only gnupg package wants (EOL-archive version skew) -- installing
# gnupg here deadlocks apt. We don't need it: skip signature verification
# on the (already-HTTPS, already end-of-life-pinned) Coral apt repo instead
# of dearmoring its key, so gnupg is never required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        python3.9 python3-pip python3.9-distutils \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb [trusted=yes] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (pinned known-good combo for py3.9 / libedgetpu1-std) ---
RUN python3.9 -m pip install --no-cache-dir --upgrade "pip==23.0.1" setuptools wheel \
    && python3.9 -m pip install --no-cache-dir \
        --extra-index-url https://google-coral.github.io/py-repo/ \
        tflite-runtime==2.5.0.post1 \
        pycoral~=2.0 \
    && python3.9 -m pip install --no-cache-dir \
        flask==2.2.5 \
        werkzeug==2.2.3 \
        pillow==9.5.0 \
        numpy==1.19.5

WORKDIR /app
COPY app/ /app/
# models/ is bind-mounted at runtime (see README) so models can be swapped
# without a rebuild; a copy is included in the image too as a fallback.
COPY models/ /app/models/

EXPOSE 8095

CMD ["python3.9", "app.py"]
