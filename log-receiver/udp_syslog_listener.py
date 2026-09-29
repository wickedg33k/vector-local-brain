#!/usr/bin/env python3
"""Tiny UDP syslog-ish receiver for the Vector robot.
Listens on UDP_PORT, writes every datagram as one line to LOG_FILE with a
receiver-side timestamp, size-capped via RotatingFileHandler (no external deps).
"""
import logging
import logging.handlers
import os
import socket

LOG_FILE = os.environ.get("LOG_FILE", "/logs/vector.log")
UDP_PORT = int(os.environ.get("UDP_PORT", "5514"))
MAX_BYTES = 20 * 1024 * 1024   # 20MB per file
BACKUP_COUNT = 5               # keep 5 rotated files -> ~100MB total

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("vector")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
)
handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
print(f"vector-log-receiver listening on udp/{UDP_PORT} -> {LOG_FILE}", flush=True)

while True:
    data, addr = sock.recvfrom(65535)
    try:
        line = data.decode("utf-8", errors="replace").rstrip("\r\n")
    except Exception:
        line = repr(data)
    logger.info("[%s] %s", addr[0], line)
