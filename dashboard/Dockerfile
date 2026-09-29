FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

# Non-root user; container never needs to write to /data (all :ro),
# only reads. Uvicorn binds 0.0.0.0:8090 inside the container; the
# container itself runs with --network host so that's the host port.
RUN useradd -u 10001 -m dash
USER dash

EXPOSE 8090

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
