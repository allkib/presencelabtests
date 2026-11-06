# syntax=docker/dockerfile:1.7
# Production image for Dokploy (or any Docker host). Browser-camera mode only:
# a server has no webcam, so visitors stream from their own browser over /ws/infer.

ARG PYTHON_VERSION=3.11

# ---------- build: install deps into a venv ----------
FROM python:${PYTHON_VERSION}-slim AS build

# CPU-only PyTorch keeps the image ~1.5 GB smaller. For an NVIDIA host, build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG MODEL=yolo11n-pose.pt
ARG OBJECT_MODEL=yolo11n.pt

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --index-url ${TORCH_INDEX_URL} torch torchvision

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    # opencv-python needs X11/GL libs; swap in the headless build for servers.
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps "opencv-python-headless>=4.9"

# Bake both models' weights into the image (no download at container start).
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
RUN mkdir -p /app/models && python -c "from ultralytics import YOLO; \
    YOLO('/app/models/${MODEL}'); YOLO('/app/models/${OBJECT_MODEL}')"

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG MODEL=yolo11n-pose.pt
ARG OBJECT_MODEL=yolo11n.pt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=app:app /app/models /app/models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/home/app/.config/Ultralytics \
    YOLO_AUTOINSTALL=false \
    YOLO_OFFLINE=true \
    POSEDEMO_HOST=0.0.0.0 \
    POSEDEMO_PORT=8000 \
    POSEDEMO_ENABLE_SOURCE=false \
    POSEDEMO_DEVICE=cpu \
    POSEDEMO_MODELS_DIR=/app/models \
    POSEDEMO_MODEL=${MODEL} \
    POSEDEMO_OBJECT_MODEL=${OBJECT_MODEL} \
    POSEDEMO_FORWARDED_ALLOW_IPS=*

USER app
WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"POSEDEMO_PORT\"]}/health', timeout=4)" || exit 1

CMD ["posedemo"]
