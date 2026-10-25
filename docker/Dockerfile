# Reuse the verified original runtime and detector without package upgrades.
ARG DETECTOR_BASE=umdetree/signal-detector@sha256:38d12791e62a026ac1cbeb0064e4ddea155e36168c9ec854fd35e65d919c7ef2
FROM ${DETECTOR_BASE}

LABEL org.opencontainers.image.title="Signal detection and modulation recognition" \
      org.opencontainers.image.version="0.1.1" \
      signal-recognition.detector.base="umdetree/signal-detector:0.2.1" \
      signal-recognition.amc.sha256="26c2606b81d599426d9786b5c4c12b434425a5dc39c374cfeafeb109d6bffbea"

ENV AMC_CHECKPOINT=/models/amc20_v1.pt \
    AMC_DEVICE=auto \
    AMC_SCORE_THRESHOLD=0.5 \
    AMC_WINDOW_BATCH_SIZE=64 \
    AMC_MAX_SIGNALS=64 \
    AMC_MAX_REQUEST_BYTES=268435456

# Only the inference modules are shipped; no training data, runs or test fixtures.
COPY src/amc/__init__.py src/amc/device.py src/amc/model.py src/amc/inference.py src/amc/prepared_inference.py src/amc/service.py src/amc/integrated.py /app/amc/
COPY models/amc20_v1.pt /models/amc20_v1.pt
COPY docker/check_runtime.py docker/healthcheck.py /app/deployment/
RUN python /app/deployment/check_runtime.py

COPY docs/README.md docs/AMC_API.md /app/docs/
COPY ["docs/单信号IQ预处理参考.md", "/app/docs/单信号IQ预处理参考.md"]
COPY docs/detector/ /app/docs/detector/
COPY docker/IMAGE_README.md /app/README.md
COPY docker-compose.yml /app/docker-compose.yml
COPY tools/client_amc_batch.py /app/examples/client_amc_batch.py

HEALTHCHECK --interval=30s --timeout=120s --start-period=180s --retries=3 \
    CMD ["python", "/app/deployment/healthcheck.py"]

WORKDIR /app
EXPOSE 5555
CMD ["uvicorn", "amc.integrated:app", "--host", "0.0.0.0", "--port", "5555", "--workers", "1", "--log-level", "info"]
