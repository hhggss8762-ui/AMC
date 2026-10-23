# Signal Detector: A Deep Learning Approach to Detecting Signals from Spectrograms

---

# Overview

The **signal-localization beta** uses a single `SIGNAL` class and two
server-side inference modes for structured IQ/TF requests: adaptive
global-then-zoom by default, and hard 2.5 MHz frequency windows when the
request declares a supported band (`specific_band: "851MHz-866MHz"`).
The current release is `umdetree/signal-detector:0.2.1` (`latest` follows it);
the eight-protocol-class image remains available at pinned tags such as
`0.1.2-gpu`. See [BETA.md](BETA.md) for the selected weights, image digests,
build arguments, and validation results, and [api.md](api.md) for the
`specific_band` field. The protocol profile described below remains the
repository's default Dockerfile build.

This repository provides a standalone FastAPI inference service for RF-DETR
signal recognition. The published protocol image contains the protocol model
configuration and checkpoint, so it does not depend on the parent SpecAgent
repository or host-side model paths.

## Quick start (localization beta)

```bash
docker pull umdetree/signal-detector:0.2.1
docker run -d --name signal-detector --gpus all -p 5555:5555 \
  -e MODEL_DEVICE=cuda umdetree/signal-detector:0.2.1
curl http://localhost:5555/ready
```

The response reports `"model": "signal-localization-beta"` and
`class_names: {"0": "SIGNAL"}`. To run the fixed 2.5 MHz windows on an
851–866 MHz capture, add `"specific_band": "851MHz-866MHz"` to the
`/v1/infer` metadata:

```bash
curl -X POST http://localhost:5555/v1/infer \
  -F 'data=@capture.bin;type=application/octet-stream' \
  -F 'metadata={"input_type":"iq","dtype":"int16","sample_rate_hz":19200000,"center_freq_hz":860250000,"specific_band":"851MHz-866MHz"}'
```

Without `specific_band` the service keeps the adaptive box-count behavior.
Use `-e MODEL_DEVICE=cpu` (and drop `--gpus all`) for CPU-only hosts.

# Installation

For local non-Docker use, put the RF-DETR checkpoint in `checkpoints/` and
configure the corresponding path in `config.py`. The published Docker image
already contains the protocol checkpoint and does not need this step.

## Conda

```bash
conda create -n signal_detector python=3.10.19
conda activate signal_detector
pip install -r requirements.txt
```

## Docker

The protocol image contains the following model files:

```text
/models/model.yaml
/models/rfdetr.pth
```

It also ships `docker-compose.yml`, `README.md`, and `api.md` at `/app/` so
recipients can extract them without access to the source repository. The image
does not require a host-side model directory, parent repository, or relative
`..` paths at runtime.

### Extract compose and documentation from the image

Recipients do not need source repository access. After pulling the image, the
deploy files can be copied out of the image itself. The published `latest` tag
currently tracks the localization beta; substitute a pinned protocol tag such
as `0.1.2-gpu` in the commands below to extract the protocol deploy files:

```bash
docker pull umdetree/signal-detector:latest

docker create --name sd-extract umdetree/signal-detector:latest
docker cp sd-extract:/app/docker-compose.yml ./docker-compose.yml
docker cp sd-extract:/app/README.md ./README.md
docker cp sd-extract:/app/api.md ./api.md
docker rm sd-extract

docker compose up -d
```

### Build and publish

Run these commands from the root of this repository. The publisher must have
`checkpoints/rfdetr.pth` available before building. Keep an immutable release
tag in addition to the moving `latest` tag.

```bash
docker build \
  -t umdetree/signal-detector:0.1.1-gpu \
  -t umdetree/signal-detector:latest \
  .
docker login
docker push umdetree/signal-detector:0.1.1-gpu
docker push umdetree/signal-detector:latest
```

Use a new immutable image tag whenever the checkpoint or model configuration
changes. `docker-compose.yml` follows `latest` for convenient updates; use an
immutable tag there when reproducibility and rollback are more important.

### Pull and run

The GPU host must provide NVIDIA Container Toolkit support.

```bash
docker pull umdetree/signal-detector:latest
docker run -d --name signal-detector \
  --gpus all \
  -p 5555:5555 \
  umdetree/signal-detector:latest
```

### Run with Compose

A `docker-compose.yml` is provided in the repository root. It fixes the image
tag, port mapping, GPU reservation, healthcheck, and restart policy, so no
`docker run` flags or host-side paths are needed.

```bash
docker compose pull
docker compose up -d
```

Check the service status:

```bash
docker compose ps
curl http://localhost:5555/ready
```

To upgrade the moving `latest` image:

```bash
docker compose pull && docker compose up -d
```

For a pinned release, replace `latest` in `docker-compose.yml` with an
immutable tag such as `0.1.1-gpu` before starting the service.

To stop the service:

```bash
docker compose down
```

The container exposes `/health`, `/ready`, and `/v1/infer`. Model loading can
take time before `/ready` returns successfully. The Docker healthcheck calls
`/ready` so Compose waits for model readiness.

To stop a manually started container:

```bash
docker rm -f signal-detector
```

---

# Usage

## HTTP Service

Run the app directly for local development:

```bash
python -m app.main --host 127.0.0.1 --port 5555 --reload
```

or with uvicorn directly

```bash
uvicorn app.main:app --host 127.0.0.1 --port 5555 --reload
```

The service loads the configured RF-DETR model on the first `/ready` or
inference request. After loading, it runs one dummy 560x560 inference so CUDA
and cuDNN initialization happen before the first real request. The published
image sets `MODEL_CONFIG_PATH=/models/model.yaml` and includes the protocol
checkpoint. For a custom local model, set `MODEL_CONFIG_PATH` to a YAML file
containing the checkpoint, model size, class names, and inference threshold.

Use `/health` for a process liveness check and `/ready` for a model readiness
check:

```bash
curl http://localhost:5555/health
curl http://localhost:5555/ready
```

## API Selection

| Endpoint | Input | Response | Preprocessing |
| --- | --- | --- | --- |
| `/v1/infer` | IQ or TF binary data | Structured detections | IQ/TF conversion, no annotated image |
| `/images/annotate/` | Rendered image files | YOLO labels and base64 image | No IQ conversion; annotates the image |
| `/bin/annotate/` | Raw IQ files | YOLO labels and base64 image | Converts IQ to a spectrogram |
| `/transcom_csv/annotate/` | Transcom CSV files | YOLO labels and base64 image | Converts CSV scans to images |

### API: Structured inference

`POST /v1/infer` accepts a binary payload in the `data` form field and a JSON
metadata string in the `metadata` form field.

IQ metadata:

```json
{
  "input_type": "iq",
  "dtype": "float16",
  "sample_rate_hz": 100000000,
  "center_freq_hz": 2450000000
}
```

2D time-frequency intensity matrix metadata:

```json
{
  "input_type": "tf_intensity",
  "dtype": "float16",
  "value_scale": "db",
  "shape": [256, 512],
  "time_window_ms": 100.0,
  "freq_start_hz": 2400000000,
  "freq_end_hz": 2500000000
}
```

For `tf_intensity`, the binary payload is a row-major flattened matrix with
`shape[0] * shape[1]` values. Frequency rows must be ordered from low to high. 
With the optional `value_scale` omitted, or set to `"db"`, `"dbm"`, or `"dbw"`,
the values are treated as already logarithmic. Set `value_scale: "linear"`
for linear power/intensity values; the service then applies `10 * log10`.
The complete finite dB range is used for the `viridis` colormap. The result is
rendered as a 560x560 RGB image and runs through the same RF-DETR inference
path as IQ input.

TF rendering normalizes the complete finite dB range, downscales the scalar
matrix to 560x560 with bilinear resampling, and only then applies the
`viridis` lookup table. This avoids applying the colormap to the
full-resolution matrix and keeps per-request conversion overhead low while
preserving the `ax.specgram` mapping.

Payloads larger than `TF_MAX_ELEMENTS` (default `20000000`) are rejected with
HTTP 400. Set the environment variable to raise or lower that limit.

The `tf_intensity` route is intended for matrices computed before the request.
`float16` is supported to reduce the upload size. Use the same `NFFT=2048` and
`noverlap=1024` settings as the IQ path when generating a compatible matrix.

The `db` form is recommended when the matrix was generated from
`matplotlib.specgram`: convert the power spectrum with `10 * log10` before
downsampling and sending it. The service then applies the same dB range,
`viridis` colormap, and frequency orientation as the IQ path. If the payload
contains linear power values instead, use `value_scale: "linear"`.

Example structured request:

```bash
curl -X POST http://localhost:5555/v1/infer \
  -F 'data=@tf.bin;type=application/octet-stream' \
  -F 'metadata={"input_type":"tf_intensity","dtype":"float16","value_scale":"db","shape":[256,512],"time_window_ms":100.0,"freq_start_hz":2400000000,"freq_end_hz":2500000000}'
```

The response always reports `image_width_px: 560` and `image_height_px: 560`,
regardless of the input matrix shape. The model receives that normalized
560x560 RGB image.

For latency-sensitive deployments, prefer `/v1/infer` with a precomputed TF
matrix or use `/images/annotate/` with a pre-rendered image. Sending raw IQ to
`/v1/infer` includes the server-side spectrogram calculation.

Successful `/v1/infer` responses have this structure:

```json
{
  "model": "protocol-classification",
  "class_names": {"0": "20 MHz WiFi"},
  "image_width_px": 560,
  "image_height_px": 560,
  "calibration": {
    "time_window_ms": 100.0,
    "freq_start_hz": 2400000000.0,
    "freq_end_hz": 2500000000.0,
    "img_width_px": 560,
    "img_height_px": 560
  },
  "detections": [
    {
      "xyxy": [10.0, 20.0, 100.0, 200.0],
      "class_id": 0,
      "class_name": "20 MHz WiFi",
      "confidence": 0.95
    }
  ]
}
```

### Stable bbox contract

The `detections[].xyxy` field is a fixed API contract and will remain in this
format:

```text
[x1, y1, x2, y2]
```

| Value | Meaning |
| --- | --- |
| `x1` | Left boundary of the box |
| `y1` | Top boundary of the box |
| `x2` | Right boundary of the box |
| `y2` | Bottom boundary of the box |

Coordinates are floating-point pixels in the model image coordinate system.
The origin `(0, 0)` is the top-left corner; `x` increases to the right and `y`
increases downward. Coordinates are not normalized: use `image_width_px` and
`image_height_px` from the response as the image bounds. The box width is
`x2 - x1` and the box height is `y2 - y1`.

The physical mapping uses the response `calibration` values. For a detection
box `[x1, y1, x2, y2]`:

```text
start_time_ms = x1 / image_width_px * time_window_ms
end_time_ms = x2 / image_width_px * time_window_ms
high_freq_hz = freq_end_hz - y1 / image_height_px * (freq_end_hz - freq_start_hz)
low_freq_hz = freq_end_hz - y2 / image_height_px * (freq_end_hz - freq_start_hz)
duration_ms = (x2 - x1) / image_width_px * time_window_ms
center_freq_hz = freq_end_hz - ((y1 + y2) / 2) / image_height_px * (freq_end_hz - freq_start_hz)
bandwidth_hz = (y2 - y1) / image_height_px * (freq_end_hz - freq_start_hz)
```

Here `x` is the time axis and `y` is the frequency axis. The top of the image
is `freq_end_hz`; the bottom is `freq_start_hz`. `start_time_ms` and
`end_time_ms` are relative to the beginning of the input window. A detector may
produce a coordinate slightly outside the image boundary; clamp it before
drawing if required. The API will not replace `xyxy` with `xywh`,
center-width-height, or normalized coordinates. Additional fields may be added
without changing the meaning or order of `xyxy`.

Physical conversion requires `time_window_ms`, `freq_start_hz`, and
`freq_end_hz` in `calibration`. If any of these values is `null`, only the
pixel coordinates are available.

The legacy `/images/annotate/`, `/bin/annotate/`, and
`/transcom_csv/annotate/` endpoints use a separate `labels_yolo` string format:
`class_id x_center y_center width height confidence`, where the four box
values are normalized. Do not confuse that legacy format with `/v1/infer`'s
pixel `xyxy` list.

### API: Images

POST /images/annotate/

Description: Accepts one or more image files and returns, for each image, a base64-encoded annotated image and
YOLO-like labels.

This legacy endpoint accepts an already rendered image and includes annotation
and JPEG/base64 encoding in its response. It does not perform IQ-to-spectrogram
conversion. In contrast, `/v1/infer` accepts IQ or time-frequency data and
returns structured detections without an annotated image.

Request
- Method: POST
- Path: /images/annotate/
- Content-Type: multipart/form-data
- Form fields:
  - files: one or more file fields (each is an image). Use the same key `files` for multiple files.

### API: Binary IQ files

POST /bin/annotate/

Description: Accepts raw IQ binary files and a `type` parameter describing the sample dtype. Each binary file is
converted to a spectrogram image internally then sent to the inference service.

Request
- Method: POST
- Path: /bin/annotate/
- Content-Type: multipart/form-data
- Form fields:
  - files: one or more file fields (raw .bin files)
  - type (optional, query/form param): one of `float16`, `float32`, `int8`, `int16` (default `float16`)

### API: Transcom CSV spectrum scans

POST /transcom_csv/annotate/

Description: Accepts Transcom-formatted CSV file(s) that contain time/frequency measurements. The server will
convert CSV rows into spectrogram images (binned by `max_milliseconds`) and run inference on each generated image.

Request
- Method: POST
- Path: /transcom_csv/annotate/
- Content-Type: multipart/form-data
- Form fields:
  - files: one or more CSV files
  - max_milliseconds: integer form field controlling how many milliseconds per generated image (default 10000)

### Response

Success response (200 JSON):

```json
{
  "results": [
    {
      "filename": "img1.jpg",
      "labels_yolo": [
        "0 0.512345 0.423456 0.123456 0.234567 0.9200",
        "..."
      ],
      "annotated_image_base64": "<base64-string>",
      "status_code": 200
    },
    {
      "filename": "img2.png",
      "error": "Error during processing: ...",
      "status_code": 500
    }, 
    {
        "..."
    }
  ]
}
```

Notes
- `labels_yolo` contains strings in the format: `class_id x_center y_center width height confidence` (normalized 0..1).
- `annotated_image_base64` is a JPEG image encoded as base64.
