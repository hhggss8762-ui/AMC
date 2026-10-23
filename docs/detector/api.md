# Signal Detector API Reference

The service listens on port `5555` by default. The published Docker image
contains the protocol model and uses `MODEL_CONFIG_PATH=/models/model.yaml`.

The localization **beta** uses the same endpoints and response schema, with
`class_names={"0":"SIGNAL"}` instead of protocol labels. `/v1/infer` may
internally refine crowded/thin detections using one additional zoom level.
Returned floating-point `xyxy` always refers to the full 560x560 input view,
never a local crop. No zoom parameters or diagnostic fields are added to the
public API. Use the returned class names rather than hard-coded protocol IDs.
See [BETA.md](BETA.md) for deployment and server-side settings.

## Service checks

### GET /health

Process liveness check. A successful response is:

```json
{"status": "ok"}
```

### GET /ready

Loads the configured model on the first request, runs one dummy 560x560
inference to initialize CUDA/cuDNN, and reports model readiness. The endpoint
returns HTTP 503 until the model can be loaded.

```json
{
  "status": "ready",
  "model": "protocol-classification",
  "checkpoint": "/models/rfdetr.pth"
}
```

## Structured inference

### POST /v1/infer

Accepts a binary payload in the `data` form field and a JSON metadata string in
the `metadata` form field. It returns structured detections and does not
return an annotated image.

#### IQ input

The IQ payload contains interleaved I/Q values.

```json
{
  "input_type": "iq",
  "dtype": "float16",
  "sample_rate_hz": 100000000,
  "center_freq_hz": 2450000000
}
```

The service uses `NFFT=2048`, `noverlap=1024`, dB scaling, the `viridis`
colormap, and the same frequency orientation as `ax.specgram`.

#### TF intensity input

The payload is a row-major flattened matrix with `shape[0] * shape[1]` values.
Frequency rows must be ordered from low to high; the service flips them for
display. The optional `value_scale` defaults to `db` and can be:

- `db`, `dbm`, or `dbw`: values are already logarithmic values.
- `linear`: values are linear power/intensity and the service applies `10 * log10`.

For an offline matrix generated from `matplotlib.specgram`, omit `value_scale`
or use `db`, preserve the dB values before downsampling, and use `float16` to
reduce the upload size.

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

Both input types are converted to the same `560x560` RGB model image. The
input matrix shape does not change the model image size.

For latency-sensitive deployments, use a precomputed TF matrix or send an
already rendered image to `/images/annotate/`. Raw IQ requests include the
server-side spectrogram calculation. TF rendering normalizes the complete
finite dB range, downscales the scalar matrix to `560x560` with bilinear
resampling, and then applies the `viridis` lookup table, preserving the dB,
dynamic-range, colormap, and orientation mapping of `ax.specgram`.

Payloads larger than `TF_MAX_ELEMENTS` (default `20000000`) are rejected with
HTTP 400. Set the environment variable to change that limit.

#### Optional target band (`specific_band`)

`metadata` may include an optional `specific_band` string. When it declares a
supported band, the service unconditionally refines the request with fixed
2.5 MHz frequency windows instead of the adaptive box-count trigger:

- A capture wider than 2.5 MHz is split into uniform windows with 250 kHz
  overlap; a narrower capture stays a single window.
- Every window is rendered and scored independently; seam boxes are merged
  back into full-image `xyxy` coordinates.
- Supported value for the localization beta: `851MHz-866MHz`
  (case-insensitive, spaces allowed).
- Without `specific_band` the default adaptive behavior is unchanged.
- Unsupported or malformed values, and `tf_intensity` requests without
  `freq_start_hz`/`freq_end_hz`, are rejected with HTTP 400.
- The response schema is unchanged; the mode is recorded in the service log.

```json
{
  "input_type": "tf_intensity",
  "dtype": "float16",
  "shape": [4800, 100],
  "freq_start_hz": 851000000,
  "freq_end_hz": 866000000,
  "specific_band": "851MHz-866MHz"
}
```

#### Response

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

The `detections[].xyxy` field is frozen as a four-value pixel box:

```text
[x1, y1, x2, y2]
```

`x1` is the left boundary, `y1` is the top boundary, `x2` is the right
boundary, and `y2` is the bottom boundary. The origin `(0, 0)` is the
top-left corner of the `560x560` model image. `x` increases to the right and
`y` increases downward. Values are floating-point pixels, not normalized
coordinates. The width is `x2 - x1` and the height is `y2 - y1`.

The physical meaning is obtained from the response `calibration` fields. For
`[x1, y1, x2, y2]`:

```text
start_time_ms = x1 / image_width_px * time_window_ms
end_time_ms = x2 / image_width_px * time_window_ms
high_freq_hz = freq_end_hz - y1 / image_height_px * (freq_end_hz - freq_start_hz)
low_freq_hz = freq_end_hz - y2 / image_height_px * (freq_end_hz - freq_start_hz)
duration_ms = (x2 - x1) / image_width_px * time_window_ms
center_freq_hz = freq_end_hz - ((y1 + y2) / 2) / image_height_px * (freq_end_hz - freq_start_hz)
bandwidth_hz = (y2 - y1) / image_height_px * (freq_end_hz - freq_start_hz)
```

The horizontal axis is time and the vertical axis is frequency. The top of
the image represents `freq_end_hz`; the bottom represents `freq_start_hz`.
`start_time_ms` and `end_time_ms` are relative to the beginning of the input
window. Physical conversion requires `time_window_ms`, `freq_start_hz`, and
`freq_end_hz` in `calibration`; if any is `null`, only pixel coordinates are
available. Clients should clamp coordinates to the image bounds before drawing
when necessary. Future API changes will not change the order or meaning of
`xyxy`, and will not replace it with `xywh` or normalized coordinates.

## Legacy endpoints

### API: Images

POST /images/annotate/

Description: Accepts one or more image files and returns, for each image, a base64-encoded annotated image and
YOLO-like labels.

This endpoint expects an already rendered image. It includes annotation and
JPEG/base64 encoding in the response, but does not perform IQ-to-spectrogram
conversion. Use `/v1/infer` when structured detections or TF input are needed.

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
