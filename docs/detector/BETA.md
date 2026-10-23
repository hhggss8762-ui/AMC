# Signal localization beta

The beta uses one `SIGNAL` class. The protocol image uses eight protocol
classes. `/v1/infer` request and response fields are unchanged; all float
`xyxy` coordinates refer to the entire 560×560 input view, including zoom
predictions. Consumers must use the returned class mapping.

## Selected final weights

Source: `signal_detector/checkpoints/0.2.1.pth`, a 12-epoch fine-tune
(lr `1e-4`, batch 2, seed 42) promoted after labelled-test and real-capture
review. Source SHA256:
`3264716e10bbee805c838d7e33fe4df8dac7219c48b48ad7ecd87c7d6c571246`.
The independently exported `checkpoints/localization.pth` contains only the
model state; all 533 tensors were checked for exact equality to the source.
Export SHA256:
`9d0cc1082123463b4a35487a94cdb90941d4a69eba4f20e0b79384d4fddc200d`.
See `models/localization/provenance.json` and `tools/export_localization.py`.
Export does not modify the source checkpoint. The previous 0.2.0 export
(`2dde46663489053c22cb3f2004f7e2c6151d1cbfa66d8abd3aeee0f7f95735a9`, from
`occupancy_iq_training_run3/checkpoint_best_total.pth`) remains published as
`0.2.0-beta.1`.

## Published tags

Both `umdetree/signal-detector:latest` and `:0.2.0-beta.1` point to
`sha256:d45c5bfe44a5a0b30858c473d3d31c324756cdb9f3d58bc02327b060d8f35cfa`
(single `SIGNAL` class localization beta). This replaced the previous
`latest`, the eight-protocol-class GPU image, which remains available as
`umdetree/signal-detector:0.1.2-gpu`
(`sha256:27dd9792ac748a362979e887be9d644f25ac1c86d8882be1ad9b19b27e07eb21`).
Clients that hard-code the protocol class IDs must pin a version tag.
To restore the old default:

```sh
docker pull umdetree/signal-detector:0.1.2-gpu
docker tag umdetree/signal-detector:0.1.2-gpu umdetree/signal-detector:latest
docker push umdetree/signal-detector:latest
```

## Build

Run from the standalone signal_detector repository:

The tested beta image reuses the published GPU runtime pinned by digest in
`Dockerfile.beta.runtime`. This avoids dependency downloads and verifies the
exported weight hash during build:

```sh
docker build -f Dockerfile.beta.runtime -t umdetree/signal-detector:0.2.1 .
```

For a complete dependency rebuild, use the original Dockerfile:

```sh
docker build --build-arg MODEL_CONFIG=models/localization/model.yaml \
  --build-arg MODEL_CHECKPOINT=checkpoints/localization.pth \
  --build-arg MODEL_FILENAME=localization.pth \
  --build-arg COMPOSE_FILE=docker-compose.beta.yml \
  --build-arg MODEL_SHA256=9d0cc1082123463b4a35487a94cdb90941d4a69eba4f20e0b79384d4fddc200d \
  --build-arg IMAGE_VERSION=0.2.1 \
  -t umdetree/signal-detector:0.2.1 .
docker run --rm -p 5555:5555 -e MODEL_DEVICE=cpu \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  umdetree/signal-detector:0.2.1
```

Use `--gpus all -e MODEL_DEVICE=cuda` for final GPU validation. For CPU-only
builds, supply `TORCH_SUFFIX=+cpu` and
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.

Alternatively, `docker compose -f docker-compose.beta.yml up --build -d`
builds the beta profile and runs on CPU. `MODEL_CHECKPOINT_PATH` may override
the configured checkpoint with an absolute local path for development.
For NVIDIA Docker hosts, apply the GPU override:
`docker compose -f docker-compose.beta.yml -f docker-compose.beta.gpu.yml up -d`.

## Adaptive inference

`inference.adaptive_zoom` in the model YAML controls one optional refinement
pass. The beta uses `input_rendering: scalar_tf`; toggling zoom does not change
preprocessing. Existing protocol profiles retain their legacy renderer.
Many boxes or many thin frequency-axis boxes trigger overlapping
frequency crops from the original TF matrix. Crop selection is based on box
geometry, not RF center frequency or an absolute bandwidth cutoff. Initial
thresholds were retained after final-checkpoint validation on 300 held-out
validation images: recall@0.5 0.8250 to 0.9006, FP/image 2.5233 to 2.0167,
with 72 images triggering zoom. These are mixed IQ-derived/replay examples;
they do not establish generalization to arbitrary signal families.

Incomplete predictions at interior crop edges are rejected; global boxes
provide fallback for wide signals. Complete local predictions are mapped back
to full-image coordinates and deduplicated. Counts, triggers and crop rows
are logged. Small TF matrices cannot recover detail through enlargement.
IQ uses `MODEL_NFFT` (default 2048), half-window overlap, PSD in dB, and the
same scalar-matrix renderer as TF input. Unknown global misses cannot trigger
zoom; evaluate non-triggered examples as well as zoomed examples.

For IQ, STFT time-cell coordinates are projected back to capture-relative
time before returning `xyxy`; calibration continues to describe the complete
capture duration. TF coordinates refer to the supplied matrix extent.
When replacing a broad coarse box with several narrow local boxes, fusion
requires a resolved frequency valley in the original TF (`split_gap_db`,
default 3 dB). This check does not participate in the zoom trigger.

The adaptive path currently applies to structured IQ/TF `/v1/infer` requests.
Legacy annotation endpoints retain their single-image inference path.

```sh
python -m pytest tests/test_adaptive.py -q
python tests/cpu_beta_smoke.py --output outputs/beta_smoke
python tests/cpu_beta_smoke.py --device cuda --output outputs/beta_gpu_smoke
```

## Declared-band hard windows (851-866 MHz)

`/v1/infer` accepts an optional `metadata.specific_band` string. When it
declares the supported `851MHz-866MHz` band, the request unconditionally runs
fixed 2.5 MHz frequency windows instead of the adaptive box-count trigger.
Captures not wider than 2.5 MHz stay a single window. Windows are planned with
250 kHz overlap and 25 kHz grid snapping and rendered from the original TF
matrix; a box belongs to the window whose core band contains its centre, and
adjacent-window boxes that touch the same seam and overlap in time are joined
(union-find, mirroring `tools/occupancy_dataset/tiling.py`). Response fields
are unchanged; the mode, span, window count and seam merges are logged
(`fixed_inference mode=hard_2p5mhz ...`). Without `specific_band` the
adaptive behavior is unchanged. `tf_intensity` requests in this mode require
`freq_start_hz`/`freq_end_hz`; IQ requests use `sample_rate_hz`.

## Development validation (epoch-two snapshot, CPU)

These are individual diagnostic examples, not population performance estimates:

| Input | Global → adaptive | Zoom |
| --- | --- | --- |
| Real 320 kSPS IQ, first 100 ms | 8 → 8 boxes (no ground truth) | none |
| IQ-derived dense test_iq_000004, 74 GT | recall@0.5 0.8243 → 0.9459; FP 26 → 15 | 5 crops |
| IQ-derived sparse test_iq_000012, 2 GT | recall@0.5 1.0 → 1.0; FP 0 → 0 | none |

CPU global forward took approximately 2.4 seconds with two threads; five
additional forwards took approximately 12 seconds. Old synthetic 15 MHz TF
examples still performed poorly (saturated recall 0 → 0.2324; sparse recall
remained zero). Those results do not establish cross-band generalization.
These historical diagnostics used the development snapshot, not the final
checkpoint reported below.

## Final checkpoint validation (2026-09-30)

Threshold 0.3, one zoom level, configuration retained after validation;
the independent test set was not used to tune thresholds. Raw global means
one forward without the service's duplicate cleanup.

| Dataset | Images / zoomed | Raw global recall@0.5 | Adaptive recall@0.5 | Raw global FP/image | Adaptive FP/image |
| --- | --- | --- | --- | --- | --- |
| Validation | 300 / 72 | 0.8250 | 0.9006 | 2.5233 | 2.0167 |
| Test | 300 / 74 | 0.8217 | 0.9043 | 2.7200 | 2.1967 |

Test recall@0.75 improved from 0.7613 to 0.8722. The merge-rate diagnostic
increased slightly from 0.0090 to 0.0118. Against cleaned global predictions,
test recall improved from 0.8212 to 0.9043 and FP/image from 2.2067 to 2.1967.
All 600 TF renders exactly matched their exported training-style RGB inputs.
These are mixed IQ-derived/legacy-replay narrowband data, not proof of
arbitrary-band or wideband generalization.

On RTX 3060 Laptop 6 GB (native Windows PyTorch 2.9/cu126), test mean
global forward was 92 ms. Render/global/refinement mean was 243 ms, p95
682 ms (includes RGB equivalence-check overhead, excludes HTTP). Peak Torch
allocated memory was 683 MiB; this is not total process/device memory.

Six unlabelled windows from the same 320 kSPS capture returned 8/10/6/10/8/6
boxes without triggering zoom. They are not independent recordings or a
recall measurement. Real model multipart HTTP checks passed on native CPU,
native GPU, Docker CPU, and Docker GPU. Container checks included dense,
sparse, empty, and real IQ inputs plus malformed payload rejection.
GPU container dense HTTP was 0.84 s; sparse/empty/real were 0.10–0.17 s.
Two-thread CPU container dense HTTP was 15.68 s; others were 2.56–2.71 s.

Parent-repository reports and vertically stacked clean/global/adaptive
previews: `outputs/signal_detector_beta_final/`. Reproduce dataset evaluation
from the parent repository with:

```sh
python -m tools.occupancy_dataset.evaluate_service_beta \
  --checkpoint signal_detector/checkpoints/localization.pth \
  --dataset outputs/occupancy_iq_finetune_v1 --split test \
  --output outputs/signal_detector_beta_final/test --device cuda
```

## 0.2.1 weights update (2026-10-09)

The 0.2.1 fine-tune replaced the 0.2.0 export in the image. Same test split,
same thresholds, same full service pipeline (raw global means one forward
without duplicate cleanup):

| Metric (test, 300 images) | 0.2.0 | 0.2.1 |
| --- | --- | --- |
| Raw global recall@0.5 | 0.8217 | 0.8463 |
| Raw global recall@0.75 | 0.7613 | 0.7934 |
| Raw global FP/image | 2.7200 | 2.5867 |
| Adaptive recall@0.5 | 0.9043 | 0.9204 |
| Adaptive recall@0.75 | 0.8722 | 0.9004 |
| Adaptive FP/image | 2.1967 | 1.9667 |
| Adaptive merge rate | 0.0118 | 0.0054 |

On the unlabelled 19.2 MSPS wideband capture, hard mode returned 136 boxes
over 9 windows with 2 seam merges (0.2.0: 122 boxes, 14 seam merges); real
captures carry no ground truth. These remain narrowband/wideband diagnostic
data, not accuracy claims for arbitrary bands.
