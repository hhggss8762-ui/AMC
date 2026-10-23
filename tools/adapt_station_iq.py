"""Full station IQ -> original 0.2.1 HTTP detector -> per-box AMC, no truncation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import urllib.request
import uuid
from amc.detector_adapter import DetectionIQAdapter
from amc.station_capture import load_station_capture, file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def post_iq(url, payload_path, metadata, timeout):
    boundary = "amc-" + uuid.uuid4().hex
    pieces = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n".encode(),
              json.dumps(metadata).encode(),
              f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"data\"; filename=\"capture.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode(),
              payload_path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()]
    request = urllib.request.Request(url.rstrip("/") + "/v1/infer", data=b"".join(pieces),
                                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="models/amc_baseline_v1.pt")
    parser.add_argument("--detector-url", help="Original 0.2.1 service URL.")
    parser.add_argument("--detector-json", help="Replay a previously saved response for exactly this entire IQ.")
    parser.add_argument("--specific-band", help="Explicit detector strategy hint; omitted by default.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if bool(args.detector_url) == bool(args.detector_json):
        parser.error("Provide exactly one of --detector-url or --detector-json.")
    if args.detector_json and args.specific_band:
        parser.error("--specific-band applies only to live detection.")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    iq, metadata, source = load_station_capture(args.capture)
    if args.specific_band:
        metadata["specific_band"] = args.specific_band
    output.mkdir(parents=True)
    write_json(output / "source.json", source)
    write_json(output / "request_metadata.json", metadata)
    print(json.dumps({"source_samples": len(iq), "duration_s": source["duration_s"], "sample_rate_hz": source["sample_rate_hz"]}), flush=True)
    started = time.perf_counter()
    if args.detector_json:
        detector = json.loads(Path(args.detector_json).read_text(encoding="utf-8"))
    else:
        detector = post_iq(args.detector_url, Path(source["payload_file"]), metadata, args.timeout)
    detector_seconds = time.perf_counter() - started
    write_json(output / "detector_response.json", detector)
    print(json.dumps({"detector_seconds": detector_seconds, "boxes": len(detector["detections"])}), flush=True)
    started = time.perf_counter()
    adapter = DetectionIQAdapter(args.checkpoint, args.device)
    # Warm CUDA and kernels separately; do not include startup in per-box timings.
    import numpy as np
    adapter.classifier.window_classifier.predict(np.ones((1, 2, 2048), dtype=np.float32))
    adapter.classifier._synchronize()
    model_ready_seconds = time.perf_counter() - started
    started = time.perf_counter()
    enriched, diagnostics = adapter.enrich(iq, source["sample_rate_hz"], source["center_freq_hz"], detector)
    adaptation_seconds = time.perf_counter() - started
    # An explicit audit: the original response must survive after removing two fields.
    original = json.loads(json.dumps(enriched))
    for row in original["detections"]:
        row.pop("modulation_type")
        row.pop("modulation_score")
    if original != detector:
        raise AssertionError("Adapter changed the detector contract.")
    if source["payload_sha256"] != file_sha256(source["payload_file"]):
        raise AssertionError("Source IQ changed during processing.")
    write_json(output / "response.json", enriched)
    write_json(output / "diagnostics.json", diagnostics)
    summary = {
        "source": source, "detector_url": args.detector_url,
        "detector_response_replay": args.detector_json,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(adapter.classifier.device), "detections": len(diagnostics),
        "statuses": dict(Counter(d["status"] for d in diagnostics)),
        "predicted_types": dict(Counter(d.get("class_name") for d in diagnostics if d.get("class_name"))),
        "detector_http_or_replay_seconds": detector_seconds,
        "amc_load_and_warmup_seconds": model_ready_seconds,
        "adaptation_seconds": adaptation_seconds,
        "original_response_preserved": True, "source_payload_unchanged": True,
        "accuracy": None,
        "note": "Unlabelled real capture. Scores are uncalibrated; predictions are not ground truth. HTTP/detector timing is separate from AMC per-box timings.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
