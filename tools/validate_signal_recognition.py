"""Internal HTTP regression of original detector vs integrated Docker image.

Uses existing local real IQ and prepared synthetic fixtures; never a receiver
reference-data deliverable. Run reference before integrated to avoid keeping two
large detector models on the GPU simultaneously.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

import numpy as np

from client_amc_batch import build_multipart


ROOT = Path(__file__).resolve().parents[1]
OPENER = build_opener(ProxyHandler({}))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def call(url, body=None, content_type=None, headers=None):
    supplied = dict(headers or {})
    if content_type:
        supplied["Content-Type"] = content_type
    request = Request(url, data=body, headers=supplied)
    started = time.perf_counter()
    try:
        with OPENER.open(request, timeout=180) as response:
            status, data = response.status, response.read()
    except HTTPError as exc:
        status, data = exc.code, exc.read()
    elapsed = (time.perf_counter() - started) * 1000
    return status, json.loads(data), elapsed


def get_ready(url, route):
    for attempt in range(30):
        try:
            status, result, elapsed = call(url + route)
            assert status == 200 and result["status"] in ("ok", "ready"), (status, result)
            print(f'{route}: {status}, {elapsed:.2f} ms', flush=True)
            return result, elapsed
        except URLError:
            if attempt == 29:
                raise
            time.sleep(0.5)


def detector_body(raw, metadata):
    boundary = "detector_" + uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n'
            + json.dumps(metadata) + f'\r\n--{boundary}\r\n'
            + 'Content-Disposition: form-data; name="data"; filename="input.bin"\r\n'
            + 'Content-Type: application/octet-stream\r\n\r\n').encode()
    return body + raw + f'\r\n--{boundary}--\r\n'.encode(), f'multipart/form-data; boundary={boundary}'


def detector_cases(capture):
    # Acquisition metadata already audited in the existing real-IQ adapter run.
    source = json.loads((ROOT / 'reports/real_iq_adapter_v2/source.json').read_text(encoding='utf-8'))
    payload = capture.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source['payload_sha256']
    iq_meta = json.loads((ROOT / 'reports/real_iq_adapter_v2/request_metadata.json').read_text())
    values = np.frombuffer(payload, dtype='<i2').reshape(-1, 2)
    iq = values[:, 0].astype(np.float32) + 1j * values[:, 1].astype(np.float32)
    # A compact TF matrix from this entire capture, for HTTP compatibility tests.
    # Sparse STFT columns are sufficient here; this is not a receiver benchmark.
    nfft = 2048
    starts = np.linspace(0, len(iq) - nfft, 256).astype(int)
    window = np.hanning(nfft)
    chunks = np.stack([iq[start:start+nfft] for start in starts]) * window
    spectrum = np.fft.fftshift(np.fft.fft(chunks, axis=1), axes=1)
    power = np.abs(spectrum) ** 2 / (iq_meta['sample_rate_hz'] * np.sum(window ** 2))
    matrix = (10 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))).T.astype('<f4')
    fs, center = iq_meta['sample_rate_hz'], iq_meta['center_freq_hz']
    tf_meta = {'input_type': 'tf_intensity', 'dtype': 'float32', 'value_scale': 'db',
               'shape': list(matrix.shape), 'time_window_ms': len(iq) / fs * 1000,
               'freq_start_hz': center-fs/2, 'freq_end_hz': center+fs/2}
    return [
        ('tf_adaptive', matrix.tobytes(), tf_meta),
        ('tf_specific_band', matrix.tobytes(), dict(tf_meta, specific_band='851MHz-866MHz')),
        ('original_iq', payload, iq_meta),
    ]


def run_detector(args, cases):
    output = args.output
    get_ready(args.url, '/health')
    _, cold_ms = get_ready(args.url, '/ready')
    status, schema, _ = call(args.url + '/openapi.json')
    assert status == 200
    save(output / f'{args.mode}_openapi.json', schema)
    summaries = []
    for name, raw, metadata in cases:
        body, content_type = detector_body(raw, metadata)
        status, response, elapsed = call(args.url + '/v1/infer', body, content_type)
        assert status == 200, (name, status, response)
        assert set(response) == {'model','class_names','image_width_px','image_height_px','calibration','detections'}
        assert response['class_names'] == {'0':'SIGNAL'}
        save(output / f'{args.mode}_{name}.json', response)
        save(output / f'{name}_metadata.json', metadata)
        if args.mode == 'integrated':
            original = json.loads((output / f'reference_{name}.json').read_text())
            # Same binary inputs, model, runtime and GPU: require complete JSON equality.
            assert response == original, f'Detector changed on {name}'
        print(f'{name}: {len(response["detections"])} boxes, {elapsed:.2f} ms', flush=True)
        summaries.append({'input':name, 'bytes':len(raw), 'boxes':len(response['detections']),
                          'http_ms':elapsed, 'matches_original': args.mode == 'integrated'})
    if args.mode == 'integrated':
        original = json.loads((output / 'reference_openapi.json').read_text())
        for path, operations in original['paths'].items():
            assert schema['paths'][path] == operations, f'Original route schema changed: {path}'
    return {'detector_ready_first_call_ms':cold_ms, 'detector_cases': summaries,
            'original_paths':sorted(schema['paths'])}


def run_amc(args):
    ready, ready_ms = get_ready(args.url, '/v1/amc/ready')
    assert ready['device'] == 'cuda' and len(ready['classes']) == 13
    fixture = ROOT / 'reports/amc_service_v1/internal_inputs'
    signals = [(fixture/f'signal_{i}.iq', fixture/f'signal_{i}.json') for i in range(5)]
    frame_id = 'container_smoke'
    body, content_type = build_multipart(frame_id, signals)
    status, response, _ = call(args.url + '/v1/amc/infer', body, content_type)
    expected = json.loads((ROOT / 'reports/amc_service_v1/client_response.json').read_text())
    expected['frame_id'] = frame_id
    assert status == 200 and response == expected, (status, response)
    save(args.output / 'amc_response.json', response)
    # Force the 70 selected windows to span the default 64-window network batch.
    bulk = [(fixture/'signal_0.iq', fixture/f'bulk_metadata_{i}.json') for i in range(14)]
    bulk_body, bulk_type = build_multipart('container_bulk', bulk)
    status, bulk_response, bulk_ms = call(args.url + '/v1/amc/infer', bulk_body, bulk_type)
    assert status == 200 and bulk_response == {'frame_id':'container_bulk', 'results':[
        {'signal_id':f'bulk_{i}', 'modulation_type':'AM'} for i in range(14)
    ]}, (status, bulk_response)
    save(args.output / 'amc_bulk_response.json', bulk_response)
    empty_body, empty_type = build_multipart('container_empty', [])
    status, empty, _ = call(args.url + '/v1/amc/infer', empty_body, empty_type)
    assert status == 200 and empty == {'frame_id':'container_empty','results':[]}
    status, error, _ = call(args.url + '/v1/amc/infer', b'{}', 'application/json')
    assert status == 400 and set(error) == {'detail'}
    status, error, _ = call(args.url + '/v1/amc/infer', b'', 'multipart/form-data; boundary=x',
                            {'Content-Length': str(256*1024*1024+1)})
    assert status == 413 and set(error) == {'detail'}
    too_many = [(fixture/'signal_0.iq', fixture/f'bulk_metadata_{i%14}.json') for i in range(65)]
    too_many_body, too_many_type = build_multipart('container_over_limit', too_many)
    status, error, _ = call(args.url + '/v1/amc/infer', too_many_body, too_many_type)
    assert status == 413 and set(error) == {'detail'}
    samples = []
    for _ in range(5):
        status, result, elapsed = call(args.url + '/v1/amc/infer', body, content_type)
        assert status == 200 and result == expected
        samples.append(elapsed)
    print(f'AMC: 5 mixed-length signals matched, 14-signal batch matched; warm median {statistics.median(samples):.2f} ms', flush=True)
    return {'amc_device': ready['device'], 'amc_classes':ready['classes'], 'amc_ready_first_call_ms':ready_ms,
            'amc_signals':5, 'amc_classified':3, 'amc_null':2, 'amc_matches_previous_result':True,
            'bulk_signals':14, 'bulk_windows':70, 'bulk_network_batches':2, 'bulk_http_ms':bulk_ms,
            'empty_frame_ok':True, 'malformed_400_ok':True, 'declared_body_413_ok':True,
            'signal_count_413_ok':True, 'warm_http_ms':samples,
            'warm_http_median_ms':statistics.median(samples),
            'timing_scope':'Local HTTP round trip including parsing/selection/normalization/network/fusion; excludes receiver preprocessing, detection, file reads, model cold start. Five repeats are a diagnostic, not throughput qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference','integrated'))
    parser.add_argument('--url', required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.url = args.url.rstrip('/')
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / f'{args.mode}_summary.json').exists(), 'Use a new output directory for a new validation'
    summary = run_detector(args, detector_cases(args.capture))
    if args.mode == 'integrated':
        summary.update(run_amc(args))
    summary['validation_mode'] = args.mode
    summary['capture_sha256'] = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    summary['note'] = 'Real IQ is unlabelled; this verifies regression and API behavior, not real modulation accuracy. Internal fixtures are not shipped.'
    save(args.output / f'{args.mode}_summary.json', summary)
    print('HTTP validation completed successfully.', flush=True)


if __name__ == '__main__':
    main()
