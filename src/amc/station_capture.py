"""Read the station's indexed int16 IQ capture without changing source files.

The 50-byte IQ index layout follows SpectrumOccupancy/tools/diagnose_921data.py
and is cross-checked against every payload offset/count, not just its first row.
Index timestamps describe device records; processing time follows sample order.
"""
from pathlib import Path
import hashlib
import struct
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_station_capture(directory):
    directory = Path(directory).resolve()
    indices = list(directory.glob("Iq_*.ix"))
    if len(indices) != 1:
        raise ValueError("Expected exactly one Iq_*.ix; select an unambiguous capture directory.")
    index_path = indices[0]
    payload_path = index_path.with_suffix(".md")
    index = index_path.read_bytes()
    if not index or len(index) % 50:
        raise ValueError("IQ index must contain complete 50-byte records.")
    rows = []
    expected_offset = 0
    for pos in range(0, len(index), 50):
        tick, seq, offset = struct.unpack_from("<qqq", index, pos)
        center, bandwidth, fs, count = struct.unpack_from("<dffi", index, pos + 25)
        if (not np.isfinite([center, bandwidth, fs]).all()
                or fs <= 0 or bandwidth <= 0 or count <= 0):
            raise ValueError(f"Invalid IQ acquisition parameters at record {pos // 50}.")
        if offset != expected_offset:
            raise ValueError(f"Non-contiguous IQ payload at record {pos // 50}.")
        if rows and (seq != rows[-1][1] + 1 or tick < rows[-1][0]):
            raise ValueError("IQ record sequence/timestamps are not ordered.")
        if rows and (center, bandwidth, fs) != rows[0][3:6]:
            raise ValueError("Acquisition parameters change within this capture.")
        rows.append((tick, seq, offset, center, bandwidth, fs, count))
        expected_offset += count * 4
    if payload_path.stat().st_size != expected_offset:
        raise ValueError("IQ payload size does not match all index records.")
    center, bandwidth, fs = rows[0][3:6]
    values = np.memmap(payload_path, dtype="<i2", mode="r")
    iq = values[::2].astype(np.float32) + 1j * values[1::2].astype(np.float32)
    metadata = {"input_type": "iq", "dtype": "int16", "sample_rate_hz": fs,
                "center_freq_hz": center}
    ticks = np.array([r[0] for r in rows], dtype=np.int64)
    provenance = {
        "index_file": str(index_path), "payload_file": str(payload_path),
        "index_sha256": file_sha256(index_path),
        "payload_sha256": file_sha256(payload_path),
        "index_records": len(rows), "payload_bytes": expected_offset,
        "complex_samples": len(iq), "duration_s": len(iq) / fs,
        "sample_rate_hz": fs, "center_freq_hz": center,
        "device_bandwidth_hz": bandwidth,
        "time_axis": "sample_index / sample_rate_hz; entire payload, no frame truncation",
        "iq_convention": "little-endian int16 interleaved I,Q; complex I+jQ, existing station reader convention",
        "timestamp_span_s": int(ticks[-1] - ticks[0]) / 1e7,
        "repeated_timestamp_intervals": int(np.count_nonzero(np.diff(ticks) == 0)),
        "first_timestamp_ticks": int(ticks[0]), "last_timestamp_ticks": int(ticks[-1]),
        "continuity_note": "Record sequence and byte offsets are contiguous; index timestamps are not used as per-sample timing and do not prove RF continuity.",
    }
    return iq, metadata, provenance
