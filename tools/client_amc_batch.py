"""Submit prepared signal IQ files and their JSON sidecars using only the stdlib."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


def build_multipart(
    frame_id: str, signals: list[tuple[Path, Path]]
) -> tuple[bytes, str]:
    """Keep each IQ/metadata pair together and number pairs in submission order."""
    boundary = "amc_" + uuid4().hex
    chunks = []

    def add_part(name: str, content: bytes, content_type: str, filename: str | None = None):
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        chunks.append(
            f"--{boundary}\r\n{disposition}\r\nContent-Type: {content_type}\r\n\r\n".encode("ascii")
        )
        chunks.extend((content, b"\r\n"))

    add_part("frame_id", frame_id.encode("utf-8"), "text/plain; charset=utf-8")
    for index, (iq_path, metadata_path) in enumerate(signals):
        with metadata_path.open(encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
        metadata_bytes = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
        add_part(f"iq_{index}", iq_path.read_bytes(), "application/octet-stream", f"signal_{index}.iq")
        # A text part, not a second uploaded file. One JSON describes exactly one IQ.
        add_part(f"metadata_{index}", metadata_bytes, "application/json; charset=utf-8")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def main(argv: list[str] | None = None) -> int:
    # Keep Chinese identifiers readable in Windows terminals and redirected logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5555/v1/amc/infer")
    parser.add_argument("--frame-id", required=True)
    parser.add_argument(
        "--signal", nargs=2, action="append", default=[], metavar=("IQ_PATH", "JSON_PATH"),
        help="A prepared binary IQ file and its parameter JSON; repeat once per signal."
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--output", type=Path, help="Optionally save the response; never overwrite a file.")
    args = parser.parse_args(argv)
    parsed_url = urlsplit(args.url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        parser.error("--url must be an http:// or https:// service URL.")
    if not args.frame_id.strip() or len(args.frame_id) > 128:
        parser.error("--frame-id must contain 1 to 128 characters and cannot be blank.")
    if args.timeout <= 0 or args.timeout == float("inf") or args.timeout != args.timeout:
        parser.error("--timeout must be positive and finite.")
    if args.output is not None:
        if args.output.exists():
            parser.error(f"Refusing to overwrite: {args.output}")
        if not args.output.parent.is_dir():
            parser.error(f"Output parent directory does not exist: {args.output.parent}")
    signals = [(Path(iq), Path(metadata)) for iq, metadata in args.signal]
    try:
        body, content_type = build_multipart(args.frame_id, signals)
        request = Request(args.url, data=body, headers={"Content-Type": content_type}, method="POST")
        # The receiver calls an explicit service endpoint, usually on the LAN.
        # Do not silently route it through HTTP(S)_PROXY environment settings.
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
        sys.stdout.write(rendered)
    except HTTPError as exc:
        details = exc.read(65536).decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} {exc.reason}\n{details}", file=sys.stderr)
        return 1
    except (OSError, URLError, ValueError) as exc:
        print(f"Batch request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
