"""Container is ready only when both original detector and AMC are ready."""
import json
import sys
from urllib.request import ProxyHandler, build_opener


def main():
    opener = build_opener(ProxyHandler({}))
    for route in ("/ready", "/v1/amc/ready"):
        with opener.open("http://127.0.0.1:5555" + route, timeout=55) as response:
            if response.status != 200 or json.load(response).get("status") != "ready":
                raise RuntimeError(f"Not ready: {route}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Service readiness check failed: {exc}", file=sys.stderr)
        sys.exit(1)
