from __future__ import annotations

import json
import logging
import os
import signal
import subprocess as sp
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


def detect_local_hardware() -> dict[str, Any]:
    from .hardware_detector import detect_local_hardware as _detect
    return _detect()


def run_agent_once() -> None:
    hardware = detect_local_hardware()
    print(json.dumps(hardware, indent=2, default=str))


def run_agent_daemon(
    endpoint: str,
    interval_seconds: int = 60,
    api_key: str = "",
) -> None:
    import urllib.request

    stop = False

    def _handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        logger.info("Agent shutting down")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Agent daemon started — endpoint=%s interval=%ds", endpoint, interval_seconds)
    while not stop:
        try:
            hardware = detect_local_hardware()
            payload = json.dumps(hardware, default=str).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    logger.info("Telemetry pushed to %s", endpoint)
                else:
                    logger.warning("Server returned %d", resp.status)
        except Exception as exc:
            logger.error("Push failed: %s", exc)
        for _ in range(interval_seconds):
            if stop:
                break
            time.sleep(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GPUOpt hardware detection agent")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--endpoint", default=os.environ.get("GPUOPT_AGENT_ENDPOINT", ""),
                        help="HTTP endpoint to push telemetry")
    parser.add_argument("--interval", type=int, default=60,
                        help="Push interval in seconds (default 60)")
    parser.add_argument("--api-key", default=os.environ.get("GPUOPT_API_KEY", ""),
                        help="API key for authentication")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.daemon:
        if not args.endpoint:
            logger.error("--endpoint required in daemon mode")
            sys.exit(1)
        run_agent_daemon(args.endpoint, args.interval, args.api_key)
    else:
        run_agent_once()


if __name__ == "__main__":
    main()
