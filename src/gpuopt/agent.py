from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import socket
import subprocess as sp
import sys
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_AGENT_ID: str = ""
_REGISTERED_ENDPOINT: str = ""
_VERSION: str = ""


def _load_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.1.0"


def _get_hostname() -> str:
    return socket.gethostname()


def _detect_capabilities() -> list[str]:
    caps = ["hardware_detection", "telemetry_push"]
    try:
        sp.run(["nvidia-smi"], capture_output=True, timeout=5)
        caps.append("nvidia_gpu")
    except Exception:
        pass
    try:
        import pynvml
        caps.append("nvml")
    except ImportError:
        pass
    return caps


def detect_local_hardware() -> dict[str, Any]:
    from .hardware_detector import detect_local_hardware as _detect
    return _detect()


def register_agent(server_endpoint: str, api_key: str = "",
                   cert_file: str = "", key_file: str = "") -> dict[str, Any]:
    global _AGENT_ID, _REGISTERED_ENDPOINT
    hostname = _get_hostname()
    version = _load_version()
    capabilities = _detect_capabilities()
    payload = json.dumps({
        "hostname": hostname,
        "version": version,
        "capabilities": capabilities,
        "labels": {"hostname": hostname},
    }).encode("utf-8")
    url = f"{server_endpoint.rstrip('/')}/api/v1/agents/register"
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    if cert_file and key_file:
        import ssl
        ctx = ssl.create_default_context()
        ctx.load_cert_chain(cert_file, key_file)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    else:
        resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode("utf-8"))
    _AGENT_ID = result.get("agent_id", "")
    _REGISTERED_ENDPOINT = server_endpoint.rstrip("/")
    logger.info("Agent registered: %s as %s", result.get("agent_id"), hostname)
    return result


def run_agent_once() -> None:
    hardware = detect_local_hardware()
    print(json.dumps(hardware, indent=2, default=str))


def run_agent_daemon(
    endpoint: str,
    interval_seconds: int = 60,
    api_key: str = "",
    cert_file: str = "",
    key_file: str = "",
    auto_register: bool = True,
) -> None:
    stop = False

    def _handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        logger.info("Agent shutting down")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    agent_id = ""
    if auto_register:
        try:
            result = register_agent(endpoint, api_key, cert_file, key_file)
            agent_id = result.get("agent_id", "")
        except Exception as exc:
            logger.error("Registration failed: %s", exc)

    server_base = endpoint.rstrip("/")
    telemetry_url = f"{server_base}/api/v1/agents/{agent_id}/telemetry" if agent_id else endpoint
    heartbeat_url = f"{server_base}/api/v1/agents/{agent_id}/heartbeat" if agent_id else ""

    logger.info("Agent daemon started — endpoint=%s interval=%ds agent_id=%s",
                endpoint, interval_seconds, agent_id or "unregistered")

    seq = 0
    push_count = 0
    while not stop:
        try:
            hardware = detect_local_hardware()
            payload = json.dumps(hardware, default=str).encode("utf-8")
            req = urllib.request.Request(telemetry_url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            if api_key:
                req.add_header("X-API-Key", api_key)
            if agent_id:
                req.add_header("X-Agent-Id", agent_id)

            ctx = None
            if cert_file and key_file:
                import ssl
                ctx = ssl.create_default_context()
                ctx.load_cert_chain(cert_file, key_file)
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                if resp.status == 200:
                    push_count += 1
                    if push_count % 10 == 0:
                        logger.info("Telemetry pushed (%d pushes)", push_count)
                else:
                    logger.warning("Server returned %d", resp.status)

            if heartbeat_url and push_count % max(interval_seconds // 30, 1) == 0:
                seq += 1
                hb_payload = json.dumps({
                    "agent_id": agent_id,
                    "sequence": seq,
                    "load": {"cpu_percent": hardware.get("cpu_percent", 0)},
                    "metrics_summary": {"gpu_count": hardware.get("gpu_count", 0)},
                }).encode("utf-8")
                hb_req = urllib.request.Request(heartbeat_url, data=hb_payload, method="POST")
                hb_req.add_header("Content-Type", "application/json")
                if api_key:
                    hb_req.add_header("X-API-Key", api_key)
                urllib.request.urlopen(hb_req, timeout=10, context=ctx)

        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                logger.error("Authentication failed — check API key")
            elif exc.code == 404:
                logger.error("Agent not found on server — re-registering")
                if auto_register:
                    try:
                        result = register_agent(endpoint, api_key, cert_file, key_file)
                        agent_id = result.get("agent_id", "")
                        telemetry_url = f"{server_base}/api/v1/agents/{agent_id}/telemetry"
                        heartbeat_url = f"{server_base}/api/v1/agents/{agent_id}/heartbeat"
                    except Exception as reg_exc:
                        logger.error("Re-registration failed: %s", reg_exc)
            else:
                logger.warning("HTTP error %d: %s", exc.code, exc)
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
    parser.add_argument("--cert-file", default=os.environ.get("GPUOPT_AGENT_CERT", ""),
                        help="mTLS client certificate file")
    parser.add_argument("--key-file", default=os.environ.get("GPUOPT_AGENT_KEY", ""),
                        help="mTLS client key file")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip registration handshake")
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
        run_agent_daemon(
            args.endpoint, args.interval, args.api_key,
            cert_file=args.cert_file, key_file=args.key_file,
            auto_register=not args.no_register,
        )
    else:
        run_agent_once()


if __name__ == "__main__":
    main()
