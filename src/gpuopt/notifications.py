from __future__ import annotations

import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Any

import httpx

from .schemas import NotificationChannel, NotificationChannelType, NotificationMessage

logger = logging.getLogger(__name__)


@dataclass
class NotificationResult:
    success: bool
    message: str = ""
    error: str = ""


class NotificationBackend(ABC):
    @abstractmethod
    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        ...


class SlackBackend(NotificationBackend):
    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        webhook_url = channel.config.get("webhook_url", "")
        if not webhook_url:
            return NotificationResult(False, error="No slack webhook_url configured")
        try:
            payload = {
                "text": f"*{subject}*\n{body}",
                "mrkdwn": True,
            }
            resp = httpx.post(webhook_url, json=payload, timeout=10)
            if resp.is_success:
                return NotificationResult(True, "Slack notification sent")
            return NotificationResult(False, error=f"Slack API error: {resp.status_code} {resp.text}")
        except Exception as exc:
            return NotificationResult(False, error=str(exc))


class PagerDutyBackend(NotificationBackend):
    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        routing_key = channel.config.get("routing_key", "")
        if not routing_key:
            return NotificationResult(False, error="No PagerDuty routing_key configured")
        try:
            dedup_key = channel.config.get("dedup_key", f"gpuopt-{channel.id}")
            severity = channel.config.get("severity", "warning")
            payload = {
                "routing_key": routing_key,
                "event_action": "trigger",
                "dedup_key": dedup_key,
                "payload": {
                    "summary": subject[:1024],
                    "source": "gpuopt-backend",
                    "severity": severity,
                    "custom_details": {"body": body[:10240]},
                },
            }
            resp = httpx.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=10)
            if resp.is_success:
                return NotificationResult(True, "PagerDuty alert sent")
            return NotificationResult(False, error=f"PagerDuty API error: {resp.status_code} {resp.text}")
        except Exception as exc:
            return NotificationResult(False, error=str(exc))


class OpsGenieBackend(NotificationBackend):
    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        api_key = channel.config.get("api_key", "")
        if not api_key:
            return NotificationResult(False, error="No OpsGenie api_key configured")
        try:
            url = channel.config.get("api_url", "https://api.opsgenie.com/v2/alerts")
            priority = channel.config.get("priority", "P3")
            payload = {
                "message": subject[:130],
                "description": body,
                "priority": priority,
                "source": "gpuopt-backend",
                "tags": channel.config.get("tags", ["gpuopt"]),
            }
            headers = {"Authorization": f"GenieKey {api_key}", "Content-Type": "application/json"}
            resp = httpx.post(url, json=payload, headers=headers, timeout=10)
            if resp.is_success:
                return NotificationResult(True, "OpsGenie alert sent")
            return NotificationResult(False, error=f"OpsGenie API error: {resp.status_code} {resp.text}")
        except Exception as exc:
            return NotificationResult(False, error=str(exc))


class WebhookBackend(NotificationBackend):
    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        url = channel.config.get("url", "")
        if not url:
            return NotificationResult(False, error="No webhook url configured")
        try:
            method = channel.config.get("method", "POST").upper()
            headers = channel.config.get("headers", {"Content-Type": "application/json"})
            payload = channel.config.get("template", {})
            if callable(payload):
                payload = payload(subject, body)
            if isinstance(payload, dict) and "subject" not in payload:
                payload = {**payload, "subject": subject, "body": body}
            resp = httpx.request(method, url, json=payload, headers=headers, timeout=10)
            if resp.is_success:
                return NotificationResult(True, f"Webhook sent to {url}")
            return NotificationResult(False, error=f"Webhook error: {resp.status_code} {resp.text}")
        except Exception as exc:
            return NotificationResult(False, error=str(exc))


class EmailBackend(NotificationBackend):
    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        smtp_host = channel.config.get("smtp_host", "")
        smtp_port = channel.config.get("smtp_port", 587)
        smtp_user = channel.config.get("smtp_user", "")
        smtp_password = channel.config.get("smtp_password", "")
        from_addr = channel.config.get("from_addr", smtp_user)
        to_addrs = channel.config.get("to_addrs", [])
        use_tls = channel.config.get("use_tls", True)
        if not smtp_host or not to_addrs:
            return NotificationResult(False, error="Email channel missing smtp_host or to_addrs")
        try:
            msg = MIMEText(body, "plain")
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = ", ".join(to_addrs) if isinstance(to_addrs, list) else to_addrs
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if use_tls:
                    server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            return NotificationResult(True, f"Email sent to {to_addrs}")
        except Exception as exc:
            return NotificationResult(False, error=str(exc))


class NotificationService:
    def __init__(self) -> None:
        self._backends: dict[NotificationChannelType, NotificationBackend] = {
            NotificationChannelType.SLACK: SlackBackend(),
            NotificationChannelType.PAGERDUTY: PagerDutyBackend(),
            NotificationChannelType.OPSGENIE: OpsGenieBackend(),
            NotificationChannelType.WEBHOOK: WebhookBackend(),
            NotificationChannelType.EMAIL: EmailBackend(),
        }

    def register_backend(self, channel_type: NotificationChannelType, backend: NotificationBackend) -> None:
        self._backends[channel_type] = backend

    def send(self, channel: NotificationChannel, subject: str, body: str) -> NotificationResult:
        backend = self._backends.get(channel.channel_type)
        if backend is None:
            return NotificationResult(False, error=f"No backend for channel type: {channel.channel_type}")
        return backend.send(channel, subject, body)

    def send_test(self, channel: NotificationChannel) -> NotificationResult:
        return self.send(
            channel,
            subject="GPUOpt Test Notification",
            body="This is a test message from GPUOpt to verify your notification channel configuration.",
        )
