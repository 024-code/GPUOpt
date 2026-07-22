from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from .models import (
    ApprovalRequest,
    ApprovalStatus,
    GovernanceConfig,
    ModelActionClass,
    ModelStatus,
    ModelVersion,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class ApprovalManager:
    def __init__(self, registry: ModelRegistry, config: GovernanceConfig | None = None) -> None:
        self._registry = registry
        self._config = config or GovernanceConfig()
        self._requests: dict[UUID, ApprovalRequest] = {}

    def requires_approval(self, version: ModelVersion) -> bool:
        return version.action_class in self._config.high_impact_action_classes

    def request_approval(
        self,
        version: ModelVersion,
        requested_by: str = "",
        certification_days: int | None = None,
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            model_version_id=version.id,
            action_class=version.action_class,
            requested_by=requested_by or version.owner,
            certification_period_days=certification_days or 90,
        )
        self._requests[req.id] = req
        logger.info("Approval requested for %s v%s by %s",
                     version.model_name, version.version, req.requested_by)
        return req

    def approve(
        self,
        request_id: UUID,
        reviewer: str,
        notes: str = "",
        certification_days: int | None = None,
    ) -> ApprovalRequest | None:
        req = self._requests.get(request_id)
        if req is None:
            return None

        days = certification_days or req.certification_period_days
        req.status = ApprovalStatus.APPROVED
        req.reviewed_by = reviewer
        req.reviewed_at = datetime.now(timezone.utc)
        req.review_notes = notes
        req.certified_until = datetime.now(timezone.utc) + timedelta(days=days)

        version = self._registry.get(req.model_version_id)
        if version:
            version.approved_by = reviewer
            version.approved_at = req.reviewed_at
            version.approval_note = notes
            version.certification_days = days
            version.certified_until = req.certified_until

        logger.info("Approved %s for %s v%s (certified %d days)",
                     request_id, version.model_name if version else "?", version.version if version else "?", days)
        return req

    def reject(self, request_id: UUID, reviewer: str, notes: str = "") -> ApprovalRequest | None:
        req = self._requests.get(request_id)
        if req is None:
            return None

        req.status = ApprovalStatus.REJECTED
        req.reviewed_by = reviewer
        req.reviewed_at = datetime.now(timezone.utc)
        req.review_notes = notes

        version = self._registry.get(req.model_version_id)
        if version:
            self._registry.update_status(version.id, ModelStatus.REJECTED)

        logger.info("Rejected %s for %s by %s: %s", request_id,
                     version.model_name if version else "?", reviewer, notes)
        return req

    def check_recertification(self) -> list[ApprovalRequest]:
        now = datetime.now(timezone.utc)
        due: list[ApprovalRequest] = []
        for req in self._requests.values():
            if req.status != ApprovalStatus.APPROVED:
                continue
            if req.certified_until and req.certified_until < now:
                req.status = ApprovalStatus.EXPIRED
                due.append(req)
                version = self._registry.get(req.model_version_id)
                if version and version.status == ModelStatus.CHAMPION:
                    logger.warning("Certification expired for champion %s v%s",
                                   version.model_name, version.version)
            elif req.certified_until and req.certified_until - now < timedelta(days=self._config.recertification_reminder_days):
                due.append(req)
        return due

    def get_request(self, request_id: UUID) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    def list_requests(
        self,
        status: ApprovalStatus | None = None,
        action_class: ModelActionClass | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        results = list(self._requests.values())
        if status:
            results = [r for r in results if r.status == status]
        if action_class:
            results = [r for r in results if r.action_class == action_class]
        results.sort(key=lambda r: r.requested_at, reverse=True)
        return results[:limit]

    def clear(self) -> None:
        self._requests.clear()
