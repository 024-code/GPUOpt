from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelActionClass(StrEnum):
    ANOMALY_DETECTION = "anomaly_detection"
    DEMAND_FORECAST = "demand_forecast"
    RECOMMENDATION_SCORING = "recommendation_scoring"
    PLACEMENT_OPTIMIZATION = "placement_optimization"
    COST_ESTIMATION = "cost_estimation"
    DRIFT_DETECTION = "drift_detection"
    POWER_OPTIMIZATION = "power_optimization"


class ModelStatus(StrEnum):
    DRAFT = "draft"
    CHALLENGER = "challenger"
    CHAMPION = "champion"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DriftType(StrEnum):
    WORKLOAD_MIX = "workload_mix"
    FEATURE_DISTRIBUTION = "feature_distribution"
    PREDICTION_ERROR = "prediction_error"
    ACTION_OUTCOME_DIVERGENCE = "action_outcome_divergence"
    DATA_STALENESS = "data_staleness"


# ── Model Version Metadata ────────────────────────────────────

class ModelVersion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model_name: str
    version: str
    action_class: ModelActionClass
    status: ModelStatus = ModelStatus.DRAFT
    owner: str = ""
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    training_data_window_start: datetime | None = None
    training_data_window_end: datetime | None = None
    training_row_count: int = 0

    features: list[str] = []
    feature_count: int = 0
    target_metric: str = ""
    training_metrics: dict[str, float] = {}
    validation_metrics: dict[str, float] = {}

    algorithm: str = ""
    hyperparameters: dict[str, Any] = {}
    model_path: str = ""
    model_size_bytes: int = 0

    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_note: str = ""
    certification_days: int = 90
    certified_until: datetime | None = None

    source_git_commit: str = ""
    training_script: str = ""
    dependencies: dict[str, str] = {}

    shadow_evaluation_id: str | None = None
    parent_version_id: UUID | None = None
    tags: dict[str, str] = {}


class ModelMetadata(BaseModel):
    model_name: str
    action_class: ModelActionClass
    current_champion: ModelVersion | None = None
    current_challenger: ModelVersion | None = None
    version_count: int = 0
    last_trained_at: datetime | None = None
    total_predictions: int = 0
    avg_confidence: float = 0.0
    fallback_active: bool = False
    drift_detected: bool = False
    needs_recertification: bool = False


# ── Champion / Challenger ─────────────────────────────────────

class ShadowEvaluation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    champion_id: UUID
    challenger_id: UUID
    action_class: ModelActionClass
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    sample_count: int = 0
    champion_wins: int = 0
    challenger_wins: int = 0
    ties: int = 0
    metric_comparison: dict[str, dict[str, float]] = {}
    promoted: bool = False
    rejected: bool = False
    summary: str = ""


class ChampionChallengerConfig(BaseModel):
    shadow_sample_minimum: int = 1000
    shadow_duration_hours: int = 168  # 7 days
    win_threshold: float = 0.55
    metric_names: list[str] = ["accuracy", "mae", "latency_ms", "confidence"]
    auto_promote: bool = False
    notify_on_completion: bool = True


# ── Drift ─────────────────────────────────────────────────────

class DriftReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model_version_id: UUID
    drift_type: DriftType
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    drift_score: float = 0.0   # 0-1
    drifted_features: list[str] = []
    feature_deltas: dict[str, float] = {}
    action_class: ModelActionClass
    severity: str = "low"
    description: str = ""
    recommended_action: str = ""
    acknowledged: bool = False
    acknowledged_by: str | None = None


# ── Fallback ──────────────────────────────────────────────────

class FallbackConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.3
    max_data_stale_hours: int = 48
    drift_score_threshold: float = 0.7
    fallback_strategy: str = "deterministic_heuristic"
    recovery_check_interval_minutes: int = 60
    max_fallback_duration_hours: int = 72


# ── Approval ──────────────────────────────────────────────────

class ApprovalRequest(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model_version_id: UUID
    action_class: ModelActionClass
    requested_by: str = ""
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_notes: str = ""
    certification_period_days: int = 90
    certified_until: datetime | None = None
    requires_recertification: bool = True
    impact_level: str = "medium"


# ── Governance Config ─────────────────────────────────────────

class GovernanceConfig(BaseModel):
    champion_challenger: ChampionChallengerConfig = Field(default_factory=ChampionChallengerConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    high_impact_action_classes: list[ModelActionClass] = [
        ModelActionClass.RECOMMENDATION_SCORING,
        ModelActionClass.PLACEMENT_OPTIMIZATION,
        ModelActionClass.POWER_OPTIMIZATION,
    ]
    recertification_reminder_days: int = 14
    max_drift_before_fallback: int = 3
    audit_log_enabled: bool = True
