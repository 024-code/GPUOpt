from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class CheckSeverity(StrEnum):
    FAIL = "fail"
    FAIL_OR_WARNING = "fail_or_warning"
    WARNING = "warning"
    WARNING_THEN_FAIL_BEFORE_R02 = "warning_then_fail_before_r0.2"
    WARNING_THEN_FAIL_BEFORE_STAGING = "warning_then_fail_before_staging"


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "na"
    NOT_CHECKED = "not_checked"


class EnvironmentType(StrEnum):
    SANDBOX = "sandbox"
    STAGING = "staging"
    PRODUCTION = "production"


class MandatoryCheck(BaseModel):
    check_name: str
    acceptance_criterion: str
    severity: CheckSeverity
    environment: EnvironmentType
    expected_to_pass: bool = False
    rationale: str = ""


class CheckResult(BaseModel):
    check_name: str
    status: CheckStatus
    severity: CheckSeverity
    detail: str = ""
    passed: bool = False


class EnvironmentCheckRun(BaseModel):
    environment: EnvironmentType
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    checks: list[CheckResult] = Field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    overall_pass: bool = False
    summary: str = ""


class EnvironmentCheckCatalog(BaseModel):
    environment: EnvironmentType
    checks: list[MandatoryCheck] = Field(default_factory=list)
