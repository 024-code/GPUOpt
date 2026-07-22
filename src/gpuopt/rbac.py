from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException
from starlette.requests import Request

logger = logging.getLogger(__name__)


class RoleType(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    FINOPS = "finops"
    DEVELOPER = "developer"


class Permission(StrEnum):
    CLUSTER_CREATE = "cluster:create"
    CLUSTER_READ = "cluster:read"
    CLUSTER_UPDATE = "cluster:update"
    CLUSTER_DELETE = "cluster:delete"
    STATE_READ = "state:read"
    STATE_COLLECT = "state:collect"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_RUN = "analysis:run"
    RECOMMENDATION_READ = "recommendation:read"
    RECOMMENDATION_GENERATE = "recommendation:generate"
    ACTUATE_DRY_RUN = "actuate:dry_run"
    ACTUATE_LIVE = "actuate:live"
    ACTUATE_ROLLBACK = "actuate:rollback"
    ALERT_READ = "alert:read"
    ALERT_MANAGE = "alert:manage"
    POLICY_READ = "policy:read"
    POLICY_MANAGE = "policy:manage"
    APPROVAL_MANAGE = "approval:manage"
    COST_READ = "cost:read"
    COST_MANAGE = "cost:manage"
    POWER_READ = "power:read"
    POWER_MANAGE = "power:manage"
    TENANT_READ = "tenant:read"
    TENANT_MANAGE = "tenant:manage"
    COMPLIANCE_READ = "compliance:read"
    DASHBOARD_READ = "dashboard:read"
    REPORT_MANAGE = "report:manage"
    REMEDIATION_MANAGE = "remediation:manage"
    RBAC_MANAGE = "rbac:manage"
    STREAM_READ = "stream:read"


@dataclass
class Role:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    role_type: RoleType = RoleType.VIEWER
    permissions: set[Permission] = field(default_factory=set)
    description: str = ""
    is_system: bool = False


@dataclass
class User:
    id: str = field(default_factory=lambda: str(uuid4()))
    username: str = ""
    email: str = ""
    role_ids: list[str] = field(default_factory=list)
    api_key_hash: str = ""
    api_key_prefix: str = ""
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_ROLE_DEFINITIONS: dict[RoleType, set[Permission]] = {
    RoleType.ADMIN: {
        Permission.CLUSTER_CREATE, Permission.CLUSTER_READ, Permission.CLUSTER_UPDATE, Permission.CLUSTER_DELETE,
        Permission.STATE_READ, Permission.STATE_COLLECT,
        Permission.ANALYSIS_READ, Permission.ANALYSIS_RUN,
        Permission.RECOMMENDATION_READ, Permission.RECOMMENDATION_GENERATE,
        Permission.ACTUATE_DRY_RUN, Permission.ACTUATE_LIVE, Permission.ACTUATE_ROLLBACK,
        Permission.ALERT_READ, Permission.ALERT_MANAGE,
        Permission.POLICY_READ, Permission.POLICY_MANAGE,
        Permission.APPROVAL_MANAGE,
        Permission.COST_READ, Permission.COST_MANAGE,
        Permission.POWER_READ, Permission.POWER_MANAGE,
        Permission.TENANT_READ, Permission.TENANT_MANAGE,
        Permission.COMPLIANCE_READ,
        Permission.DASHBOARD_READ,
        Permission.REPORT_MANAGE,
        Permission.REMEDIATION_MANAGE,
        Permission.RBAC_MANAGE,
        Permission.STREAM_READ,
    },
    RoleType.OPERATOR: {
        Permission.CLUSTER_READ, Permission.CLUSTER_UPDATE,
        Permission.STATE_READ, Permission.STATE_COLLECT,
        Permission.ANALYSIS_READ, Permission.ANALYSIS_RUN,
        Permission.RECOMMENDATION_READ, Permission.RECOMMENDATION_GENERATE,
        Permission.ACTUATE_DRY_RUN, Permission.ACTUATE_LIVE,
        Permission.ALERT_READ, Permission.ALERT_MANAGE,
        Permission.POLICY_READ, Permission.POLICY_MANAGE,
        Permission.APPROVAL_MANAGE,
        Permission.COST_READ,
        Permission.POWER_READ,
        Permission.DASHBOARD_READ,
        Permission.REPORT_MANAGE,
        Permission.REMEDIATION_MANAGE,
        Permission.STREAM_READ,
    },
    RoleType.VIEWER: {
        Permission.CLUSTER_READ,
        Permission.STATE_READ,
        Permission.ANALYSIS_READ,
        Permission.RECOMMENDATION_READ,
        Permission.ALERT_READ,
        Permission.POLICY_READ,
        Permission.COST_READ,
        Permission.POWER_READ,
        Permission.COMPLIANCE_READ,
        Permission.DASHBOARD_READ,
        Permission.STREAM_READ,
    },
    RoleType.FINOPS: {
        Permission.CLUSTER_READ,
        Permission.STATE_READ,
        Permission.ANALYSIS_READ,
        Permission.RECOMMENDATION_READ,
        Permission.ALERT_READ,
        Permission.COST_READ, Permission.COST_MANAGE,
        Permission.POWER_READ,
        Permission.COMPLIANCE_READ,
        Permission.DASHBOARD_READ,
        Permission.REPORT_MANAGE,
        Permission.STREAM_READ,
    },
    RoleType.DEVELOPER: {
        Permission.CLUSTER_READ,
        Permission.STATE_READ, Permission.STATE_COLLECT,
        Permission.ANALYSIS_READ, Permission.ANALYSIS_RUN,
        Permission.RECOMMENDATION_READ, Permission.RECOMMENDATION_GENERATE,
        Permission.ACTUATE_DRY_RUN,
        Permission.ALERT_READ,
        Permission.POLICY_READ,
        Permission.COST_READ,
        Permission.POWER_READ,
        Permission.DASHBOARD_READ,
        Permission.STREAM_READ,
    },
}


class RBACManager:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._roles: dict[str, Role] = {}
        self._api_key_to_user: dict[str, str] = {}
        self._seed_system_roles()
        self._seed_default_admin()

    def _seed_system_roles(self) -> None:
        for role_type, permissions in _ROLE_DEFINITIONS.items():
            role = Role(
                name=role_type.value,
                role_type=role_type,
                permissions=permissions,
                description=f"System {role_type.value} role",
                is_system=True,
            )
            self._roles[role.id] = role

    def _seed_default_admin(self) -> None:
        admin_role = next((r for r in self._roles.values() if r.role_type == RoleType.ADMIN), None)
        if admin_role:
            api_key = self._generate_api_key()
            user = User(
                username="admin",
                email="admin@gpuopt.local",
                role_ids=[admin_role.id],
                api_key_hash=self._hash_api_key(api_key),
                api_key_prefix=api_key[:8],
            )
            self._users[user.id] = user
            self._api_key_to_user[user.api_key_prefix] = user.id
            logger.info("Default admin user created with API key prefix %s", user.api_key_prefix)

    def _generate_api_key(self) -> str:
        return "gpuopt_" + secrets.token_hex(32)

    def _hash_api_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def create_user(self, username: str, email: str, role_type: RoleType = RoleType.VIEWER) -> tuple[User, str]:
        role = next((r for r in self._roles.values() if r.role_type == role_type), None)
        if role is None:
            raise ValueError(f"Unknown role type: {role_type}")
        api_key = self._generate_api_key()
        user = User(
            username=username,
            email=email,
            role_ids=[role.id],
            api_key_hash=self._hash_api_key(api_key),
            api_key_prefix=api_key[:8],
        )
        self._users[user.id] = user
        self._api_key_to_user[user.api_key_prefix] = user.id
        logger.info("Created user %s with role %s", username, role_type.value)
        return user, api_key

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def get_user_by_api_key(self, api_key: str) -> User | None:
        prefix = api_key[:8]
        user_id = self._api_key_to_user.get(prefix)
        if user_id is None:
            return None
        user = self._users.get(user_id)
        if user and user.api_key_hash == self._hash_api_key(api_key) and user.enabled:
            return user
        return None

    def list_users(self) -> list[User]:
        return sorted(self._users.values(), key=lambda u: u.username)

    def update_user(self, user_id: str, updates: dict) -> User | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        for key, value in updates.items():
            if hasattr(user, key) and key not in ("id", "api_key_hash", "created_at"):
                setattr(user, key, value)
        return user

    def delete_user(self, user_id: str) -> bool:
        user = self._users.pop(user_id, None)
        if user:
            self._api_key_to_user.pop(user.api_key_prefix, None)
            return True
        return False

    def rotate_api_key(self, user_id: str) -> str | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        old_prefix = user.api_key_prefix
        new_key = self._generate_api_key()
        user.api_key_hash = self._hash_api_key(new_key)
        user.api_key_prefix = new_key[:8]
        self._api_key_to_user.pop(old_prefix, None)
        self._api_key_to_user[user.api_key_prefix] = user.id
        return new_key

    def get_role(self, role_id: str) -> Role | None:
        return self._roles.get(role_id)

    def list_roles(self) -> list[Role]:
        return sorted(self._roles.values(), key=lambda r: r.name)

    def create_role(self, name: str, permissions: list[Permission], description: str = "") -> Role:
        role = Role(name=name, permissions=set(permissions), description=description)
        self._roles[role.id] = role
        return role

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        user = self._users.get(user_id)
        if user is None:
            return set()
        permissions: set[Permission] = set()
        for rid in user.role_ids:
            role = self._roles.get(rid)
            if role:
                permissions |= role.permissions
        return permissions

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        return permission in self.get_user_permissions(user_id)

    def check_api_key_permission(self, api_key: str, permission: Permission) -> bool:
        user = self.get_user_by_api_key(api_key)
        if user is None:
            return False
        return permission in self.get_user_permissions(user.id)

    def authenticate(self, api_key: str) -> User | None:
        return self.get_user_by_api_key(api_key)

    def reset(self) -> None:
        self._users.clear()
        self._roles.clear()
        self._api_key_to_user.clear()
        self._seed_system_roles()
        self._seed_default_admin()
        logger.info("RBACManager reset")


class PermissionChecker:
    def __init__(self, permission: Permission) -> None:
        self._permission = permission

    def __call__(self, request: Request) -> None:
        from .config import get_settings
        from .dependencies import get_rbac_manager

        settings = get_settings()
        rbac = get_rbac_manager()
        user_id: str = getattr(request.state, "user_id", "")
        if settings.api_keyless_mode and (not user_id or user_id == "system"):
            return
        if not user_id or user_id == "system":
            raise HTTPException(status_code=401, detail="Authentication required")
        if not rbac.check_permission(user_id, self._permission):
            raise HTTPException(status_code=403, detail=f"Missing permission: {self._permission.value}")


_require_permission_cache: dict[str, PermissionChecker] = {}


def require_permission(permission: Permission):
    key = permission.value
    if key not in _require_permission_cache:
        _require_permission_cache[key] = PermissionChecker(permission)
    return _require_permission_cache[key]
