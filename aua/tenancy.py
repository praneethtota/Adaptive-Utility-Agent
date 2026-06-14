"""
aua/tenancy.py — Multi-tenancy support (#44).

Provides:
  TenantContext   — request-scoped tenant identity, carried via contextvars
  get_tenant_id() — read current tenant (None for anonymous / single-tenant)
  set_tenant_id() — set in middleware; auto-cleared after request

The tenant ID is injected into state writes (corrections, promotions,
audit_log, model_runs) and used as a query filter so each tenant sees
only its own data.

Wire-up summary:
  1. TenantPolicyMiddleware (aua/middleware.py) calls set_tenant_id() from
     the X-Tenant-ID request header and enforces per-tenant policy.
  2. Router reads get_tenant_id() when writing to the state store.
  3. SQLiteStateStore.query() accepts tenant_id kwarg for filtered reads.
  4. RateLimitMiddleware uses tenant_id as part of the rate-limit key so
     per-tenant quotas are enforced independently.

Configuration (aua_config.yaml):

    middleware:
      - import_path: aua.middleware:TenantPolicyMiddleware
        config:
          tenants:
            tenant-a:
              allowed_fields: [software_engineering, mathematics]
              rate_limit_rpm: 60          # requests per minute for this tenant
              model_binding: swe          # force all queries to this specialist
            tenant-b:
              allowed_fields: [law, software_engineering]
              rate_limit_rpm: 120
          reject_unknown: true   # 403 for unknown X-Tenant-ID (default: false)
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

# ── Context variable ──────────────────────────────────────────────────────────

_tenant_id_var: ContextVar[str | None] = ContextVar("aua_tenant_id", default=None)


def get_tenant_id() -> str | None:
    """Return the current request's tenant ID, or None (anonymous)."""
    return _tenant_id_var.get()


def set_tenant_id(tenant_id: str | None) -> Token:
    """
    Set the tenant ID for the current async task.
    Returns a token that can be used to reset to the previous value.
    """
    return _tenant_id_var.set(tenant_id)


def reset_tenant_id(token: Token) -> None:
    """Reset the tenant ID to the value before set_tenant_id() was called."""
    _tenant_id_var.reset(token)


# ── Tenant policy ─────────────────────────────────────────────────────────────


@dataclass
class TenantPolicy:
    """
    Per-tenant access and resource policy.
    Parsed from the `tenants:` block in TenantPolicyMiddleware config.
    """

    tenant_id: str
    allowed_fields: list[str] = field(default_factory=list)  # [] = all fields allowed
    rate_limit_rpm: int | None = None  # None = inherit global default
    model_binding: str | None = None  # None = normal routing; name = force this specialist


def parse_tenant_policies(
    tenants_raw: dict[str, dict[str, Any]],
) -> dict[str, TenantPolicy]:
    """
    Parse the `tenants:` dict from TenantPolicyMiddleware config into
    TenantPolicy objects keyed by tenant ID.
    """
    policies: dict[str, TenantPolicy] = {}
    for tid, cfg in (tenants_raw or {}).items():
        policies[tid] = TenantPolicy(
            tenant_id=tid,
            allowed_fields=list(cfg.get("allowed_fields", [])),
            rate_limit_rpm=cfg.get("rate_limit_rpm"),
            model_binding=cfg.get("model_binding"),
        )
    return policies
