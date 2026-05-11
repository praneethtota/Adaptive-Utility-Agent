"""
aua/plugins/errors.py — Stable AUA_* error code taxonomy.

Every error AUA can emit has:
  - A stable string code (AUA_*)
  - An HTTP status code
  - A CLI exit code
  - A human-readable description

These codes are stable from v0.8. New codes may be added; existing codes
will not be removed or renamed in v1.x.

REST error format:
    {
        "error": "AUA_BACKEND_UNREACHABLE",
        "message": "Specialist 'swe' did not respond within 60s",
        "status_code": 503,
        "request_id": "abc-123",
        "details": {}
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AUAErrorCode:
    """A stable AUA error code with HTTP and CLI mappings."""

    code: str
    http_status: int
    cli_exit_code: int
    description: str
    details_schema: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.code


# ── Config errors ─────────────────────────────────────────────────────────────

AUA_CONFIG_INVALID = AUAErrorCode(
    code="AUA_CONFIG_INVALID",
    http_status=500,
    cli_exit_code=1,
    description="aua_config.yaml failed schema validation.",
)

AUA_CONFIG_VERSION_UNSUPPORTED = AUAErrorCode(
    code="AUA_CONFIG_VERSION_UNSUPPORTED",
    http_status=500,
    cli_exit_code=1,
    description="Config version is not supported. Run: aua config migrate",
)

AUA_CONFIG_SECRET_MISSING = AUAErrorCode(
    code="AUA_CONFIG_SECRET_MISSING",
    http_status=500,
    cli_exit_code=1,
    description="A referenced secret could not be resolved from the configured provider.",
)

# ── Backend / specialist errors ───────────────────────────────────────────────

AUA_BACKEND_UNREACHABLE = AUAErrorCode(
    code="AUA_BACKEND_UNREACHABLE",
    http_status=503,
    cli_exit_code=1,
    description="A specialist or arbiter server could not be reached (connection refused).",
)

AUA_SPECIALIST_TIMEOUT = AUAErrorCode(
    code="AUA_SPECIALIST_TIMEOUT",
    http_status=504,
    cli_exit_code=1,
    description="A specialist server did not respond within the configured timeout.",
)

AUA_ARBITER_TIMEOUT = AUAErrorCode(
    code="AUA_ARBITER_TIMEOUT",
    http_status=504,
    cli_exit_code=1,
    description="The arbiter server did not respond within the configured timeout.",
)

AUA_BACKEND_ERROR = AUAErrorCode(
    code="AUA_BACKEND_ERROR",
    http_status=502,
    cli_exit_code=1,
    description="A specialist server returned an unexpected error response.",
)

# ── Plugin / extension errors ─────────────────────────────────────────────────

AUA_PLUGIN_LOAD_FAILED = AUAErrorCode(
    code="AUA_PLUGIN_LOAD_FAILED",
    http_status=500,
    cli_exit_code=1,
    description="A plugin could not be imported or instantiated.",
)

AUA_PLUGIN_CONTRACT_INVALID = AUAErrorCode(
    code="AUA_PLUGIN_CONTRACT_INVALID",
    http_status=500,
    cli_exit_code=1,
    description="A plugin does not implement the required Protocol interface.",
)

AUA_HOOK_FAILED = AUAErrorCode(
    code="AUA_HOOK_FAILED",
    http_status=500,
    cli_exit_code=1,
    description="A lifecycle hook raised an exception (fail-closed hook).",
)

AUA_MIDDLEWARE_FAILED = AUAErrorCode(
    code="AUA_MIDDLEWARE_FAILED",
    http_status=500,
    cli_exit_code=1,
    description="A middleware component raised an exception.",
)

# ── Auth / security errors ────────────────────────────────────────────────────

AUA_AUTH_REQUIRED = AUAErrorCode(
    code="AUA_AUTH_REQUIRED",
    http_status=401,
    cli_exit_code=1,
    description="This endpoint requires a valid bearer token.",
)

AUA_FORBIDDEN = AUAErrorCode(
    code="AUA_FORBIDDEN",
    http_status=403,
    cli_exit_code=1,
    description="The provided token does not have the required scope for this endpoint.",
)

AUA_RATE_LIMITED = AUAErrorCode(
    code="AUA_RATE_LIMITED",
    http_status=429,
    cli_exit_code=1,
    description="Request rate limit exceeded. Retry after the indicated delay.",
)

AUA_TOKEN_REVOKED = AUAErrorCode(
    code="AUA_TOKEN_REVOKED",
    http_status=401,
    cli_exit_code=1,
    description="The provided token has been revoked.",
)

# ── Deployment / state errors ─────────────────────────────────────────────────

AUA_PROMOTION_REJECTED = AUAErrorCode(
    code="AUA_PROMOTION_REJECTED",
    http_status=409,
    cli_exit_code=1,
    description="The GREEN model did not meet the promotion threshold.",
)

AUA_ROLLBACK_FAILED = AUAErrorCode(
    code="AUA_ROLLBACK_FAILED",
    http_status=500,
    cli_exit_code=1,
    description="Rollback failed — no prior promotion record found.",
)

AUA_STATE_STORE_UNAVAILABLE = AUAErrorCode(
    code="AUA_STATE_STORE_UNAVAILABLE",
    http_status=503,
    cli_exit_code=1,
    description="The state store (SQLite/Postgres) is unavailable.",
)

AUA_MIGRATION_REQUIRED = AUAErrorCode(
    code="AUA_MIGRATION_REQUIRED",
    http_status=500,
    cli_exit_code=2,
    description="State store schema is outdated. Run: aua config migrate",
)

# ── Registry ──────────────────────────────────────────────────────────────────

ALL_ERROR_CODES: dict[str, AUAErrorCode] = {
    obj.code: obj for name, obj in list(globals().items()) if isinstance(obj, AUAErrorCode)
}


def get_error_code(code: str) -> AUAErrorCode | None:
    """Look up an error code by its string identifier."""
    return ALL_ERROR_CODES.get(code)
