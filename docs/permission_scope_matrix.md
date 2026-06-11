# AUA Framework — Permission / Scope Matrix

**Version:** 1.1.0  
**Status:** Canonical. Authentication implemented in v0.9-rc1.

---

## Scopes

| Scope | Description |
|---|---|
| `aua:query` | Send queries via `POST /query` |
| `aua:stream` | Send streaming queries via `POST /query/stream` |
| `aua:batch` | Send batch queries via `POST /query/batch` |
| `aua:status` | Read `GET /status`, `GET /health/*`, `GET /version` |
| `aua:config:read` | Read `GET /config` (secrets redacted) |
| `aua:config:write` | Reload config via `POST /config/reload` |
| `aua:corrections:read` | Read `GET /corrections` |
| `aua:corrections:write` | Inject corrections via `POST /corrections` |
| `aua:deploy` | Trigger green evaluation via `POST /deploy/green` |
| `aua:rollback` | Execute rollback (CLI + REST) |
| `aua:extensions:read` | Read `GET /extensions`, `GET /extensions/{name}` |
| `aua:extensions:write` | Load/reload extensions, test imports |
| `aua:tokens:read` | List and inspect tokens (CLI: `aua token list`) |
| `aua:tokens:write` | Create and revoke tokens (CLI: `aua token create/revoke`) |
| `aua:admin` | All scopes — for operator/admin use only |

---

## Endpoint → Required Scope

| Endpoint | Method | Required Scope | Notes |
|---|---|---|---|
| `/query` | POST | `aua:query` | |
| `/query/stream` | POST | `aua:stream` | |
| `/query/batch` | POST | `aua:batch` | |
| `/health/live` | GET | none | Public — used by load balancers |
| `/health/ready` | GET | none | Public |
| `/health/startup` | GET | none | Public |
| `/version` | GET | none | Public |
| `/docs` | GET | none | Disable in production via config |
| `/status` | GET | `aua:status` | |
| `/config` | GET | `aua:config:read` | Secrets always redacted |
| `/config/reload` | POST | `aua:config:write` | |
| `/corrections` | GET | `aua:corrections:read` | |
| `/corrections` | POST | `aua:corrections:write` | |
| `/deploy/green` | POST | `aua:deploy` | |
| `/deploy/rollback` | POST | `aua:rollback` | |
| `/extensions` | GET | `aua:extensions:read` | Disabled in production |
| `/extensions/{name}` | GET | `aua:extensions:read` | Disabled in production |
| `/extensions/reload` | POST | `aua:extensions:write` | Disabled in production |
| `/extensions/test` | POST | `aua:extensions:write` | Dev only |
| `/metrics` | GET | `aua:status` | Prometheus scrape endpoint |
| **v1.1 — persistence, search & production ops** | | | |
| `/conversations` | POST / GET | `aua:query` | |
| `/conversations/{id}/title` | PATCH | `aua:query` | |
| `/conversations/{id}/messages` | GET / POST | `aua:query` | |
| `/projects` | POST / GET | `aua:query` | |
| `/search` | GET | `aua:query` | |
| `/context/backup/coverage` | GET | `aua:status` | |
| `/context/backup/run-coverage-job` | POST | `aua:query` | |
| `/corrections/confirm-implicit` | POST | `aua:corrections:write` | |
| `/corrections/{id}` | PATCH / DELETE | `aua:corrections:write` | DELETE is a soft delete (scope='superseded') |
| `/corrections/evidence` | GET | `aua:corrections:read` | |
| `/analytics`, `/reliability`, `/usage`, `/pricing` | GET | `aua:status` | |
| `/version/check`, `/update/skipped` | GET | none | Public |
| `/update/skip` | POST | `aua:config:write` | |
| `/bug-report` | POST | none | Returns 200 even without a PAT configured |
| `/local/models`, `/local/settings` | GET | `aua:status` | |
| `/local/models`, `/local/settings` | POST | `aua:config:write` | |
| `/local/specialist/{id}` | PATCH | `aua:config:write` | |
| `/domain-tree` | GET | `aua:status` | |

---

## Default Token Scopes by Role

| Role | Scopes granted |
|---|---|
| `reader` | `aua:query aua:stream aua:status` |
| `operator` | All except `aua:admin aua:extensions:write` |
| `admin` | `aua:admin` (all scopes) |
| `ci-deploy` | `aua:deploy aua:rollback aua:config:read` |
| `monitoring` | `aua:status` |

---

## Auth Behaviour

- **v0.7:** No auth. All endpoints are open. Not suitable for public exposure.
- **v0.8:** Scope matrix defined. Auth implementation ships in v0.9-rc1.
- **v0.9:** Bearer token required on all non-public endpoints. Local dev can disable with `security: {auth_enabled: false}` and explicit warning.
- **v1.0:** mTLS between router and specialists. Audit log for every auth event.

---

## Local Development

```yaml
# aua_config.yaml — local dev only
security:
  auth_enabled: false   # NEVER set this in production
```

When `auth_enabled: false`, `aua doctor` prints a prominent WARNING. The warning cannot be suppressed.
