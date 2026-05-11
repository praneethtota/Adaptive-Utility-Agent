# AUA Framework — Deprecations

Deprecated features emit a `DeprecationWarning` for at least one minor version before removal.

---

## Active deprecations (v1.0)

None. v1.0 is the first stable release — no deprecations yet.

---

## Deprecation policy

1. Feature is marked deprecated with a warning in the code and in this file.
2. Warning persists for ≥1 minor version (e.g., deprecated in v1.1 → removed in v1.3+).
3. Migration path is documented here before removal.
4. Plugin protocol method signatures will not be deprecated in v1.x.

---

## Removed in v1.0 (from pre-release)

- Internal `ArbiterAgent` class: use `Arbiter` alias from public API.
- Flat JSONL state files as default: SQLite is now default. Use `backend: files` to opt back in.
