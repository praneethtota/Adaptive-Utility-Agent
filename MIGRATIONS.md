# AUA Framework — Migration Guide

---

## v0.9 → v1.0

### Config version field
Update `aua_config.yaml`:
```yaml
aua:
  version: "1.0"   # was "0.9" or "0.8"
```

### State store migration
If upgrading from flat files (v0.7):
```bash
aua config migrate --from 0.7 --to 1.0
```

### No breaking API changes
All public Python APIs, plugin interfaces, and REST endpoints from v0.9 are unchanged in v1.0.

---

## v0.8 → v0.9

### New required env vars (if auth enabled)
```bash
export AUA_TOKEN_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
```

### State store
SQLite is now the default. If you have flat JSONL files:
```bash
aua config migrate --from 0.8 --to 0.9
```

---

## v0.7 → v0.8

### Plugin system
Plugins previously copied into `aua/` source must be moved to `plugins/` directory and registered via `import_path` in YAML. No AUA source edits required from v0.8 onwards.

### State files
`.aua/state/promotions.jsonl` and `dpo_pairs/*.jsonl` are now managed by the SQLite state store. Migrate:
```bash
aua config migrate --from 0.7 --to 0.8
```

---

## v0.5/v0.6 → v0.7

### Tier names
Old name → New canonical name:
- `rtx4090` → `single-4090` (alias still works)
- `a100` → `a100-cluster` (alias still works)

### Config backend field
Add `backend: ollama` or `backend: vllm` explicitly if not set.
