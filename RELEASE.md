# AUA Framework — Release Checklist

Use this checklist for every release. Complete every item before tagging.

---

## Pre-release checklist

### 1. Tests & quality
- [ ] `pytest -q` — all tests pass, 0 failures
- [ ] `ruff check aua tests` — 0 errors
- [ ] `black --check aua tests` — 0 reformats needed
- [ ] `mypy aua --ignore-missing-imports` — 0 errors

### 2. Version bump
- [ ] `aua/version.py` updated to new version
- [ ] `apps/aua_chat/package.json` version updated
- [ ] Version assertions in `tests/test_imports.py` and `tests/test_router_api.py` updated
- [ ] `CHANGELOG.md` updated with release notes

### 3. Documentation
- [ ] Architecture spec reflects any new components
- [ ] Compatibility matrix updated (new deps, new tested hardware)
- [ ] Tutorial Part 1 tested from scratch on a fresh directory
- [ ] All example projects run (`examples/*/README.md` commands verified)

### 4. Docker
- [ ] `docker compose up` starts cleanly
- [ ] `docker compose --profile obs up` shows Grafana dashboard
- [ ] `docker compose --profile ollama up` pulls models and serves
- [ ] Dockerfile builds from scratch: `docker build --no-cache -t aua:test .`

### 5. Fresh-clone validation (D-04)
Run from a completely fresh directory with no prior AUA install:
```bash
pip install adaptive-utility-agent
python -c "from aua import Router, Arbiter, UtilityScorer, BlueGreenDeployment, CorrectionLoop; print('ok')"
aua init --tier macbook --force
aua doctor --strict
aua serve --dry-run
aua models list
aua fields list
aua presets list
aua defaults show
aua extensions test --kind middleware --import-path aua.middleware:PIIRedactionMiddleware
```

### 6. Security profile
- [ ] `aua token create --scope aua:query --expires 30d` works
- [ ] `aua certs generate` works (requires `pip install cryptography`)
- [ ] `curl http://localhost:8000/metrics` returns Prometheus text
- [ ] `curl http://localhost:8000/metrics/cost` returns JSON

### 7. Chat UI smoke test
- [ ] `npm install` in `apps/aua_chat/` succeeds
- [ ] `aua ui` starts on port 3001
- [ ] Login page appears at `http://localhost:3001/login`
- [ ] Login with `admin / aua-admin` succeeds
- [ ] New chat → send message → response appears
- [ ] Framework Debugger shows domain + U score
- [ ] AUA Controls drawer opens and shows config

### 8. Eval smoke test
```bash
aua serve &
aua eval run --dataset evals/coding_smoke.yaml
aua eval report
```
- [ ] Pass rate ≥ 60% on coding_smoke

### 9. Migration notes
- [ ] `MIGRATIONS.md` updated if any config schema changed
- [ ] `DEPRECATIONS.md` updated if any APIs deprecated
- [ ] `aua config check-version` passes on configs from previous release

---

## Release steps

1. Merge all PRs to `main`
2. Complete all checklist items above
3. Update version in `aua/version.py`
4. Update `CHANGELOG.md`
5. `git tag v{VERSION} && git push origin v{VERSION}`
6. `python -m build && twine upload dist/*` (PyPI)
7. `docker buildx build --platform linux/amd64,linux/arm64 -t praneethtota/aua:{VERSION} --push .`
8. Update GitHub Release notes from `CHANGELOG.md`
9. Post release announcement

---

## Versioning policy

- **SemVer**: `MAJOR.MINOR.PATCH`
- **v1.x**: Public API stable. Plugin Protocol interfaces will not break.
- **v2.0**: May break plugin contracts — migration guide required.
- **Deprecation**: Deprecated features emit a warning for ≥1 minor version before removal.
- **Pre-releases**: `alpha` → `beta` → `rc1` → `rc2` → stable
