# Releasing AUA Framework

One-time setup, then every release is a single command.

## One-time: connect PyPI trusted publishing (no token ever stored)

Trusted publishing lets GitHub Actions publish to PyPI using a short-lived OIDC
JWT — no API tokens in secrets, no rotation, no revocation risk.

### Step 1 — PyPI side
1. Log in at **pypi.org**
2. Go to your account → **Publishing** → **Add a new pending publisher**
3. Fill in:
   | Field | Value |
   |---|---|
   | PyPI project name | `adaptive-utility-agent` |
   | Owner | `praneethtota` |
   | Repository name | `Adaptive-Utility-Agent` |
   | Workflow file | `release.yml` |
   | Environment | `pypi` |
4. Click **Add**

That's the only PyPI step. No token is created or stored.

### Step 2 — GitHub side
1. Go to **github.com/praneethtota/Adaptive-Utility-Agent** → Settings → Environments
2. Click **New environment** → name it exactly `pypi`
3. (Optional but recommended) Add a protection rule: **Required reviewers** → your
   username. This means the publish step pauses for your approval before going live.
4. Click **Save protection rules**

That's it. The workflow's `environment: pypi` block tells GitHub to issue an OIDC
token scoped to that environment, and PyPI verifies it came from this exact repo +
workflow + environment.

---

## Cutting a release

### The checklist (automated by `make-release`)
1. All tests green on `main`: `python -m pytest tests/ -q`
2. `aua/version.py` matches the tag you're about to push
3. `CHANGELOG.md` has a `## [X.Y.Z]` entry for this version
4. `git status` is clean (nothing uncommitted)

### One-liner
```bash
./make-release v1.2.0
```

That script does the checklist, bumps the version, adds the CHANGELOG skeleton,
commits, tags, and pushes — then GitHub Actions takes over:

```
Tag push
  └── release.yml
        ├── full-ci (3 × Python matrix, fail-fast)
        │     ruff → black → isort → mypy → pytest
        └── release (only runs after full-ci passes)
              ├── Validate: tag == aua/version.py
              ├── Build: wheel + sdist + twine check
              ├── Publish: PyPI (OIDC, no token)
              ├── Create: GitHub Release with CHANGELOG notes
              └── Upload: .whl + .tar.gz as release assets
```

The entire pipeline takes ~3 minutes. If any step fails, nothing is published.

### Manual (if you prefer)
```bash
# 1. Edit aua/version.py
# 2. Add ## [X.Y.Z] entry to CHANGELOG.md
# 3. Commit both
git add aua/version.py CHANGELOG.md
git commit -m "chore: release vX.Y.Z"
git push origin main

# 4. Tag and push — this is the trigger
git tag vX.Y.Z
git push origin vX.Y.Z
```

---

## Pre-releases

Tags containing `-rc`, `-beta`, or `-alpha` are marked as pre-releases on GitHub
and are **not** set as the latest release. PyPI will upload them but `pip install
adaptive-utility-agent` (no version pin) won't install them.

```bash
git tag v1.2.0-rc1 && git push origin v1.2.0-rc1
```

---

## Rollback / yanking

If a bad release ships, yank it on PyPI (users who already installed it keep it;
new installs skip it):
```bash
pip install twine
twine yank adaptive-utility-agent --version X.Y.Z
```

Then cut a patch: bump `aua/version.py` to `X.Y.(Z+1)`, fix the issue, run
`./make-release vX.Y.(Z+1)`.

---

## What the workflow does not do

- It does **not** create draft releases before publishing — the CHANGELOG entry is
  the review step. If the notes aren't ready, don't push the tag.
- It does **not** publish to Test PyPI first. Add a `test-pypi` environment +
  job if you want that.
