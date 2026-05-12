"""Tests for aua/policy.py — policy system."""

import pytest
import yaml

from aua.policy import Policy, load_policy, validate_policy_yaml


@pytest.fixture
def simple_policy_yaml(tmp_path):
    content = {
        "name": "TestPolicy",
        "version": "1.0",
        "max_retries": 2,
        "max_total_bonus": 0.30,
        "assertions": [],
    }
    p = tmp_path / "test_policy.yaml"
    p.write_text(yaml.dump(content))
    return str(p)


@pytest.fixture
def policy_with_overrides_yaml(tmp_path):
    content = {
        "name": "WeightPolicy",
        "version": "1.0",
        "utility_overrides": {"w_k": 0.35},
        "assertions": [],
    }
    p = tmp_path / "weight_policy.yaml"
    p.write_text(yaml.dump(content))
    return str(p)


# ── Policy construction ───────────────────────────────────────────────────────


def test_policy_defaults():
    p = Policy(name="Defaults")
    assert p.version == "1.0"
    assert p.max_retries == 3
    assert p.max_total_bonus == 0.30
    assert p.utility_overrides == {}
    assert p.assertions == []


def test_policy_add_wrong_type_raises():
    p = Policy(name="WrongType")
    with pytest.raises(TypeError, match="@assertion"):
        p.add(lambda output, ctx: (True, None))  # type: ignore


# ── YAML loading ──────────────────────────────────────────────────────────────


def test_load_policy_basic(simple_policy_yaml):
    pol = load_policy(simple_policy_yaml)
    assert pol.name == "TestPolicy"
    assert pol.version == "1.0"
    assert pol.max_retries == 2
    assert pol.max_total_bonus == 0.30


def test_load_policy_weight_overrides(policy_with_overrides_yaml):
    pol = load_policy(policy_with_overrides_yaml)
    assert pol.utility_overrides == {"w_k": 0.35}


def test_load_policy_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_policy("/nonexistent/path/policy.yaml")


def test_load_policy_missing_name_raises(tmp_path):
    p = tmp_path / "no_name.yaml"
    p.write_text(yaml.dump({"version": "1.0"}))
    with pytest.raises(ValueError, match="name"):
        load_policy(str(p))


def test_load_policy_bad_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(":: invalid: yaml: {{{")
    with pytest.raises(Exception):
        load_policy(str(p))


# ── Validation ────────────────────────────────────────────────────────────────


def test_validate_policy_yaml_valid(simple_policy_yaml):
    errors = validate_policy_yaml(simple_policy_yaml)
    assert errors == []


def test_validate_policy_yaml_missing_name(tmp_path):
    p = tmp_path / "no_name.yaml"
    p.write_text(yaml.dump({}))
    errors = validate_policy_yaml(str(p))
    assert any("name" in e for e in errors)


def test_validate_policy_yaml_invalid_level(tmp_path):
    content = {"name": "Test", "assertions": [{"import_path": "x:y", "level": "invalid_level"}]}
    p = tmp_path / "bad_level.yaml"
    p.write_text(yaml.dump(content))
    errors = validate_policy_yaml(str(p))
    assert any("level" in e for e in errors)


def test_validate_policy_yaml_bonus_out_of_range(tmp_path):
    content = {"name": "Test", "assertions": [{"import_path": "x:y", "bonus": 5.0}]}
    p = tmp_path / "bad_bonus.yaml"
    p.write_text(yaml.dump(content))
    errors = validate_policy_yaml(str(p))
    assert any("bonus" in e for e in errors)


def test_validate_policy_yaml_unknown_weight_key(tmp_path):
    content = {"name": "Test", "utility_overrides": {"w_unknown": 0.5}}
    p = tmp_path / "bad_key.yaml"
    p.write_text(yaml.dump(content))
    errors = validate_policy_yaml(str(p))
    assert any("unknown key" in e.lower() or "w_unknown" in e for e in errors)


def test_validate_policy_yaml_not_found():
    errors = validate_policy_yaml("/no/such/file.yaml")
    assert len(errors) == 1
    assert "not found" in errors[0].lower()


def test_validate_policy_yaml_missing_import_path(tmp_path):
    content = {"name": "Test", "assertions": [{"level": "soft"}]}  # no import_path
    p = tmp_path / "no_path.yaml"
    p.write_text(yaml.dump(content))
    errors = validate_policy_yaml(str(p))
    assert any("import_path" in e for e in errors)


# ── Integration: Policy.run with utility overrides ────────────────────────────


def test_policy_utility_overrides_accessible():
    pol = Policy(name="WithOverrides", utility_overrides={"w_k": 0.40})
    assert pol.utility_overrides["w_k"] == 0.40


def test_policy_summary_includes_all_fields():
    pol = Policy(name="Full", version="2.0", max_total_bonus=0.25)
    s = pol.summary()
    assert "name" in s
    assert "version" in s
    assert "max_total_bonus" in s
    assert "assertions" in s
    assert "utility_overrides" in s
