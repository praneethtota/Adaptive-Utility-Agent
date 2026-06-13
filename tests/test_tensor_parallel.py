"""
tests/test_tensor_parallel.py — Tests for #66 tensor parallel specialist serving.

Coverage:
  SpecialistConfig.vllm_command(): TP/PP flags emitted correctly
  ArbiterConfig.vllm_command(): TP/PP flags emitted correctly
  _build_env(): single GPU, multi-GPU gpu_ids, Ollama (no CUDA var)
  YAML loading: tensor_parallel_size, pipeline_parallel_size, gpu_ids
  Validation: TP must be power of 2; gpu_ids length must match TP
  Default behaviour: TP=1 and PP=1 emit no extra flags (backward compat)
"""

from __future__ import annotations

import pytest

from aua.config import ArbiterConfig, SpecialistConfig, load_config
from aua.serve import _build_env

# ── Helper ────────────────────────────────────────────────────────────────────


def _make_spec(**kwargs) -> SpecialistConfig:
    defaults = dict(name="swe", model="Qwen/model", port=9001, field="software_engineering")
    return SpecialistConfig(**{**defaults, **kwargs})


def _make_arb(**kwargs) -> ArbiterConfig:
    defaults = dict(model="Qwen/arb", port=9003)
    return ArbiterConfig(**{**defaults, **kwargs})


# ── vllm_command — SpecialistConfig ──────────────────────────────────────────


class TestSpecialistVllmCommand:
    def test_no_tp_pp_flags_by_default(self) -> None:
        cmd = _make_spec().vllm_command()
        assert "--tensor-parallel-size" not in cmd
        assert "--pipeline-parallel-size" not in cmd

    def test_tp_flag_emitted_when_gt_1(self) -> None:
        cmd = _make_spec(tensor_parallel_size=4).vllm_command()
        assert "--tensor-parallel-size" in cmd
        idx = cmd.index("--tensor-parallel-size")
        assert cmd[idx + 1] == "4"

    def test_pp_flag_emitted_when_gt_1(self) -> None:
        cmd = _make_spec(pipeline_parallel_size=2).vllm_command()
        assert "--pipeline-parallel-size" in cmd
        idx = cmd.index("--pipeline-parallel-size")
        assert cmd[idx + 1] == "2"

    def test_tp_and_pp_both_present(self) -> None:
        cmd = _make_spec(tensor_parallel_size=4, pipeline_parallel_size=2).vllm_command()
        assert "--tensor-parallel-size" in cmd
        assert "--pipeline-parallel-size" in cmd
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
        assert cmd[cmd.index("--pipeline-parallel-size") + 1] == "2"

    def test_tp_1_emits_no_flag(self) -> None:
        cmd = _make_spec(tensor_parallel_size=1).vllm_command()
        assert "--tensor-parallel-size" not in cmd

    def test_standard_flags_still_present_with_tp(self) -> None:
        cmd = _make_spec(
            tensor_parallel_size=8, quantization=None, enforce_eager=False
        ).vllm_command()
        assert "--model" in cmd
        assert "--port" in cmd
        assert "--tensor-parallel-size" in cmd

    def test_enforce_eager_and_tp_compatible(self) -> None:
        cmd = _make_spec(tensor_parallel_size=4, enforce_eager=True).vllm_command()
        assert "--enforce-eager" in cmd
        assert "--tensor-parallel-size" in cmd


# ── vllm_command — ArbiterConfig ─────────────────────────────────────────────


class TestArbiterVllmCommand:
    def test_no_tp_flags_by_default(self) -> None:
        cmd = _make_arb().vllm_command()
        assert "--tensor-parallel-size" not in cmd
        assert "--pipeline-parallel-size" not in cmd

    def test_tp_flag_emitted(self) -> None:
        cmd = _make_arb(tensor_parallel_size=2).vllm_command()
        assert "--tensor-parallel-size" in cmd
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"

    def test_pp_flag_emitted(self) -> None:
        cmd = _make_arb(pipeline_parallel_size=4).vllm_command()
        assert "--pipeline-parallel-size" in cmd


# ── _build_env ────────────────────────────────────────────────────────────────


class TestBuildEnv:
    def test_single_gpu_sets_cuda_visible(self) -> None:
        env = _build_env(gpu_index=2, backend="vllm")
        assert env["CUDA_VISIBLE_DEVICES"] == "2"

    def test_multi_gpu_ids_joined(self) -> None:
        env = _build_env(gpu_index=0, backend="vllm", gpu_ids=[0, 1, 2, 3])
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    def test_gpu_ids_overrides_gpu_index(self) -> None:
        env = _build_env(gpu_index=5, backend="vllm", gpu_ids=[2, 3])
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_ollama_backend_no_cuda_var(self) -> None:
        env = _build_env(gpu_index=0, backend="ollama")
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_gpu_ids_none_falls_back_to_index(self) -> None:
        env = _build_env(gpu_index=3, backend="vllm", gpu_ids=None)
        assert env["CUDA_VISIBLE_DEVICES"] == "3"

    def test_eight_gpu_h100_layout(self) -> None:
        env = _build_env(gpu_index=0, backend="vllm", gpu_ids=list(range(8)))
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


# ── YAML loading ──────────────────────────────────────────────────────────────


class TestYamlLoading:
    def test_defaults_load_correctly(self, tmp_path) -> None:
        yaml_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: Qwen/model
    port: 9001
    field: software_engineering
arbiter:
  model: Qwen/arb
  port: 9003
router:
  port: 8000
"""
        cfg_file = tmp_path / "aua_config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file)
        spec = cfg.specialists[0]
        assert spec.tensor_parallel_size == 1
        assert spec.pipeline_parallel_size == 1
        assert spec.gpu_ids is None
        assert cfg.arbiter.tensor_parallel_size == 1

    def test_tp_pp_loaded_from_yaml(self, tmp_path) -> None:
        yaml_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: meta-llama/Llama-3-70B
    port: 9001
    field: software_engineering
    tensor_parallel_size: 4
    pipeline_parallel_size: 2
    gpu_ids: [0, 1, 2, 3]
    gpu_memory_utilization: 0.90
    quantization: null
    enforce_eager: false
arbiter:
  model: Qwen/arb
  port: 9003
  tensor_parallel_size: 2
  gpu_ids: [4, 5]
  gpu_memory_utilization: 0.20
router:
  port: 8000
"""
        cfg_file = tmp_path / "aua_config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file)
        spec = cfg.specialists[0]
        assert spec.tensor_parallel_size == 4
        assert spec.pipeline_parallel_size == 2
        assert spec.gpu_ids == [0, 1, 2, 3]

        arb = cfg.arbiter
        assert arb.tensor_parallel_size == 2
        assert arb.gpu_ids == [4, 5]

    def test_vllm_command_from_loaded_config(self, tmp_path) -> None:
        yaml_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: meta-llama/Llama-3-70B
    port: 9001
    field: software_engineering
    tensor_parallel_size: 8
    gpu_memory_utilization: 0.90
    quantization: null
    enforce_eager: false
arbiter:
  model: Qwen/arb
  port: 9003
  gpu_memory_utilization: 0.10
router:
  port: 8000
"""
        cfg_file = tmp_path / "aua_config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file)
        cmd = cfg.specialists[0].vllm_command()
        assert "--tensor-parallel-size" in cmd
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "8"


# ── Validation ────────────────────────────────────────────────────────────────


class TestValidation:
    def test_tp_not_power_of_2_raises(self, tmp_path) -> None:
        yaml_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: model
    port: 9001
    field: software_engineering
    tensor_parallel_size: 3
arbiter:
  model: arb
  port: 9003
router:
  port: 8000
"""
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="power of 2"):
            load_config(cfg_file)

    def test_gpu_ids_length_mismatch_raises(self, tmp_path) -> None:
        yaml_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: model
    port: 9001
    field: software_engineering
    tensor_parallel_size: 4
    gpu_ids: [0, 1]
arbiter:
  model: arb
  port: 9003
router:
  port: 8000
"""
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="gpu_ids"):
            load_config(cfg_file)

    def test_tp_1_with_no_gpu_ids_is_valid(self, tmp_path) -> None:
        yaml_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: model
    port: 9001
    field: software_engineering
    tensor_parallel_size: 1
arbiter:
  model: arb
  port: 9003
router:
  port: 8000
"""
        cfg_file = tmp_path / "ok.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file)
        assert cfg.specialists[0].tensor_parallel_size == 1

    def test_power_of_2_values_accepted(self, tmp_path) -> None:
        for tp in [1, 2, 4, 8]:
            yaml_content = f"""
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: model
    port: 9001
    field: software_engineering
    tensor_parallel_size: {tp}
    gpu_ids: {list(range(tp))}
    gpu_memory_utilization: 0.90
    quantization: null
arbiter:
  model: arb
  port: 9003
router:
  port: 8000
"""
            cfg_file = tmp_path / f"tp{tp}.yaml"
            cfg_file.write_text(yaml_content)
            cfg = load_config(cfg_file)
            assert cfg.specialists[0].tensor_parallel_size == tp
