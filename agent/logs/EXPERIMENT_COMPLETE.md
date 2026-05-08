# AUA POC Overnight Experiment — COMPLETE

- **Start:** 2026-05-08T12:20:01Z
- **End:**   2026-05-08T13:57:32Z
- **GPU hours:** ~1.63

## Stage 3 — SWE Calibration

| Metric | Value |
|--------|-------|
| Cycle 1 promoted | N/A |
| Cycle 2 promoted | N/A |
| Best SWE model | ./models/swe |
| Training method | AWQ dequantize → fp16 LoRA (QLoRA bypass) |

## Stage 4 — Math Calibration

| Metric | Value |
|--------|-------|
| Cycle 1 promoted | N/A |
| Cycle 2 promoted | N/A |
| Best Math model | ./models/math |

## Stage 5 — Cross-Domain Validation

See logs/final_metrics.log and logs/routing_calibrated.log for full numbers.

import json
from pathlib import Path

def load(p):
    try:    return json.load(open(p))
    except: return {}

stage1 = {
    "arm_A_accuracy": 0.433, "arm_D_accuracy": 0.867,
    "arm_A_mean_u":   0.543, "arm_D_mean_u":   0.633,
    "p_value": 0.0003
}

blue        = load("results/blue_baseline.json")
math_base   = load("results/math_blue_baseline.json")
swe_c1      = load("results/swe_shift_cycle1.json")
swe_c2      = load("results/swe_shift_cycle2.json")
math_c1     = load("results/math_shift_cycle1.json")
math_c2     = load("results/math_shift_cycle2.json")
battery     = load("results/cross_domain_battery.json")

print("=" * 65)
print("COMPLETE EXPERIMENT RESULTS")
print("=" * 65)

print("\nStage 1 routing (uncalibrated):")
print(f"  Arm A (no routing):  {stage1['arm_A_accuracy']*100:.1f}% accuracy,  mean_U={stage1['arm_A_mean_u']:.4f}")
print(f"  Arm D (VCG routing): {stage1['arm_D_accuracy']*100:.1f}% accuracy,  mean_U={stage1['arm_D_mean_u']:.4f}  p={stage1['p_value']}")

print("\nSWE BLUE baseline (pre-training):")
for k in ["accuracy", "mean_u", "brier_score", "contradiction_rate"]:
    print(f"  {k}: {blue.get(k, 'N/A')}")

print("\nSWE calibration cycles:")
for label, d in [("Cycle 1", swe_c1), ("Cycle 2", swe_c2)]:
    if d:
        prom = d.get("promoted", d.get("promote", "N/A"))
        u_d  = d.get("u_delta", d.get("green_mean_u","?"))
        print(f"  {label}: promoted={prom}  u_delta/green_u={u_d}")
    else:

## Key Fixes Applied

- AWQ models cannot be loaded fp16 directly (no weight tensors) → autoawq per-layer dequantize
- PEFT cannot wrap WQLinear layers → dequantize to fp16 Linear first
- DPOConfig: removed max_prompt_length (removed in TRL), added gradient_checkpointing
- DPOTrainer: processing_class= not tokenizer= (TRL API change)
- bitsandbytes: libnvJitLink.so.13 symlinked from nvidia cu13 wheel
- OOM at batch_size=2: use batch_size=1 + gradient_checkpointing + max_length=512
- Merged fp16 models served with --dtype float16, no --quantization flag

## Deviations from Plan

- Training takes ~12 min not 4-5 hours (fp16 LoRA vs quantized training)
- Inference servers killed during training (VRAM constraint on 24GB 4090)
- GREEN models served as merged fp16 (not AWQ LoRA) for vLLM compatibility
- Math BLUE baseline recorded before any training (reordered for VRAM efficiency)

## Next Session Should Start With

- Review EXPERIMENT_COMPLETE.md and logs/final_metrics.log
- If SWE or Math not promoted: consider more DPO accumulation runs
- Stage 6 planning: distributed microservices architecture
- Current production models: 9001=./models/swe  9002=./models/math  9003=arbiter
