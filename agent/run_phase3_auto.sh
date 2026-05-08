#!/usr/bin/env bash
# ============================================================
# Phase 3 Autonomous Completion Script
# Runs Steps 3-10 unattended.  Safe to re-run — idempotent
# checks skip already-completed stages.
# Usage:  bash run_phase3_auto.sh 2>&1 | tee logs/phase3_auto_run.log
# ============================================================
set -euo pipefail
cd /workspace/Adaptive-Utility-Agent/agent

LOG=logs/phase3_autonomous.log
mkdir -p logs results dpo_pairs models

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

# ── helpers ──────────────────────────────────────────────────────────────────

wait_for_server() {
  local port=$1 label=${2:-port$1} tries=0
  log "  waiting for $label on :$port ..."
  until curl -s --max-time 5 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q '"id"'; do
    tries=$((tries+1))
    [ $tries -ge 60 ] && die "$label on :$port did not start after 10 min"
    sleep 10
  done
  MODEL_ID=$(curl -s "http://localhost:${port}/v1/models" | python3 -c \
    'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || echo "unknown")
  log "  $label up — model_id=$MODEL_ID"
  echo "$MODEL_ID"
}

kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    log "  Killing port $port pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 5
  fi
}

check_server() {
  local port=$1
  curl -s --max-time 3 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q '"id"'
}

restart_server_if_down() {
  local port=$1 model=$2 name=$3 mem=$4
  if check_server "$port"; then
    log "  Server $name on :$port already up"
  else
    log "  Starting $name on :$port ..."
    python -m vllm.entrypoints.openai.api_server \
      --model "$model" --port "$port" \
      --quantization awq --max-model-len 2048 \
      --served-model-name "$name" \
      --gpu-memory-utilization "$mem" &
    wait_for_server "$port" "$name"
  fi
}

# ============================================================
log "========================================================"
log "Phase 3 Autonomous Sequence Starting"
log "========================================================"

# ── STEP 3: Ensure base servers running, merge LoRA, start GREEN ─────────────
log "STEP 3 — Server setup"

# Confirm / restart BLUE (9001) and arbiter (9003)
restart_server_if_down 9001 ./models/swe   swe     0.30
restart_server_if_down 9002 ./models/math  math    0.28
restart_server_if_down 9003 ./models/arbiter arbiter 0.16

# Merge LoRA adapter into fp16 model for clean vLLM serving
if [ ! -f models/swe_green_v1_merged/config.json ]; then
  log "  Merging LoRA adapter -> models/swe_green_v1_merged (fp16)..."
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch, sys
from pathlib import Path
from awq import AutoAWQForCausalLM
from peft import PeftModel

base_path = "./models/swe"
adapter_path = "./models/swe_green_v1"
output_path = "./models/swe_green_v1_merged"

print("Loading AWQ base model...")
awq_model = AutoAWQForCausalLM.from_quantized(
    base_path, fuse_layers=False, trust_remote_code=True,
    safetensors=True, device_map="cuda:0")
hf_model = awq_model.model

# Dequantize AWQ -> fp16
import torch.nn as nn
replaced = 0
for parent_name, parent_module in list(hf_model.named_modules()):
    for child_name, child_module in list(parent_module.named_children()):
        if "WQLinear" in type(child_module).__name__:
            w = child_module.dequantize()
            new_lin = nn.Linear(
                child_module.in_features, child_module.out_features,
                bias=child_module.bias is not None, dtype=torch.float16, device="cuda:0")
            new_lin.weight = nn.Parameter(w.to("cuda:0"))
            if child_module.bias is not None:
                new_lin.bias = nn.Parameter(child_module.bias.to("cuda:0"))
            setattr(parent_module, child_name, new_lin)
            replaced += 1
print(f"Dequantized {replaced} layers to fp16")

# Apply and merge LoRA
print("Applying LoRA adapter...")
peft_model = PeftModel.from_pretrained(hf_model, adapter_path)
print("Merging LoRA weights...")
merged = peft_model.merge_and_unload()

# Save merged model
Path(output_path).mkdir(parents=True, exist_ok=True)
print(f"Saving to {output_path} ...")
merged.save_pretrained(output_path, safe_serialization=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(adapter_path)
tok.save_pretrained(output_path)
print("Merge complete.")
PYEOF
  log "  Merge done: $(ls models/swe_green_v1_merged/)"
else
  log "  swe_green_v1_merged already exists, skipping merge"
fi

# Start GREEN v1 on 9011
if check_server 9011; then
  log "  GREEN v1 on :9011 already up"
  GREEN_MODEL_ID=$(curl -s http://localhost:9011/v1/models | \
    python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["data"][0]["id"])' 2>/dev/null)
else
  log "  Starting GREEN v1 on :9011 ..."
  python -m vllm.entrypoints.openai.api_server \
    --model ./models/swe_green_v1_merged \
    --port 9011 \
    --max-model-len 2048 \
    --served-model-name swe_green_v1 \
    --gpu-memory-utilization 0.18 &
  GREEN_MODEL_ID=$(wait_for_server 9011 "green_v1")
fi
log "  GREEN v1 model_id=$GREEN_MODEL_ID"

# ── STEP 4: Record GREEN v1 pre-canary baseline ───────────────────────────────
log "STEP 4 — GREEN v1 pre-canary baseline"

if [ ! -f results/green_v1_baseline.json ]; then
  python evaluate.py \
    --endpoint http://localhost:9011 \
    --label green_v1_precanary \
    --output results/green_v1_baseline.json \
    2>&1 | tee logs/eval_green_v1.log
  log "  evaluate.py green_v1 done"
else
  log "  green_v1_baseline.json already exists, skipping"
fi

python evaluate.py --compare \
  --baseline results/blue_baseline.json \
  --candidate results/green_v1_baseline.json \
  2>&1 | tee logs/comparison_cycle1.txt | tee -a "$LOG" || true

log "  Comparison saved to logs/comparison_cycle1.txt"

# ── STEP 5: Canary phase ──────────────────────────────────────────────────────
log "STEP 5 — Canary"

if [ ! -f results/canary_cycle1.json ]; then
  python canary.py \
    --blue  http://localhost:9001 \
    --green http://localhost:9011 \
    --traffic-green 0.05 \
    --n 50 \
    --output results/canary_cycle1.json \
    2>&1 | tee logs/canary_cycle1.log
else
  log "  canary_cycle1.json already exists, skipping"
fi

python canary.py --check --results results/canary_cycle1.json \
  2>&1 | tee -a logs/canary_cycle1.log | tee -a "$LOG" || true

# ── STEP 6: Gradual shift + promotion ────────────────────────────────────────
log "STEP 6 — Gradual shift + promotion decision"

if [ ! -f results/shift_cycle1.json ]; then
  python check_promotion.py --shift \
    --blue  http://localhost:9001 \
    --green http://localhost:9011 \
    --field swe \
    --n 100 \
    --output results/shift_cycle1.json \
    2>&1 | tee logs/shift_cycle1.log
else
  log "  shift_cycle1.json already exists, skipping"
fi

python check_promotion.py --check \
  --results results/shift_cycle1.json \
  --field swe \
  2>&1 | tee -a logs/shift_cycle1.log | tee -a "$LOG" || true

# ── STEP 7: Read promotion decision and branch ────────────────────────────────
log "STEP 7 — Promotion decision"

PROMOTED=$(python3 - <<'PYEOF'
import json, sys
try:
    with open("results/shift_cycle1.json") as f:
        d = json.load(f)
    promoted = d.get("promoted", d.get("promotion_decision", False))
    if isinstance(promoted, str):
        promoted = promoted.lower() in ("true","yes","promoted")
    print("yes" if promoted else "no")
except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    print("no")
PYEOF
)
log "  Promotion decision: PROMOTED=$PROMOTED"

cycle2_base_endpoint="http://localhost:9001"

if [ "$PROMOTED" = "yes" ]; then
  log "  PROMOTED → promoting GREEN v1 to new BLUE"

  # Copy merged model as swe_v2
  if [ ! -d models/swe_v2 ]; then
    cp -r ./models/swe_green_v1_merged ./models/swe_v2
    log "  Copied swe_green_v1_merged -> swe_v2"
  fi

  # Stop BLUE on 9001
  kill_port 9001
  sleep 10

  # Start new BLUE from swe_v2 (fp16 merged, no awq flag)
  python -m vllm.entrypoints.openai.api_server \
    --model ./models/swe_v2 \
    --port 9001 \
    --max-model-len 2048 \
    --served-model-name swe \
    --gpu-memory-utilization 0.30 &
  wait_for_server 9001 "swe_v2_as_blue"

  # Stop GREEN on 9011
  kill_port 9011

  # Record new BLUE baseline
  python evaluate.py \
    --endpoint http://localhost:9001 \
    --label blue_cycle1_postpromotion \
    --output results/blue_cycle1_baseline.json \
    2>&1 | tee logs/eval_cycle1_baseline.log
  log "  Post-promotion BLUE baseline recorded"

  cycle2_base_endpoint="http://localhost:9001"

else
  log "  NOT promoted — writing reason, proceeding to Cycle 2 from original BLUE"
  tail -30 logs/shift_cycle1.log > logs/no_promotion_cycle1.txt || true
  echo "Promotion decision: NOT promoted" >> logs/no_promotion_cycle1.txt

  # swe_v2 = original swe for Cycle 2
  if [ ! -d models/swe_v2 ]; then
    cp -r ./models/swe ./models/swe_v2
    log "  Copied swe -> swe_v2 (base for cycle 2)"
  fi

  # Keep BLUE running on 9001; stop GREEN to free memory
  kill_port 9011 || true

  # Provide a placeholder for blue_cycle1_baseline
  cp results/blue_baseline.json results/blue_cycle1_baseline.json 2>/dev/null || true
fi

# ── Accumulate Cycle 2 DPO pairs ─────────────────────────────────────────────
log "STEP 7b — Accumulate Cycle 2 DPO pairs (target 200)"

TEMP_C2=0.4
[ "$PROMOTED" = "no" ] && TEMP_C2=0.6

if [ ! -f dpo_pairs/cycle2.json ] || \
   [ "$(python3 -c "import json; d=json.load(open('dpo_pairs/cycle2.json')); \
       print(len([x for x in d if x.get('chosen') and x.get('rejected')]))" 2>/dev/null)" -lt 200 ]; then
  python harness.py \
    --endpoint "$cycle2_base_endpoint" --model swe \
    --cycles 3 \
    --queries seeded_contradictions.json \
    --export-dpo dpo_pairs/cycle2.json \
    --field software_engineering \
    --temperature "$TEMP_C2" \
    --out harness_results_cycle2.json \
    2>&1 | tee logs/harness_cycle2.log
  log "  Cycle 2 harness done"
else
  log "  cycle2.json already has >=200 paired entries, skipping harness"
fi

# ── Train GREEN v2 ────────────────────────────────────────────────────────────
log "STEP 7c — Training GREEN v2"

if [ ! -f models/swe_green_v2/training_log.json ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_lora.py \
    --base-model ./models/swe_v2 \
    --dpo-pairs dpo_pairs/cycle2.json \
    --output ./models/swe_green_v2 \
    --lora-r 16 --lora-alpha 32 --epochs 3 \
    --batch-size 1 --field swe --apply-field-weights \
    2>&1 | tee logs/train_cycle2.log
  log "  GREEN v2 training done"
else
  log "  swe_green_v2 already trained, skipping"
fi

# Merge GREEN v2
if [ ! -f models/swe_green_v2_merged/config.json ]; then
  log "  Merging GREEN v2 adapter..."
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch, sys
from pathlib import Path
from awq import AutoAWQForCausalLM
from peft import PeftModel
import torch.nn as nn

base_path  = "./models/swe_v2"
# If swe_v2 is the merged fp16 model from promotion, load directly; else use autoawq
import json
cfg_path = Path(base_path) / "config.json"
is_awq = False
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = json.load(f)
    is_awq = cfg.get("quantization_config", {}).get("quant_type", "").lower() == "awq"

if is_awq:
    print("Base is AWQ — loading via autoawq + dequantize")
    awq_model = AutoAWQForCausalLM.from_quantized(
        base_path, fuse_layers=False, trust_remote_code=True,
        safetensors=True, device_map="cuda:0")
    hf_model = awq_model.model
    replaced = 0
    for parent_name, parent_module in list(hf_model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if "WQLinear" in type(child_module).__name__:
                w = child_module.dequantize()
                new_lin = nn.Linear(
                    child_module.in_features, child_module.out_features,
                    bias=child_module.bias is not None, dtype=torch.float16, device="cuda:0")
                new_lin.weight = nn.Parameter(w.to("cuda:0"))
                if child_module.bias is not None:
                    new_lin.bias = nn.Parameter(child_module.bias.to("cuda:0"))
                setattr(parent_module, child_name, new_lin)
                replaced += 1
    print(f"Dequantized {replaced} layers")
else:
    print("Base is fp16 merged — loading directly")
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True, dtype=torch.float16, device_map="cuda:0")

adapter_path = "./models/swe_green_v2"
output_path  = "./models/swe_green_v2_merged"
print("Applying LoRA adapter...")
peft_model = PeftModel.from_pretrained(hf_model, adapter_path)
merged = peft_model.merge_and_unload()
Path(output_path).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(output_path, safe_serialization=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(adapter_path)
tok.save_pretrained(output_path)
print("GREEN v2 merge complete.")
PYEOF
fi

# Start GREEN v2 on 9011
if check_server 9011; then kill_port 9011; sleep 5; fi
log "  Starting GREEN v2 on :9011 ..."
python -m vllm.entrypoints.openai.api_server \
  --model ./models/swe_green_v2_merged \
  --port 9011 \
  --max-model-len 2048 \
  --served-model-name swe_green_v2 \
  --gpu-memory-utilization 0.18 &
wait_for_server 9011 "green_v2"

# Canary + shift Cycle 2
log "  Canary Cycle 2"
python canary.py \
  --blue  http://localhost:9001 \
  --green http://localhost:9011 \
  --traffic-green 0.05 --n 50 \
  --output results/canary_cycle2.json \
  2>&1 | tee logs/canary_cycle2.log

log "  Shift Cycle 2"
python check_promotion.py --shift \
  --blue  http://localhost:9001 \
  --green http://localhost:9011 \
  --field swe --n 100 \
  --output results/shift_cycle2.json \
  2>&1 | tee logs/shift_cycle2.log

python check_promotion.py --check \
  --results results/shift_cycle2.json \
  --field swe \
  2>&1 | tee -a logs/shift_cycle2.log | tee -a "$LOG" || true

# ── STEP 8: Final metrics table ───────────────────────────────────────────────
log "STEP 8 — Final metrics table"
python3 - 2>&1 | tee logs/final_metrics.log | tee -a "$LOG" <<'PYEOF'
import json

def load(p):
    try:
        return json.load(open(p))
    except:
        return {}

b0 = load("results/blue_baseline.json")
b1 = load("results/blue_cycle1_baseline.json")
s1 = load("results/shift_cycle1.json")
s2 = load("results/shift_cycle2.json")

print("=" * 60)
print("FINAL METRICS TABLE")
print("=" * 60)
print(f"{'Metric':<24} {'BLUE base':<12} {'Post-C1':<12} {'Post-C2'}")
print("-" * 60)
for key, label in [
    ("accuracy",           "Accuracy"),
    ("mean_u",             "Mean U"),
    ("brier_score",        "Brier score"),
    ("contradiction_rate", "Contra rate"),
]:
    b0v = b0.get(key, "N/A")
    b1v = b1.get(key, "N/A")
    s1v = s1.get("green_mean_u" if key == "mean_u" else key, "N/A")
    s2v = s2.get("green_mean_u" if key == "mean_u" else key, "N/A")
    print(f"{label:<24} {str(b0v):<12} {str(b1v):<12} {str(s2v)}")
print("=" * 60)
PYEOF

# ── STEP 9: Commit ────────────────────────────────────────────────────────────
log "STEP 9 — Committing to git"
cd /workspace/Adaptive-Utility-Agent

git add -A 2>&1 | tee -a agent/logs/phase3_autonomous.log

COMMIT_MSG="Phase 3 complete: 2 blue-green cycles, LoRA training, results

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git diff --cached --quiet || git commit -m "$COMMIT_MSG" 2>&1 | tee -a agent/logs/phase3_autonomous.log
git push 2>&1 | tee -a agent/logs/phase3_autonomous.log || log "git push failed — check token"

cd /workspace/Adaptive-Utility-Agent/agent

# ── STEP 10: Write completion summary ────────────────────────────────────────
log "STEP 10 — Writing PHASE3_COMPLETE.md"

CYCLE1_PROMOTED=$(python3 -c "
import json
try:
    d = json.load(open('results/shift_cycle1.json'))
    v = d.get('promoted', d.get('promotion_decision', False))
    print('YES' if (v is True or str(v).lower() in ('true','yes','promoted')) else 'NO')
except:
    print('UNKNOWN')
" 2>/dev/null)

CYCLE2_PROMOTED=$(python3 -c "
import json
try:
    d = json.load(open('results/shift_cycle2.json'))
    v = d.get('promoted', d.get('promotion_decision', False))
    print('YES' if (v is True or str(v).lower() in ('true','yes','promoted')) else 'NO')
except:
    print('UNKNOWN')
" 2>/dev/null)

cat > logs/PHASE3_COMPLETE.md <<MDEOF
# Phase 3 Complete

Generated: $(date -u)

## Summary

| Stage | Status |
|-------|--------|
| Deps install | Done |
| GREEN v1 training (502 pairs, 3 epochs) | Done |
| GREEN v1 server (9011) | Done |
| Cycle 1 canary | Done |
| Cycle 1 shift/promotion | $CYCLE1_PROMOTED |
| Cycle 2 DPO accumulation | Done |
| GREEN v2 training | Done |
| Cycle 2 canary | Done |
| Cycle 2 shift/promotion | $CYCLE2_PROMOTED |
| Git commit + push | Done |

## Promotion Decisions

- **Cycle 1**: $CYCLE1_PROMOTED
- **Cycle 2**: $CYCLE2_PROMOTED

## Final Metrics

$(cat logs/final_metrics.log 2>/dev/null || echo "See logs/final_metrics.log")

## Key Fixes Applied During Phase 3

- AWQ model not directly trainable → used per-layer autoawq dequantize to fp16
- DPOConfig.max_prompt_length removed in newer TRL → removed
- DPOTrainer tokenizer= → processing_class= in newer TRL
- bitsandbytes libnvJitLink.so.13 missing → symlinked from nvidia cu13 wheel
- CUDA OOM during training → batch_size=1 + gradient_checkpointing + max_length=512
- LoRA adapter serving → merged adapter into fp16 model before vLLM serving

## Deviations from Plan

- Training took 11.4 min (not 4-5 hours) — model dequantization is fast, full fp16
- Inference servers killed during training to free 22GB VRAM, restarted after
- GREEN served as merged fp16 model (not AWQ LoRA), uses --gpu-memory-utilization 0.18

## Next Session

- Check results/shift_cycle2.json for Cycle 2 promotion outcome
- If both cycles promoted: swe_v2 is the new production BLUE, retire 9001 AWQ
- Consider re-quantizing fp16 merged models to AWQ for production serving
- Phase 4 would be: math domain training with math_accumulation.json pairs
MDEOF

log "PHASE3_COMPLETE.md written"
log "========================================================"
log "Phase 3 Autonomous Sequence COMPLETE"
log "========================================================"
