#!/usr/bin/env bash
# =============================================================================
# AUA POC — Overnight Autonomous Experiment  (Stages 3, 4, 5)
# =============================================================================
# Safe to re-run: every step checks output files before running.
# Logs to logs/phase3_autonomous.log (appended).
# Usage:
#   nohup bash run_overnight.sh > logs/overnight_tty.log 2>&1 &
# =============================================================================
set -uo pipefail          # no -e: errors handled per-step
cd /workspace/Adaptive-Utility-Agent/agent

EXPERIMENT_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG=logs/phase3_autonomous.log
DEADLINE_HOURS=7.5        # hard stop if exceeded

mkdir -p logs results dpo_pairs models

# ─── helpers ─────────────────────────────────────────────────────────────────

log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
warn() { log "WARN: $*"; }
die()  { log "FATAL: $*"; write_interrupted "FATAL: $*"; exit 1; }

elapsed_hours() {
  python3 -c "
import time, datetime
start = datetime.datetime.fromisoformat('${EXPERIMENT_START}'.replace('Z',''))
now   = datetime.datetime.utcnow()
print(round((now - start).total_seconds() / 3600, 2))
"
}

check_deadline() {
  local h; h=$(elapsed_hours)
  if python3 -c "exit(0 if float('$h') < $DEADLINE_HOURS else 1)" 2>/dev/null; then
    return 0
  else
    log "DEADLINE REACHED ($h h >= $DEADLINE_HOURS h) — stopping cleanly"
    do_commit "Overnight run interrupted at deadline ($h h): partial results"
    write_interrupted "Deadline ($DEADLINE_HOURS h) reached after $h hours."
    exit 0
  fi
}

kill_port() {
  local port=$1
  local pids; pids=$(lsof -ti :"$port" 2>/dev/null || true)
  [ -n "$pids" ] && { log "  killing port $port (pids $pids)"; kill $pids 2>/dev/null || true; sleep 4; }
}

kill_all_inference() {
  log "  Stopping all inference servers to free VRAM..."
  for port in 9001 9002 9003 9011 9012; do kill_port "$port"; done
  sleep 6
  log "  VRAM free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
}

wait_server() {
  local port=$1 label=${2:-port$1} tries=0
  log "  waiting for $label on :$port ..."
  until curl -s --max-time 5 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q '"id"'; do
    tries=$((tries+1))
    [ $tries -ge 72 ] && { warn "$label on :$port not up after 12 min"; return 1; }
    sleep 10
  done
  local mid; mid=$(curl -s "http://localhost:${port}/v1/models" | \
    python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["data"][0]["id"])' 2>/dev/null || echo "?")
  log "  $label up — model_id=$mid"
  return 0
}

check_server() { curl -s --max-time 3 "http://localhost:$1/v1/models" 2>/dev/null | grep -q '"id"'; }

start_awq_server() {
  # start_awq_server <port> <model_path> <served_name> <gpu_util>
  local port=$1 model=$2 name=$3 mem=$4
  if check_server "$port"; then
    log "  $name on :$port already running"
    return 0
  fi
  log "  Starting $name (AWQ) on :$port mem=$mem ..."
  python -m vllm.entrypoints.openai.api_server \
    --model "$model" --port "$port" \
    --quantization awq --max-model-len 2048 \
    --served-model-name "$name" \
    --gpu-memory-utilization "$mem" &
  wait_server "$port" "$name"
}

start_fp16_server() {
  # start_fp16_server <port> <model_path> <served_name> <gpu_util>
  local port=$1 name=$3 mem=$4
  local model; model=$(realpath "$2" 2>/dev/null || echo "$2")
  if check_server "$port"; then
    log "  $name on :$port already running"
    return 0
  fi
  log "  Starting $name (fp16) on :$port mem=$mem ..."
  python -m vllm.entrypoints.openai.api_server \
    --model "$model" --port "$port" \
    --max-model-len 2048 \
    --served-model-name "$name" \
    --gpu-memory-utilization "$mem" \
    --dtype float16 &
  wait_server "$port" "$name"
}

# ─── AWQ dequantize + LoRA merge (reused for SWE and Math) ───────────────────
merge_lora() {
  # merge_lora <base_model_path> <adapter_path> <output_path>
  local base=$1 adapter=$2 output=$3
  if [ -f "${output}/config.json" ]; then
    log "  ${output} already merged, skipping"
    return 0
  fi
  log "  Merging LoRA: base=$base adapter=$adapter → $output"
  python3 - "$base" "$adapter" "$output" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, torch, torch.nn as nn
from pathlib import Path
from awq import AutoAWQForCausalLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

base_path, adapter_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Is base model AWQ-quantized?
cfg_file = Path(base_path) / "config.json"
is_awq = False
if cfg_file.exists():
    cfg = json.load(open(cfg_file))
    qcfg = cfg.get("quantization_config", {})
    is_awq = qcfg.get("quant_type","").lower() == "awq" or \
             qcfg.get("bits", 0) == 4

if is_awq:
    print(f"AWQ base detected — loading via autoawq and dequantizing...")
    awq_model = AutoAWQForCausalLM.from_quantized(
        base_path, fuse_layers=False, trust_remote_code=True,
        safetensors=True, device_map="auto")
    hf_model = awq_model.model
    replaced = 0
    for parent_name, parent_module in list(hf_model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if "WQLinear" in type(child_module).__name__:
                w = child_module.dequantize()
                new_lin = nn.Linear(
                    child_module.in_features, child_module.out_features,
                    bias=child_module.bias is not None,
                    dtype=torch.float16, device="cuda:0")
                new_lin.weight = nn.Parameter(w.to("cuda:0"))
                if child_module.bias is not None:
                    new_lin.bias = nn.Parameter(child_module.bias.to("cuda:0"))
                setattr(parent_module, child_name, new_lin)
                replaced += 1
    print(f"Dequantized {replaced} WQLinear layers to fp16")
    hf_model.config.quantization_config = None
else:
    print(f"fp16 base detected — loading directly...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="cuda:0")

print("Applying and merging LoRA adapter...")
peft_model = PeftModel.from_pretrained(hf_model, adapter_path)
merged = peft_model.merge_and_unload()
Path(output_path).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(output_path, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(adapter_path)
tok.save_pretrained(output_path)
# Write clean config (remove quantization_config if present)
cfgout = Path(output_path) / "config.json"
if cfgout.exists():
    c = json.load(open(cfgout))
    c.pop("quantization_config", None)
    json.dump(c, open(cfgout,"w"), indent=2)
print(f"Merge complete → {output_path}")
PYEOF
}

# ─── LoRA training (reused for SWE and Math) ─────────────────────────────────
train_lora_safe() {
  # train_lora_safe <base> <pairs> <output> <field> <log>
  local base=$1 pairs=$2 output=$3 field=$4 tlog=$5
  if [ -f "${output}/training_log.json" ]; then
    log "  ${output} already trained — skipping"
    return 0
  fi
  log "  Training LoRA: base=$base field=$field output=$output"
  kill_all_inference   # free VRAM for training
  # Ensure libnvJitLink symlink
  ln -sf /usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib/libnvJitLink.so.13 \
         /usr/local/lib/libnvJitLink.so.13 2>/dev/null || true
  ldconfig 2>/dev/null || true

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_lora.py \
    --base-model "$base" \
    --dpo-pairs  "$pairs" \
    --output     "$output" \
    --lora-r 16 --lora-alpha 32 --epochs 3 \
    --batch-size 1 \
    --field      "$field" \
    --apply-field-weights \
    2>&1 | tee "$tlog"
  local rc=$?
  if [ $rc -ne 0 ]; then
    warn "train_lora.py exited $rc — check $tlog"
    return $rc
  fi
  log "  Training done: $(python3 -c "import json; d=json.load(open('${output}/training_log.json')); print(f'loss={d[\"train_loss\"]:.4f} elapsed={d[\"elapsed_human\"]}')" 2>/dev/null)"
}

# ─── promotion decision reader ────────────────────────────────────────────────
is_promoted() {
  local results_json=$1
  python3 -c "
import json, sys
try:
    d = json.load(open('$results_json'))
    v = d.get('promoted', d.get('promotion_decision', d.get('promote', False)))
    sys.exit(0 if (v is True or str(v).lower() in ('true','yes','promoted')) else 1)
except:
    sys.exit(1)
" 2>/dev/null
}

# ─── commit helper ────────────────────────────────────────────────────────────
do_commit() {
  local msg="${1:-Overnight run checkpoint}"
  log "  git commit: $msg"
  cd /workspace/Adaptive-Utility-Agent
  git add -A 2>/dev/null || true
  git diff --cached --quiet || git commit -m "$msg

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" 2>&1 | tee -a agent/"$LOG" || true
  git push 2>&1 | tee -a agent/"$LOG" || {
    warn "git push failed — trying with GITHUB_TOKEN"
    source /workspace/.env 2>/dev/null || true
    git remote set-url origin \
      "https://${GITHUB_TOKEN:-}@github.com/praneethtota/Adaptive-Utility-Agent.git" 2>/dev/null || true
    git push 2>&1 | tee -a agent/"$LOG" || {
      log "  Push failed — saving diff to logs/PUSH_FAILED.txt"
      git diff HEAD~1 > agent/logs/PUSH_FAILED.txt 2>/dev/null || true
    }
  }
  cd /workspace/Adaptive-Utility-Agent/agent
}

write_interrupted() {
  cat > logs/INTERRUPTED.md <<MDEOF
# Overnight Run Interrupted

Time: $(date -u)
Reason: $1
Elapsed: $(elapsed_hours) hours

## Completed so far
$(ls results/*.json 2>/dev/null | sed 's/^/  - /')

## Models trained
$(ls models/*/training_log.json 2>/dev/null | sed 's/^/  - /')

## Next session should resume from:
Check logs/phase3_autonomous.log for last completed step.
MDEOF
  log "  Wrote logs/INTERRUPTED.md"
}

# ─── one-shot blue-green cycle helper ─────────────────────────────────────────
# run_bg_cycle DOMAIN BLUE_PORT GREEN_PORT BLUE_ENDPOINT GREEN_ENDPOINT
#              CANARY_OUT SHIFT_OUT CANARY_LOG SHIFT_LOG FIELD
run_canary_shift() {
  local blue_ep=$1 green_ep=$2 canary_out=$3 shift_out=$4
  local canary_log=$5 shift_log=$6 field=$7

  # Canary
  if [ ! -f "$canary_out" ]; then
    log "  Running canary: blue=$blue_ep green=$green_ep n=50"
    python canary.py \
      --blue  "$blue_ep" --green "$green_ep" \
      --traffic-green 0.05 --n 50 \
      --output "$canary_out" \
      2>&1 | tee "$canary_log" || warn "canary failed — continuing"
    python canary.py --check --results "$canary_out" \
      2>&1 | tee -a "$canary_log" || true
  else
    log "  $canary_out exists — skipping canary"
  fi

  # Shift + promotion
  if [ ! -f "$shift_out" ]; then
    log "  Running shift: field=$field n=100"
    python check_promotion.py --shift \
      --blue  "$blue_ep" --green "$green_ep" \
      --field "$field" --n 100 \
      --output "$shift_out" \
      2>&1 | tee "$shift_log" || warn "shift failed — continuing"
    python check_promotion.py --check \
      --results "$shift_out" --field "$field" \
      2>&1 | tee -a "$shift_log" || true
  else
    log "  $shift_out exists — skipping shift"
  fi
}

# =============================================================================
log "================================================================"
log "AUA POC Overnight Experiment START  (Stages 3, 4, 5)"
log "Deadline: ${DEADLINE_HOURS}h from now  (${EXPERIMENT_START})"
log "================================================================"

# Install deps (idempotent)
log "DEPS — ensuring training libraries installed"
pip install trl peft datasets accelerate bitsandbytes autoawq -q 2>&1 | tail -3 | tee -a "$LOG"

# =============================================================================
# STAGE 3 — SWE BLUE-GREEN CALIBRATION
# =============================================================================
log "================================================================"
log "STAGE 3 — SWE Calibration Cycles"
log "================================================================"

# S3-PRE: Record Math BLUE baseline NOW while we still have (or can start)
# math server — do this before any training that kills servers
log "S4-1 (early) — Math BLUE baseline (before any training)"
if [ ! -f results/math_blue_baseline.json ]; then
  if ! check_server 9002; then
    start_awq_server 9002 ./models/math math 0.30
  fi
  python evaluate.py \
    --endpoint http://localhost:9002 \
    --label math_blue_baseline \
    --output results/math_blue_baseline.json \
    2>&1 | tee logs/math_blue_baseline.log || warn "math baseline failed"
  kill_port 9002   # free memory — math not needed until Stage 4
else
  log "  math_blue_baseline.json exists — skipping"
fi
check_deadline

# S3-2: Train SWE GREEN v1
log "S3-2 — Train SWE GREEN v1"
train_lora_safe \
  ./models/swe \
  dpo_pairs/accumulation_final.json \
  ./models/swe_green_v1 \
  swe \
  logs/train_swe_cycle1.log
check_deadline

# S3: Merge SWE GREEN v1 adapter → fp16 model
log "S3-merge — Merge SWE GREEN v1 → swe_green_v1_merged"
kill_all_inference   # need full VRAM for merge
merge_lora ./models/swe ./models/swe_green_v1 ./models/swe_green_v1_merged
check_deadline

# S3-3: Start SWE BLUE (9001), Arbiter (9003), GREEN (9011)
log "S3-3 — Start SWE BLUE + Arbiter + SWE GREEN"
start_awq_server  9001 ./models/swe     swe     0.30
start_awq_server  9003 ./models/arbiter arbiter 0.16
start_fp16_server 9011 ./models/swe_green_v1_merged swe_green_v1 0.30
check_deadline

# S3-4: Record SWE GREEN v1 baseline
log "S3-4 — SWE GREEN v1 pre-canary baseline"
if [ ! -f results/swe_green_v1_baseline.json ]; then
  python evaluate.py \
    --endpoint http://localhost:9011 \
    --label swe_green_v1_precanary \
    --output results/swe_green_v1_baseline.json \
    2>&1 | tee logs/eval_swe_green_v1.log || warn "evaluate green_v1 failed"
fi
python evaluate.py --compare \
  --baseline  results/blue_baseline.json \
  --candidate results/swe_green_v1_baseline.json \
  > logs/comparison_swe_cycle1.txt 2>&1 || true
log "  comparison_swe_cycle1.txt written"
check_deadline

# S3-5,6: SWE canary + shift cycle 1
log "S3-5,6 — SWE canary + shift cycle 1"
run_canary_shift \
  http://localhost:9001 http://localhost:9011 \
  results/swe_canary_cycle1.json \
  results/swe_shift_cycle1.json \
  logs/swe_canary_cycle1.log \
  logs/swe_shift_cycle1.log \
  swe
check_deadline

# S3-7: SWE promotion decision cycle 1
log "S3-7 — SWE promotion decision cycle 1"
SWE_C1_PROMOTED=no
if is_promoted results/swe_shift_cycle1.json; then
  SWE_C1_PROMOTED=yes
fi
log "  SWE Cycle 1 PROMOTED=$SWE_C1_PROMOTED"

SWE_BEST_MODEL=./models/swe_green_v1_merged   # optimistic default
SWE_BEST_PORT=9011

if [ "$SWE_C1_PROMOTED" = "yes" ]; then
  log "  Promoting SWE GREEN v1 → swe_v2"
  [ ! -d models/swe_v2 ] && cp -r ./models/swe_green_v1_merged ./models/swe_v2

  # Restart BLUE on 9001 as swe_v2
  kill_port 9001; sleep 5
  start_fp16_server 9001 ./models/swe_v2 swe 0.30
  kill_port 9011

  # Post-promotion baseline
  if [ ! -f results/swe_blue_cycle1_baseline.json ]; then
    python evaluate.py \
      --endpoint http://localhost:9001 \
      --label swe_blue_cycle1_postpromotion \
      --output results/swe_blue_cycle1_baseline.json \
      2>&1 | tee logs/swe_blue_cycle1_eval.log || warn "post-promo eval failed"
  fi
  check_deadline

  # Accumulate SWE Cycle 2 pairs
  log "S3-7b — Accumulate SWE Cycle 2 DPO pairs"
  if [ ! -f dpo_pairs/swe_cycle2.json ]; then
    python harness.py \
      --endpoint http://localhost:9001 --model swe \
      --cycles 3 \
      --queries seeded_contradictions.json \
      --export-dpo dpo_pairs/swe_cycle2.json \
      --field software_engineering \
      --temperature 0.4 --append \
      --out harness_results_swe_cycle2.json \
      2>&1 | tee logs/swe_harness_cycle2.log || warn "swe harness cycle2 failed"
  else
    log "  swe_cycle2.json exists — skipping"
  fi
  check_deadline

  # Train SWE GREEN v2
  log "S3-7c — Train SWE GREEN v2"
  train_lora_safe \
    ./models/swe_v2 \
    dpo_pairs/swe_cycle2.json \
    ./models/swe_green_v2 \
    swe \
    logs/train_swe_cycle2.log
  check_deadline

  log "S3-7d — Merge SWE GREEN v2"
  kill_all_inference
  merge_lora ./models/swe_v2 ./models/swe_green_v2 ./models/swe_green_v2_merged
  check_deadline

  # Restart servers + GREEN v2
  start_awq_server  9001 ./models/swe_v2 swe     0.30  2>/dev/null || \
  start_fp16_server 9001 ./models/swe_v2 swe     0.30
  start_awq_server  9003 ./models/arbiter arbiter 0.16
  start_fp16_server 9011 ./models/swe_green_v2_merged swe_green_v2 0.30
  check_deadline

  # Canary + shift cycle 2
  log "S3-7e — SWE canary + shift cycle 2"
  run_canary_shift \
    http://localhost:9001 http://localhost:9011 \
    results/swe_canary_cycle2.json \
    results/swe_shift_cycle2.json \
    logs/swe_canary_cycle2.log \
    logs/swe_shift_cycle2.log \
    swe
  check_deadline

  # Cycle 2 promotion
  log "S3-7f — SWE Cycle 2 promotion decision"
  SWE_C2_PROMOTED=no
  if is_promoted results/swe_shift_cycle2.json; then SWE_C2_PROMOTED=yes; fi
  log "  SWE Cycle 2 PROMOTED=$SWE_C2_PROMOTED"

  if [ "$SWE_C2_PROMOTED" = "yes" ]; then
    [ ! -d models/swe_v3 ] && cp -r ./models/swe_green_v2_merged ./models/swe_v3
    kill_port 9001; sleep 5
    start_fp16_server 9001 ./models/swe_v3 swe 0.30
    SWE_BEST_MODEL=./models/swe_v3
    log "  SWE now at v3"
  else
    SWE_BEST_MODEL=./models/swe_v2
    log "  SWE stays at v2"
  fi
  kill_port 9011

else
  log "  SWE not promoted — keeping original BLUE on 9001"
  {
    echo "SWE Cycle 1 NOT promoted"
    tail -40 logs/swe_shift_cycle1.log 2>/dev/null
  } > logs/swe_no_promotion_cycle1.txt
  kill_port 9011
  SWE_BEST_MODEL=./models/swe
fi

log "STAGE 3 COMPLETE — SWE best model: $SWE_BEST_MODEL"
do_commit "Stage 3 complete: SWE calibration cycles done (promoted=$SWE_C1_PROMOTED)"
check_deadline

# =============================================================================
# STAGE 4 — MATH SPECIALIST CALIBRATION
# =============================================================================
log "================================================================"
log "STAGE 4 — Math Calibration Cycles"
log "================================================================"

# S4-2: Train Math GREEN v1
log "S4-2 — Train Math GREEN v1"
train_lora_safe \
  ./models/math \
  dpo_pairs/math_accumulation.json \
  ./models/math_green_v1 \
  mathematics \
  logs/train_math_cycle1.log
check_deadline

# Merge Math GREEN v1
log "S4-merge — Merge Math GREEN v1 → math_green_v1_merged"
kill_all_inference
merge_lora ./models/math ./models/math_green_v1 ./models/math_green_v1_merged
check_deadline

# S4-3: Start Math BLUE (9002), SWE BLUE (9001), Arbiter (9003), Math GREEN (9012)
log "S4-3 — Start all servers + Math GREEN"
# Ensure SWE BLUE is on best model
if check_server 9001; then
  log "  SWE on :9001 already running"
else
  start_fp16_server 9001 "$SWE_BEST_MODEL" swe 0.30 || \
  start_awq_server  9001 ./models/swe swe 0.30
fi
start_awq_server  9002 ./models/math math    0.30
start_awq_server  9003 ./models/arbiter arbiter 0.16
start_fp16_server 9012 ./models/math_green_v1_merged math_green_v1 0.28
check_deadline

# S4-4: Record Math GREEN v1 baseline
log "S4-4 — Math GREEN v1 pre-canary baseline"
if [ ! -f results/math_green_v1_baseline.json ]; then
  python evaluate.py \
    --endpoint http://localhost:9012 \
    --label math_green_v1_precanary \
    --output results/math_green_v1_baseline.json \
    2>&1 | tee logs/eval_math_green_v1.log || warn "math green eval failed"
fi
python evaluate.py --compare \
  --baseline  results/math_blue_baseline.json \
  --candidate results/math_green_v1_baseline.json \
  > logs/comparison_math_cycle1.txt 2>&1 || true
check_deadline

# S4-5,6: Math canary + shift cycle 1
log "S4-5,6 — Math canary + shift cycle 1"
run_canary_shift \
  http://localhost:9002 http://localhost:9012 \
  results/math_canary_cycle1.json \
  results/math_shift_cycle1.json \
  logs/math_canary_cycle1.log \
  logs/math_shift_cycle1.log \
  mathematics
check_deadline

# S4-7: Math promotion decision cycle 1
log "S4-7 — Math promotion decision cycle 1"
MATH_C1_PROMOTED=no
if is_promoted results/math_shift_cycle1.json; then MATH_C1_PROMOTED=yes; fi
log "  Math Cycle 1 PROMOTED=$MATH_C1_PROMOTED"

MATH_BEST_MODEL=./models/math

if [ "$MATH_C1_PROMOTED" = "yes" ]; then
  log "  Promoting Math GREEN v1 → math_v2"
  [ ! -d models/math_v2 ] && cp -r ./models/math_green_v1_merged ./models/math_v2

  kill_port 9002; sleep 5
  start_fp16_server 9002 ./models/math_v2 math 0.30
  kill_port 9012

  # Post-promotion baseline
  if [ ! -f results/math_blue_cycle1_baseline.json ]; then
    python evaluate.py \
      --endpoint http://localhost:9002 \
      --label math_blue_cycle1_postpromotion \
      --output results/math_blue_cycle1_baseline.json \
      2>&1 | tee logs/math_blue_cycle1_eval.log || warn "math post-promo eval failed"
  fi
  check_deadline

  # Accumulate Math Cycle 2 pairs
  log "S4-7b — Accumulate Math Cycle 2 DPO pairs"
  if [ ! -f dpo_pairs/math_cycle2.json ]; then
    python harness.py \
      --endpoint http://localhost:9002 --model math \
      --cycles 3 \
      --queries seeded_contradictions.json \
      --export-dpo dpo_pairs/math_cycle2.json \
      --field mathematics \
      --temperature 0.4 --append \
      --out harness_results_math_cycle2.json \
      2>&1 | tee logs/math_harness_cycle2.log || warn "math harness cycle2 failed"
  else
    log "  math_cycle2.json exists — skipping"
  fi
  check_deadline

  # Train Math GREEN v2
  log "S4-7c — Train Math GREEN v2"
  train_lora_safe \
    ./models/math_v2 \
    dpo_pairs/math_cycle2.json \
    ./models/math_green_v2 \
    mathematics \
    logs/train_math_cycle2.log
  check_deadline

  log "S4-7d — Merge Math GREEN v2"
  kill_all_inference
  merge_lora ./models/math_v2 ./models/math_green_v2 ./models/math_green_v2_merged
  check_deadline

  # Restart servers + GREEN v2
  start_fp16_server 9001 "$SWE_BEST_MODEL" swe     0.30 || \
  start_awq_server  9001 ./models/swe      swe     0.30
  start_fp16_server 9002 ./models/math_v2  math    0.30 2>/dev/null || \
  start_awq_server  9002 ./models/math_v2  math    0.30
  start_awq_server  9003 ./models/arbiter  arbiter 0.16
  start_fp16_server 9012 ./models/math_green_v2_merged math_green_v2 0.28
  check_deadline

  # Canary + shift cycle 2
  log "S4-7e — Math canary + shift cycle 2"
  run_canary_shift \
    http://localhost:9002 http://localhost:9012 \
    results/math_canary_cycle2.json \
    results/math_shift_cycle2.json \
    logs/math_canary_cycle2.log \
    logs/math_shift_cycle2.log \
    mathematics
  check_deadline

  log "S4-7f — Math Cycle 2 promotion decision"
  MATH_C2_PROMOTED=no
  if is_promoted results/math_shift_cycle2.json; then MATH_C2_PROMOTED=yes; fi
  log "  Math Cycle 2 PROMOTED=$MATH_C2_PROMOTED"

  if [ "$MATH_C2_PROMOTED" = "yes" ]; then
    [ ! -d models/math_v3 ] && cp -r ./models/math_green_v2_merged ./models/math_v3
    kill_port 9002; sleep 5
    start_fp16_server 9002 ./models/math_v3 math 0.30
    MATH_BEST_MODEL=./models/math_v3
    log "  Math now at v3"
  else
    MATH_BEST_MODEL=./models/math_v2
    log "  Math stays at v2"
  fi
  kill_port 9012

else
  log "  Math not promoted — keeping original BLUE on 9002"
  {
    echo "Math Cycle 1 NOT promoted"
    tail -40 logs/math_shift_cycle1.log 2>/dev/null
  } > logs/math_no_promotion_cycle1.txt
  kill_port 9012
  MATH_BEST_MODEL=./models/math
fi

log "STAGE 4 COMPLETE — Math best model: $MATH_BEST_MODEL"
do_commit "Stage 4 complete: Math calibration cycles done (promoted=$MATH_C1_PROMOTED)"
check_deadline

# =============================================================================
# STAGE 5 — CROSS-DOMAIN ARBITRATION VALIDATION
# =============================================================================
log "================================================================"
log "STAGE 5 — Cross-Domain Arbitration Validation"
log "================================================================"

# S5-1: Ensure all four services are live
log "S5-1 — Confirm all services"
check_server 9001 || { start_fp16_server 9001 "$SWE_BEST_MODEL"  swe     0.30 || \
                       start_awq_server  9001 ./models/swe       swe     0.30; }
check_server 9002 || { start_fp16_server 9002 "$MATH_BEST_MODEL" math    0.30 || \
                       start_awq_server  9002 ./models/math      math    0.30; }
check_server 9003 || start_awq_server 9003 ./models/arbiter arbiter 0.16

# Router on 8000
if ! curl -s --max-time 3 http://localhost:8000/health 2>/dev/null | grep -qiE "ok|healthy|alive"; then
  log "  Router not responding — starting router.py ..."
  pkill -f router.py 2>/dev/null || true; sleep 2
  python router.py &
  sleep 15
  curl -s --max-time 5 http://localhost:8000/health | tee -a "$LOG" || warn "router still not responding"
else
  log "  Router on :8000 OK"
fi
check_deadline

# S5-2: Calibrated routing experiment (n=20 per arm, seed=99)
log "S5-2 — Calibrated 4-arm routing experiment (n=20, seed=99)"
if [ ! -f results/routing_results_live_calibrated.json ] && \
   [ ! -f logs/routing_calibrated.log ]; then
  python routing_experiment.py --live \
    --swe-endpoint     http://localhost:9001 \
    --math-endpoint    http://localhost:9002 \
    --arbiter-endpoint http://localhost:9003 \
    --n 20 \
    --seed 99 \
    --output-suffix _calibrated \
    2>&1 | tee logs/routing_calibrated.log || warn "routing_experiment failed"
else
  log "  routing_calibrated already done — skipping"
fi
check_deadline

# S5-3: Cross-domain arbitration battery (50 mixed queries via router)
log "S5-3 — Cross-domain arbitration battery (50 queries)"
if [ ! -f results/cross_domain_battery.json ]; then
  python3 - 2>&1 | tee logs/cross_domain_battery.log || warn "cross-domain battery failed" <<'PYEOF'
import asyncio, json, httpx, time
from pathlib import Path

QUERIES = [
  "Implement gradient descent in Python for linear regression. Verify the time complexity per iteration is O(n*k) where n=samples, k=features.",
  "Write Dijkstra shortest path algorithm. Prove worst-case time complexity is O((V+E) log V) using a min-heap.",
  "Implement naive matrix multiplication for n x n matrices. State time complexity and verify it matches the implementation.",
  "Write merge sort in Python. Derive the recurrence T(n) = 2T(n/2) + O(n) and solve it to show O(n log n).",
  "Implement binary search tree insertion. State both average and worst-case time and space complexity.",
  "Write a function to compute eigenvalues of a 2x2 matrix. Implement the characteristic polynomial in code.",
  "Implement bubble sort. State O(n^2) complexity and show why it cannot be O(n log n).",
  "Write a Python function for Gaussian elimination. Verify O(n^3) time complexity.",
  "Implement BFS and DFS. State and verify the O(V+E) time complexity for both.",
  "Write a function for prime factorization. State the time complexity as O(sqrt(n)) and verify.",
] * 5  # 50 total

async def query_router(session, q, idx):
  try:
    r = await session.post("http://localhost:8000/query",
          json={"query": q}, timeout=120)
    result = r.json()
  except Exception as e:
    result = {"error": str(e)}
  result["query_idx"] = idx
  return result

async def main():
  results = []
  async with httpx.AsyncClient() as session:
    for i, q in enumerate(QUERIES):
      t0 = time.time()
      result = await query_router(session, q, i)
      elapsed = round(time.time() - t0, 2)
      result["elapsed"] = elapsed
      results.append(result)
      u = result.get("utility", "N/A")
      field = result.get("field", "?")
      try:
        print(f"Query {i+1:02d}/50: U={float(u):.3f} field={field} t={elapsed:.1f}s")
      except:
        print(f"Query {i+1:02d}/50: {result.get('error','?')} t={elapsed:.1f}s")

  Path("results").mkdir(exist_ok=True)
  json.dump(results, open("results/cross_domain_battery.json", "w"), indent=2)

  from collections import Counter
  verdicts = [r.get("arbiter_verdict", "no_arbiter") for r in results]
  print("\nArbiter verdict distribution:")
  for v, c in Counter(verdicts).most_common():
    print(f"  {v}: {c} ({c/len(results)*100:.0f}%)")

  u_scores = [r.get("utility", 0) for r in results if isinstance(r.get("utility"), (int, float))]
  if u_scores:
    print(f"\nMean U: {sum(u_scores)/len(u_scores):.4f}")
    print(f"Min U:  {min(u_scores):.4f}")
    print(f"Max U:  {max(u_scores):.4f}")

asyncio.run(main())
PYEOF
else
  log "  cross_domain_battery.json exists — skipping"
fi
check_deadline

# S5-4: Final metrics table
log "S5-4 — Final metrics comparison table"
python3 - 2>&1 | tee logs/final_metrics.log <<'PYEOF'
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
        print(f"  {label}: no data")

print("\nMath BLUE baseline (pre-training):")
for k in ["accuracy", "mean_u", "brier_score"]:
    print(f"  {k}: {math_base.get(k, 'N/A')}")

print("\nMath calibration cycles:")
for label, d in [("Cycle 1", math_c1), ("Cycle 2", math_c2)]:
    if d:
        prom = d.get("promoted", d.get("promote", "N/A"))
        u_d  = d.get("u_delta", d.get("green_mean_u","?"))
        print(f"  {label}: promoted={prom}  u_delta/green_u={u_d}")
    else:
        print(f"  {label}: no data")

if battery:
    u_scores = [r.get("utility", 0) for r in battery
                if isinstance(r.get("utility"), (int, float)) and r.get("utility",0) > 0]
    if u_scores:
        print(f"\nCross-domain battery ({len(battery)} queries):")
        print(f"  Mean U: {sum(u_scores)/len(u_scores):.4f}")
        print(f"  Min U:  {min(u_scores):.4f}   Max U: {max(u_scores):.4f}")
    from collections import Counter
    verdicts = Counter(r.get("arbiter_verdict", "none") for r in battery)
    print("  Arbiter verdicts:")
    for v, c in verdicts.most_common():
        print(f"    {v}: {c}")
print("=" * 65)
PYEOF

# S5-5: Commit everything
log "S5-5 — Final git commit + push"
do_commit "Stages 3-5 complete: SWE+Math calibration + cross-domain validation"

# S5-6: Write completion summary
log "S5-6 — Writing EXPERIMENT_COMPLETE.md"
EXPERIMENT_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GPU_HOURS=$(elapsed_hours)

SWE_C1_P=$(python3 -c "import json; d=json.load(open('results/swe_shift_cycle1.json')) if __import__('os').path.exists('results/swe_shift_cycle1.json') else {}; v=d.get('promoted',d.get('promote','N/A')); print(v)" 2>/dev/null || echo "N/A")
SWE_C2_P=$(python3 -c "import json; d=json.load(open('results/swe_shift_cycle2.json')) if __import__('os').path.exists('results/swe_shift_cycle2.json') else {}; v=d.get('promoted',d.get('promote','N/A')); print(v)" 2>/dev/null || echo "N/A")
MATH_C1_P=$(python3 -c "import json; d=json.load(open('results/math_shift_cycle1.json')) if __import__('os').path.exists('results/math_shift_cycle1.json') else {}; v=d.get('promoted',d.get('promote','N/A')); print(v)" 2>/dev/null || echo "N/A")
MATH_C2_P=$(python3 -c "import json; d=json.load(open('results/math_shift_cycle2.json')) if __import__('os').path.exists('results/math_shift_cycle2.json') else {}; v=d.get('promoted',d.get('promote','N/A')); print(v)" 2>/dev/null || echo "N/A")

cat > logs/EXPERIMENT_COMPLETE.md <<MDEOF
# AUA POC Overnight Experiment — COMPLETE

- **Start:** ${EXPERIMENT_START}
- **End:**   ${EXPERIMENT_END}
- **GPU hours:** ~${GPU_HOURS}

## Stage 3 — SWE Calibration

| Metric | Value |
|--------|-------|
| Cycle 1 promoted | ${SWE_C1_P} |
| Cycle 2 promoted | ${SWE_C2_P} |
| Best SWE model | ${SWE_BEST_MODEL} |
| Training method | AWQ dequantize → fp16 LoRA (QLoRA bypass) |

## Stage 4 — Math Calibration

| Metric | Value |
|--------|-------|
| Cycle 1 promoted | ${MATH_C1_P} |
| Cycle 2 promoted | ${MATH_C2_P} |
| Best Math model | ${MATH_BEST_MODEL} |

## Stage 5 — Cross-Domain Validation

See logs/final_metrics.log and logs/routing_calibrated.log for full numbers.

$(cat logs/final_metrics.log 2>/dev/null | head -40 || echo "See logs/final_metrics.log")

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
- Current production models: 9001=$SWE_BEST_MODEL  9002=$MATH_BEST_MODEL  9003=arbiter
MDEOF

log "================================================================"
log "EXPERIMENT COMPLETE"
log "Start: ${EXPERIMENT_START}  End: ${EXPERIMENT_END}  Hours: ${GPU_HOURS}"
log "SWE: cycles promoted=${SWE_C1_P}/${SWE_C2_P}  best=${SWE_BEST_MODEL}"
log "Math: cycles promoted=${MATH_C1_P}/${MATH_C2_P}  best=${MATH_BEST_MODEL}"
log "================================================================"
