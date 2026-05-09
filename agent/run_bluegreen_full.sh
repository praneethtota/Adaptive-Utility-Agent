#!/usr/bin/env bash
# =============================================================================
# run_bluegreen_full.sh — Complete §10.7 Blue-Green Deployment Cycle
# =============================================================================
# Implements the full whitepaper lifecycle for BOTH SWE and Math, sequentially.
#
# VRAM budget (RTX 4090, 24564 MiB total):
#   Per-domain pair:
#     Blue AWQ  @ 0.20 util → reserves 4913 MiB  (model ~3338 MiB, KV ~1575 MiB)
#     Green fp16 @ 0.70 util → reserves 17195 MiB (model ~13351 MiB, KV ~3843 MiB)
#     Total: 22108 MiB = 90.1% — fits comfortably
#   Running both domain pairs simultaneously (4 servers): ~30070 MiB → OVERFLOW.
#   Therefore domains run SEQUENTIALLY — each cycle starts/stops its own servers.
#
# Flow:
#   STEP 1 : Free VRAM
#   STEP 2 : Train SWE  seed LoRA  (30 gold pairs → swe_green_v1 adapter)
#   STEP 3 : Merge SWE  seed       (swe_green_v1 → swe_green_v1_merged)
#   STEP 3b: SWE organic accumulation — start :9011 fp16 server, run harness
#            in --append loop until 250 organic DPO pairs, stop :9011
#   STEP 3c: SWE v2 retrain + merge  (swe_green_v1_merged + 250 organic
#            pairs → swe_green_v2 adapter → swe_green_v2_merged)
#   STEP 4 : Train Math seed LoRA  (22 gold pairs → math_green_v1 adapter)
#   STEP 5 : Merge Math seed       (math_green_v1 → math_green_v1_merged)
#   STEP 5b: Math organic accumulation — same pattern as 3b, port :9012
#   STEP 5c: Math v2 retrain + merge  (→ math_green_v2_merged)
#   STEP 6 : SWE  blue-green cycle (9001 AWQ blue + 9011 fp16 green → phases 2-5)
#   STEP 7 : Math blue-green cycle (9002 AWQ blue + 9012 fp16 green → phases 2-5)
#   STEP 8 : Summary + git commit
#
# Usage:
#   nohup bash run_bluegreen_full.sh > logs/bluegreen_tty.log 2>&1 &
# =============================================================================
set -uo pipefail
cd /workspace/Adaptive-Utility-Agent/agent

LOG=logs/bluegreen_full.log
mkdir -p logs results models

log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
warn() { log "WARN: $*"; }

# ── GPU helper ────────────────────────────────────────────────────────────────
vram_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo "?"; }
vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "?"; }

# ── Port / process management ─────────────────────────────────────────────────

kill_port() {
  local port=$1
  local pids; pids=$(lsof -ti :"$port" 2>/dev/null || true)
  [ -n "$pids" ] && { log "  killing :$port (pids $pids)"; kill $pids 2>/dev/null || true; sleep 4; }
}

kill_all_inference() {
  log "  Stopping all inference servers to free VRAM..."
  for port in 9001 9002 9003 9011 9012 8000; do kill_port "$port"; done
  pkill -f "router.py" 2>/dev/null || true
  sleep 8
  log "  VRAM free: $(vram_free)"
}

# ── Server management ─────────────────────────────────────────────────────────

wait_server() {
  local port=$1 label=${2:-port$1} tries=0
  log "  waiting for $label on :$port ..."
  until curl -s --max-time 5 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q '"id"'; do
    tries=$((tries+1))
    [ $tries -ge 90 ] && { warn "$label on :$port not up after 15 min"; return 1; }
    sleep 10
  done
  local mid; mid=$(curl -s "http://localhost:${port}/v1/models" | \
    python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["data"][0]["id"])' 2>/dev/null || echo "?")
  log "  $label up — model_id=$mid"
}

check_server() { curl -s --max-time 3 "http://localhost:$1/v1/models" 2>/dev/null | grep -q '"id"'; }

start_awq_server() {
  local port=$1 model=$2 name=$3 mem=$4
  if check_server "$port"; then log "  $name :$port already up"; return 0; fi
  log "  Starting $name (AWQ) on :$port mem=$mem | VRAM used: $(vram_used)"
  local abs_model; abs_model=$(realpath "$model" 2>/dev/null || echo "$model")
  python -m vllm.entrypoints.openai.api_server \
    --model "$abs_model" --port "$port" \
    --quantization awq --max-model-len 2048 \
    --served-model-name "$name" \
    --gpu-memory-utilization "$mem" &
  wait_server "$port" "$name"
}

start_fp16_server() {
  local port=$1 name=$3 mem=$4 maxlen=${5:-2048}
  local model; model=$(realpath "$2" 2>/dev/null || echo "$2")
  if check_server "$port"; then log "  $name :$port already up"; return 0; fi
  log "  Starting $name (fp16) on :$port mem=$mem max-model-len=$maxlen | VRAM used: $(vram_used)"
  python -m vllm.entrypoints.openai.api_server \
    --model "$model" --port "$port" \
    --max-model-len "$maxlen" \
    --served-model-name "$name" \
    --gpu-memory-utilization "$mem" \
    --dtype float16 &
  wait_server "$port" "$name"
}

# ── Training helpers ──────────────────────────────────────────────────────────

merge_lora() {
  local base=$1 adapter=$2 output=$3
  if [ -f "${output}/config.json" ] && [ -f "${output}/model.safetensors" ]; then log "  ${output} already merged — skipping"; return 0; fi
  log "  Merging LoRA: $base + $adapter → $output | VRAM used: $(vram_used)"
  python3 - "$base" "$adapter" "$output" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, torch, torch.nn as nn
from pathlib import Path
from awq import AutoAWQForCausalLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json

base_path, adapter_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

cfg = json.load(open(Path(base_path) / "config.json"))
qcfg = cfg.get("quantization_config") or {}
is_awq = (str(qcfg.get("quant_type")  or "").lower() == "awq" or
          str(qcfg.get("quant_method") or "").lower() == "awq" or
          qcfg.get("bits", 0) == 4)

if is_awq:
    print("AWQ base — loading via autoawq and dequantizing per-layer...")
    awq_model = AutoAWQForCausalLM.from_quantized(
        base_path, fuse_layers=False, trust_remote_code=True,
        safetensors=True, device_map="auto")
    hf_model = awq_model.model
    replaced = 0
    for parent_name, parent_module in list(hf_model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            cls = type(child_module).__name__
            if "WQLinear" in cls:
                try:
                    w = child_module.dequantize()   # returns (in_features, out_features)
                except AttributeError:
                    from awq.utils.packing_utils import dequantize_gemm
                    w = dequantize_gemm(child_module.qweight, child_module.qzeros,
                                        child_module.scales, child_module.w_bit,
                                        child_module.group_size)
                dev = next(child_module.parameters(), child_module.qweight).device
                new_lin = nn.Linear(
                    child_module.in_features, child_module.out_features,
                    bias=child_module.bias is not None,
                    dtype=torch.float16, device=dev)
                new_lin.weight = nn.Parameter(w.T.to(dev))  # nn.Linear expects (out, in)
                if child_module.bias is not None:
                    new_lin.bias = nn.Parameter(child_module.bias.to(dev))
                setattr(parent_module, child_name, new_lin)
                replaced += 1
    print(f"Dequantized {replaced} WQLinear layers to fp16")
    hf_model.config.quantization_config = None
    model = hf_model
else:
    print("fp16 base — loading directly...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto")

print("Applying and merging LoRA adapter...")
peft_model = PeftModel.from_pretrained(model, adapter_path)
merged = peft_model.merge_and_unload()
# Move to CPU before serializing: merge_and_unload leaves the full fp16 model
# on GPU (~13 GiB). save_pretrained needs additional headroom for serialization
# which causes OOM on a 24 GiB GPU. CPU offload avoids this entirely.
import gc
print("Moving merged model to CPU for serialization...")
merged = merged.cpu()
torch.cuda.empty_cache()
gc.collect()
print("Saving merged model from CPU...")
Path(output_path).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(output_path, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(adapter_path)
tok.save_pretrained(output_path)
cfg_out = Path(output_path) / "config.json"
if cfg_out.exists():
    c = json.load(open(cfg_out))
    c.pop("quantization_config", None)
    # Remove AWQ-inherited max_seq_len so vLLM uses max_position_embeddings (32768).
    # Without this, vLLM caps context at 2048 and refuses --max-model-len 4096.
    c.pop("max_seq_len", None)
    json.dump(c, open(cfg_out,"w"), indent=2)
print(f"Merge complete → {output_path}")
PYEOF
}

train_lora() {
  local base=$1 pairs=$2 output=$3 field=$4
  if [ -f "${output}/training_log.json" ] && [ -f "${output}/adapter_model.safetensors" ]; then log "  ${output} already trained — skipping"; return 0; fi
  log "  Training LoRA: base=$base field=$field output=$output | VRAM used: $(vram_used)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_lora.py \
    --base-model "$base" \
    --dpo-pairs  "$pairs" \
    --output     "$output" \
    --field      "$field" \
    --epochs 5 --lora-r 16 --lora-alpha 32 \
    --batch-size 1 \
    --apply-field-weights \
    2>&1 | tee -a "$LOG" || warn "train_lora.py exited non-zero"
  # Remove intermediate checkpoints and ref model — only final adapter weights needed.
  rm -rf "${output}"/checkpoint-* "${output}/ref" 2>/dev/null || true
  local freed; freed=$(du -sh "${output}" 2>/dev/null | cut -f1)
  log "  Checkpoints/ref purged from ${output} (kept: ${freed})"
}

# ── DPO pair counter ──────────────────────────────────────────────────────────
# Prints count of entries that have both 'chosen' and 'rejected' fields.

_count_dpo_pairs() {
  python3 - "$1" <<'EOF' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    entries = d if isinstance(d, list) else d.get("pairs", d.get("entries", []))
    print(sum(1 for x in entries if x.get("chosen") and x.get("rejected")))
except:
    print(0)
EOF
}

# =============================================================================
# accumulate_organic_dpo — Harness loop to generate organic DPO pairs
# =============================================================================
# Runs harness.py against a LIVE server in --append mode, repeating until
# TARGET paired entries are in OUT_FILE or MAX_RUNS is exhausted.
#
# Args: DOMAIN  PORT  MODEL_NAME  QUERIES_FILE  OUT_FILE  FIELD  TARGET
#   DOMAIN       – label for logs (swe, math)
#   PORT         – port of the live inference server
#   MODEL_NAME   – must match the server's --served-model-name
#   QUERIES_FILE – 100-problem adversarial set (seeded_contradictions.json)
#   OUT_FILE     – accumulation file; each harness run appends to it
#   FIELD        – harness field (software_engineering, mathematics)
#   TARGET       – paired-entry goal (e.g. 250)
#
# VRAM contract: caller starts/stops the server.  Only ONE server should be
# live during accumulation (fp16 @ 0.80 → ~19651 MiB = 82% of 24564 MiB).
# =============================================================================
accumulate_organic_dpo() {
  local domain=$1 port=$2 model_name=$3 queries=$4 out_file=$5 field=$6 target=$7

  local current; current=$(_count_dpo_pairs "$out_file")
  if [ "$current" -ge "$target" ]; then
    log "  $domain organic DPO: $current/$target pairs already present — skipping"
    return 0
  fi

  log "──────────────────────────────────────────────"
  log "ORGANIC ACCUMULATION: domain=$domain  port=$port  target=$target"
  log "  model=$model_name  field=$field"
  log "  queries=$queries  output=$out_file"
  log "  starting pairs: $current"
  log "──────────────────────────────────────────────"
  mkdir -p dpo_pairs

  local run=0 max_runs=20
  while [ "$current" -lt "$target" ] && [ "$run" -lt "$max_runs" ]; do
    run=$((run+1))
    log "  Organic run $run/$max_runs — pairs so far: $current/$target | VRAM: $(vram_used)"

    python harness.py \
      --endpoint "http://localhost:${port}" \
      --model    "$model_name" \
      --cycles   3 \
      --queries  "$queries" \
      --export-dpo "$out_file" \
      --field    "$field" \
      --temperature 0.4 \
      --append \
      --out "logs/${domain}_organic_run${run}.json" \
      2>&1 | tee "logs/${domain}_organic_run${run}.log" \
      || warn "harness run $run exited non-zero (continuing)"

    current=$(_count_dpo_pairs "$out_file")
    log "  After run $run: $current/$target paired entries"
  done

  if [ "$current" -lt "$target" ]; then
    warn "$domain organic: $current/$target pairs after $max_runs runs — proceeding with what we have"
  else
    log "$domain organic accumulation COMPLETE: $current entries ≥ $target target"
  fi
}

# =============================================================================
# run_bg_cycle — Full §10.7 blue-green cycle for one domain
# =============================================================================
# Starts its own servers, runs canary → shift → promotion, then tears down.
# Args: DOMAIN BLUE_PORT GREEN_PORT BLUE_MODEL GREEN_MODEL FIELD
# VRAM contract: caller must have freed VRAM before calling.
#   Blue  (AWQ)  0.20 util → ~4913 MiB reserved
#   Green (fp16) 0.70 util → ~17195 MiB reserved
#   Total:               ~22108 MiB = 90.1% of 24564 MiB
# =============================================================================
run_bg_cycle() {
  local domain=$1 blue_port=$2 green_port=$3
  local blue_model=$4 green_model=$5 field=$6
  local blue_ep="http://localhost:${blue_port}"
  local green_ep="http://localhost:${green_port}"

  log "──────────────────────────────────────────────"
  log "BLUE-GREEN CYCLE: domain=$domain  BLUE=:$blue_port  GREEN=:$green_port"
  log "  Blue  model: $blue_model"
  log "  Green model: $green_model"
  log "  VRAM free before start: $(vram_free)"
  log "──────────────────────────────────────────────"

  # ── Start servers (sequential — green only after blue is healthy) ──────────
  # Blue: AWQ at 0.30 reserves 7369 MiB (model 5.2GiB + CUDA graphs 0.79GiB = 5.99GiB actual,
  #   KV ~1.34GiB). vLLM v0.20.1 CUDA-graph profiling adds ~0.79GiB overhead vs earlier estimates.
  start_awq_server "$blue_port" "$blue_model" "$domain" 0.30 || {
    warn "BLUE server :${blue_port} failed to start — aborting ${domain} cycle"
    return 1
  }
  log "  VRAM after blue start: $(vram_used)"

  # Green: fp16 at 0.65 reserves 15290 MiB (model ~13.25GiB + CUDA graphs 0.55GiB, KV ~1.49GiB)
  # Combined with blue 0.30: 0.95 → 22759 MiB = 92.7% — fits within 24564 MiB (15.46 GiB free after blue)
  start_fp16_server "$green_port" "$green_model" "${domain}_green_v1" 0.65 || {
    warn "GREEN server :${green_port} failed to start — aborting ${domain} cycle"
    kill_port "$blue_port"
    return 1
  }
  log "  VRAM after both servers: $(vram_used)"

  # ── Phase 2: Canary (50 queries, 5% → GREEN) ──────────────────────────────
  local canary_out="results/${domain}_bg_canary.json"
  if [ ! -f "$canary_out" ]; then
    log "Phase 2 — Canary: 5% GREEN / 95% BLUE (50 queries)"
    python canary.py \
      --blue  "$blue_ep" --green "$green_ep" \
      --traffic-green 0.05 --n 50 \
      --output "$canary_out" \
      2>&1 | tee "logs/${domain}_bg_canary.log" || warn "canary.py exited non-zero"
    python canary.py --check --results "$canary_out" \
      2>&1 | tee -a "logs/${domain}_bg_canary.log" || true
  else
    log "  ${canary_out} exists — skipping canary"
  fi

  # ── Phase 3+4: Gradual shift → promotion threshold ────────────────────────
  local shift_out="results/${domain}_bg_shift.json"
  if [ ! -f "$shift_out" ]; then
    log "Phase 3+4 — Gradual shift + promotion check (150 queries)"
    python check_promotion.py --shift \
      --blue  "$blue_ep" --green "$green_ep" \
      --field "$field" --n 150 \
      --output "$shift_out" \
      2>&1 | tee "logs/${domain}_bg_shift.log" || warn "check_promotion.py shift exited non-zero"
    python check_promotion.py --check \
      --results "$shift_out" --field "$field" \
      2>&1 | tee -a "logs/${domain}_bg_shift.log" || true
  else
    log "  ${shift_out} exists — skipping shift"
  fi

  # ── Phase 5: Promotion decision ───────────────────────────────────────────
  local promoted=no
  python3 - "$shift_out" "$domain" <<'PYEOF' 2>/dev/null && promoted=yes
import json, sys
p = sys.argv[1]; domain = sys.argv[2]
try:
    d = json.load(open(p))
    v = d.get("promoted", d.get("promote", False))
    sys.exit(0 if (v is True or str(v).lower() in ("true","yes","promoted")) else 1)
except:
    sys.exit(1)
PYEOF

  log "Phase 5 — Promotion decision for $domain: PROMOTED=$promoted"

  if [ "$promoted" = "yes" ]; then
    log "  *** GREEN PROMOTED — Blue retires, Green becomes new Blue ***"

    # Retire old blue server
    log "  Retiring BLUE on :${blue_port}"
    kill_port "$blue_port"
    sleep 6

    # Kill green server so we can restart it on the blue port
    log "  Stopping GREEN on :${green_port} before re-launching on blue port"
    kill_port "$green_port"
    sleep 6
    log "  VRAM after killing both: $(vram_free)"

    # Green model restarts on blue port as the new blue (0.90 util — only server)
    log "  Starting promoted GREEN on BLUE port :${blue_port} (this is now the new BLUE)"
    start_fp16_server "$blue_port" "$green_model" "$domain" 0.90 || \
      warn "Promoted server :${blue_port} failed to restart — check $green_model"

    # Write promotion record
    python3 - <<PYEOF 2>/dev/null | tee -a "$LOG"
import json, time
from pathlib import Path
try:
    shift = json.load(open("results/${domain}_bg_shift.json"))
except Exception:
    shift = {}
rec = {
    "domain": "${domain}",
    "event": "promotion_complete",
    "old_blue": "${blue_model}",
    "new_blue": "${green_model}",
    "green_port_retired": ${green_port},
    "new_blue_port": ${blue_port},
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase": "Phase 5 complete — retirement done",
    "note": "GREEN promoted per §10.7: traffic_green >= 1-delta, T_sustained met. BLUE retired.",
    "final_blue_mean_u":  shift.get("blue_mean_u",  shift.get("final_blue_u")),
    "final_green_mean_u": shift.get("green_mean_u", shift.get("final_green_u")),
    "u_delta":            shift.get("u_delta"),
    "t_sustained":        shift.get("t_sustained"),
    "promoted":           shift.get("promoted", shift.get("promote")),
}
Path("results").mkdir(exist_ok=True)
json.dump(rec, open("results/${domain}_bg_promotion_record.json", "w"), indent=2)
print(json.dumps(rec, indent=2))
PYEOF

    log "  ${domain}: promotion complete — GREEN is new BLUE on :${blue_port}"
    log "  VRAM after promotion: $(vram_used)"
    return 0

  else
    log "  ${domain}: GREEN not promoted — BLUE unchanged"
    python3 -c "
import json, sys
try:
    d = json.load(open('results/${domain}_bg_shift.json'))
    bu = d.get('blue_mean_u', d.get('final_blue_u', 'N/A'))
    gu = d.get('green_mean_u', d.get('final_green_u', 'N/A'))
    ud = d.get('u_delta', 'N/A')
    ts = d.get('t_sustained', 'N/A')
    pm = d.get('promoted', 'N/A')
    print(f'  Blue mean U:  {bu}')
    print(f'  Green mean U: {gu}')
    print(f'  U delta:      {ud}')
    print(f'  T sustained:  {ts}')
    print(f'  Promoted:     {pm}')
except Exception as e:
    print(f'  (no shift data: {e})')
" 2>/dev/null | tee -a "$LOG"

    # Tear down both servers before next domain
    log "  Tearing down :${blue_port} and :${green_port} before next domain"
    kill_port "$green_port"
    kill_port "$blue_port"
    sleep 8
    log "  VRAM free after teardown: $(vram_free)"
    return 1
  fi
}

# =============================================================================
# MAIN
# =============================================================================

log "================================================================"
log "BLUE-GREEN FULL CYCLE  (§10.7 whitepaper-compliant)"
log "================================================================"
log "  Run date:   $(date -u)"
log "  VRAM total: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo '?')"
log "  VRAM free:  $(vram_free)"
log "================================================================"

# State variables — updated by 3b/3c and 5b/5c; promotion steps read them.
SWE_ORGANIC_SKIP=no
MATH_ORGANIC_SKIP=no
SWE_GREEN_MODEL=./models/swe_green_v1_merged   # upgraded to v2 after 3c
MATH_GREEN_MODEL=./models/math_green_v1_merged  # upgraded to v2 after 5c

# Ensure libnvJitLink symlink
ln -sf /usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib/libnvJitLink.so.13 \
       /usr/local/lib/libnvJitLink.so.13 2>/dev/null || true
ldconfig 2>/dev/null || true

# ── STEP 1: Free all VRAM for offline training/merge ─────────────────────────
log "STEP 1 — Free all VRAM (offline training requires full GPU)"
kill_all_inference

# ── STEP 2: SWE training (clean DPO pairs — corrupted pairs replaced) ────────
log "STEP 2 — Train SWE GREEN v1 (clean gold-standard DPO pairs)"
train_lora ./models/swe dpo_pairs/swe_accumulation.json ./models/swe_green_v1 software_engineering
log "  SWE train done. VRAM free: $(vram_free)"

# ── STEP 3: SWE merge ────────────────────────────────────────────────────────
log "STEP 3 — Merge SWE GREEN v1"
merge_lora ./models/swe ./models/swe_green_v1 ./models/swe_green_v1_merged
log "  SWE merge done. VRAM free: $(vram_free)"

# ── STEP 3b: SWE organic accumulation ────────────────────────────────────────
# Start the seed GREEN v1 server alone (fp16, 0.80 util → ~19651 MiB = 82%).
# Run harness in --append loop until 250 organic DPO pairs are in the file.
# seeded_contradictions.json: 100 adversarial complexity questions (Type A + B).
log "STEP 3b — SWE organic DPO accumulation (target: 250 pairs)"
_SWE_PRE=$(_count_dpo_pairs dpo_pairs/swe_organic.json)
if [ "$_SWE_PRE" -ge 250 ]; then
  log "  swe organic DPO: $_SWE_PRE/250 pairs already present — skipping server start"
  SWE_ORGANIC_SKIP=yes
elif [ -f ./models/swe_green_v1_merged/config.json ] && [ -f ./models/swe_green_v1_merged/model.safetensors ]; then
  start_fp16_server 9011 ./models/swe_green_v1_merged swe 0.80 4096 || {
    warn "STEP 3b — seed GREEN :9011 failed to start — skipping organic accumulation"
    SWE_ORGANIC_SKIP=yes
  }
  if [ "$SWE_ORGANIC_SKIP" != "yes" ]; then
    accumulate_organic_dpo \
      swe 9011 swe seeded_contradictions.json \
      dpo_pairs/swe_organic.json software_engineering 250
    log "  Stopping seed GREEN :9011 before offline retraining..."
    kill_port 9011
    sleep 8
    log "  VRAM free after :9011 shutdown: $(vram_free)"
  fi
else
  warn "STEP 3b — swe_green_v1_merged missing or incomplete, skipping SWE organic accumulation"
  SWE_ORGANIC_SKIP=yes
fi

# ── STEP 3c: SWE v2 retrain on organic pairs + merge ─────────────────────────
# Base model: fp16 swe_green_v1_merged (train_lora fallback handles non-AWQ base).
# LoRA adapter: swe_green_v2   Merged: swe_green_v2_merged
# Falls back to v1_merged for promotion if accumulation produced < 10 pairs.
log "STEP 3c — Retrain SWE GREEN v2 on organic pairs + merge"
SWE_ORGANIC_COUNT=$(_count_dpo_pairs dpo_pairs/swe_organic.json)
log "  Organic pairs available: $SWE_ORGANIC_COUNT"
if [ "$SWE_ORGANIC_COUNT" -ge 10 ]; then
  train_lora ./models/swe_green_v1_merged \
             dpo_pairs/swe_organic.json \
             ./models/swe_green_v2 \
             software_engineering
  log "  SWE v2 train done. VRAM free: $(vram_free)"
  merge_lora ./models/swe_green_v1_merged \
             ./models/swe_green_v2 \
             ./models/swe_green_v2_merged
  log "  SWE v2 merge done. VRAM free: $(vram_free)"
else
  warn "STEP 3c — skipping v2 retrain (skip=$SWE_ORGANIC_SKIP pairs=$SWE_ORGANIC_COUNT)"
fi
[ -f ./models/swe_green_v2_merged/config.json ] && SWE_GREEN_MODEL=./models/swe_green_v2_merged
log "  SWE green model for promotion: $SWE_GREEN_MODEL"

# ── STEP 4: Math training (with WQLinear_GEMM.dequantize() fix active) ───────
log "STEP 4 — Train Math GREEN v1 (clean DPO pairs + WQLinear_GEMM fix)"
train_lora ./models/math dpo_pairs/math_accumulation.json ./models/math_green_v1 mathematics
log "  Math train done. VRAM free: $(vram_free)"

# ── STEP 5: Math merge ───────────────────────────────────────────────────────
log "STEP 5 — Merge Math GREEN v1"
merge_lora ./models/math ./models/math_green_v1 ./models/math_green_v1_merged
log "  Math merge done. VRAM free: $(vram_free)"

# ── STEP 5b: Math organic accumulation ───────────────────────────────────────
# Uses seeded_contradictions.json with --field mathematics. The adversarial
# complexity questions are valid for both SWE and math specialist models.
log "STEP 5b — Math organic DPO accumulation (target: 250 pairs)"
if [ -f ./models/math_green_v1_merged/config.json ]; then
  start_fp16_server 9012 ./models/math_green_v1_merged math 0.80 4096 || {
    warn "STEP 5b — seed GREEN :9012 failed to start — skipping organic accumulation"
    MATH_ORGANIC_SKIP=yes
  }
  if [ "$MATH_ORGANIC_SKIP" != "yes" ]; then
    accumulate_organic_dpo \
      math 9012 math seeded_contradictions.json \
      dpo_pairs/math_organic.json mathematics 250
    log "  Stopping seed GREEN :9012 before offline retraining..."
    kill_port 9012
    sleep 8
    log "  VRAM free after :9012 shutdown: $(vram_free)"
  fi
else
  warn "STEP 5b — math_green_v1_merged missing, skipping Math organic accumulation"
  MATH_ORGANIC_SKIP=yes
fi

# ── STEP 5c: Math v2 retrain on organic pairs + merge ────────────────────────
log "STEP 5c — Retrain Math GREEN v2 on organic pairs + merge"
MATH_ORGANIC_COUNT=$(_count_dpo_pairs dpo_pairs/math_organic.json)
log "  Organic pairs available: $MATH_ORGANIC_COUNT"
if [ "$MATH_ORGANIC_SKIP" != "yes" ] && [ "$MATH_ORGANIC_COUNT" -ge 10 ]; then
  train_lora ./models/math_green_v1_merged \
             dpo_pairs/math_organic.json \
             ./models/math_green_v2 \
             mathematics
  log "  Math v2 train done. VRAM free: $(vram_free)"
  merge_lora ./models/math_green_v1_merged \
             ./models/math_green_v2 \
             ./models/math_green_v2_merged
  log "  Math v2 merge done. VRAM free: $(vram_free)"
else
  warn "STEP 5c — skipping v2 retrain (skip=$MATH_ORGANIC_SKIP pairs=$MATH_ORGANIC_COUNT)"
fi
[ -f ./models/math_green_v2_merged/config.json ] && MATH_GREEN_MODEL=./models/math_green_v2_merged
log "  Math green model for promotion: $MATH_GREEN_MODEL"

# Verify promotion models exist before proceeding
for m in "$SWE_GREEN_MODEL" "$MATH_GREEN_MODEL"; do
  if [ ! -f "${m}/config.json" ]; then
    warn "MISSING merged model: $m — corresponding cycle will be skipped"
  fi
done
log "  SWE green for promotion:  $SWE_GREEN_MODEL"
log "  Math green for promotion: $MATH_GREEN_MODEL"

# ── STEP 6: SWE blue-green cycle ─────────────────────────────────────────────
# VRAM: Blue AWQ 0.20 (~4913 MiB) + Green fp16 0.70 (~17195 MiB) = ~22108 MiB (90.1%)
# Uses SWE_GREEN_MODEL: swe_green_v2_merged if organic retrain succeeded, else v1_merged.
log "STEP 6 — SWE blue-green cycle (§10.7)  green=$SWE_GREEN_MODEL"
SWE_PROMOTED=no
if [ -f "${SWE_GREEN_MODEL}/config.json" ]; then
  run_bg_cycle swe 9001 9011 ./models/swe "$SWE_GREEN_MODEL" software_engineering \
    && SWE_PROMOTED=yes
  log "STEP 6 done. SWE_PROMOTED=$SWE_PROMOTED. VRAM free: $(vram_free)"
else
  warn "STEP 6 — $SWE_GREEN_MODEL missing, skipping SWE cycle"
fi

# Ensure ports free before Math cycle
kill_port 9001
kill_port 9011
sleep 8
log "  VRAM before Math cycle: $(vram_free)"

# ── STEP 7: Math blue-green cycle ────────────────────────────────────────────
# VRAM: Blue AWQ 0.20 (~4913 MiB) + Green fp16 0.70 (~17195 MiB) = ~22108 MiB (90.1%)
# Uses MATH_GREEN_MODEL: math_green_v2_merged if organic retrain succeeded, else v1_merged.
log "STEP 7 — Math blue-green cycle (§10.7)  green=$MATH_GREEN_MODEL"
MATH_PROMOTED=no
if [ -f "${MATH_GREEN_MODEL}/config.json" ]; then
  run_bg_cycle math 9002 9012 ./models/math "$MATH_GREEN_MODEL" mathematics \
    && MATH_PROMOTED=yes
  log "STEP 7 done. MATH_PROMOTED=$MATH_PROMOTED. VRAM free: $(vram_free)"
else
  warn "STEP 7 — $MATH_GREEN_MODEL missing, skipping Math cycle"
fi

# Ensure all ports free after math cycle
kill_port 9002
kill_port 9012
sleep 8

# ── STEP 7: Final summary ─────────────────────────────────────────────────────
log "================================================================"
log "BLUE-GREEN FULL CYCLE COMPLETE"
log "================================================================"
log "  SWE  promoted: $SWE_PROMOTED"
log "  Math promoted: $MATH_PROMOTED"
log ""

python3 - <<'PYEOF' 2>/dev/null | tee -a "$LOG"
import json
from pathlib import Path

def load(p):
    try:    return json.load(open(p))
    except: return {}

print("=== BLUE-GREEN DEPLOYMENT RESULTS (§10.7) ===")
print()

for domain in ["swe", "math"]:
    canary = load(f"results/{domain}_bg_canary.json")
    shift  = load(f"results/{domain}_bg_shift.json")
    promo  = load(f"results/{domain}_bg_promotion_record.json")

    field = "software_engineering" if domain == "swe" else "mathematics"
    print(f"Domain: {domain.upper()}  ({field})")

    if canary:
        bu  = canary.get("blue_mean_u", "N/A")
        gu  = canary.get("green_mean_u", "N/A")
        ngc = canary.get("n_green_calls", 0)
        bu_str = f"{bu:.4f}" if isinstance(bu, float) else str(bu)
        gu_str = f"{gu:.4f}" if isinstance(gu, float) else str(gu)
        print(f"  Phase 2 Canary:   blue_U={bu_str}  green_U={gu_str}  green_calls={ngc}")

    if shift:
        bu  = shift.get("blue_mean_u",  shift.get("final_blue_u",  "N/A"))
        gu  = shift.get("green_mean_u", shift.get("final_green_u", "N/A"))
        ud  = shift.get("u_delta", "N/A")
        ts  = shift.get("t_sustained", "N/A")
        pm  = shift.get("promoted", shift.get("promote", "N/A"))
        bu_str = f"{bu:.4f}" if isinstance(bu, float) else str(bu)
        gu_str = f"{gu:.4f}" if isinstance(gu, float) else str(gu)
        ud_str = f"{ud:+.4f}" if isinstance(ud, float) else str(ud)
        print(f"  Phase 3+4 Shift:  blue_U={bu_str}  green_U={gu_str}  delta={ud_str}")
        print(f"                    T_sustained={ts}  promoted={pm}")

    if promo:
        print(f"  Phase 5 Retired:  old_blue={promo.get('old_blue','?')}")
        print(f"                    new_blue={promo.get('new_blue','?')}")
        print(f"                    new_blue_port={promo.get('new_blue_port','?')}")
    elif not promo and shift and shift.get("promoted", shift.get("promote")):
        print(f"  Phase 5:          [promotion record missing]")
    else:
        print(f"  Phase 5:          not promoted")

    print()
PYEOF

# ── git commit results ────────────────────────────────────────────────────────
cd /workspace/Adaptive-Utility-Agent
git add -A 2>/dev/null || true
git diff --cached --quiet || git commit -m "$(cat <<COMMITMSG
Blue-green full cycle (§10.7): SWE promoted=${SWE_PROMOTED}, Math promoted=${MATH_PROMOTED}

Phase 2 (canary 5%), Phase 3+4 (gradual shift 150q), Phase 5 (promotion/retirement).
Sequential execution: each domain runs its own 2-server pair (AWQ blue 0.20 + fp16 green 0.70).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
COMMITMSG
)" 2>&1 | tee -a agent/"$LOG" || true
git push 2>&1 | tee -a agent/"$LOG" || true
cd /workspace/Adaptive-Utility-Agent/agent

log "All done. Results: results/*_bg_*.json  Logs: logs/bluegreen_full.log"
