#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Adaptive-Utility-Agent/agent

TARGET=250
TEMPS=(0.6 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8)

paired_count() {
    local path="$1"
    python -c "
import json, os
if os.path.exists('$path'):
    d = json.load(open('$path'))
    e = d if isinstance(d, list) else d.get('pairs', d.get('entries', []))
    print(len([x for x in e if x.get('chosen') and x.get('rejected')]))
else:
    print(0)
"
}

progress_bar() {
    local val=$1 total=$2 width=25
    local filled=$(( val * width / total ))
    [ $filled -gt $width ] && filled=$width
    local empty=$(( width - filled ))
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty;  i++)); do bar+="░"; done
    local pct=$(( val * 100 / total ))
    printf "[%s] %3d/%d (%d%%)" "$bar" "$val" "$total" "$pct"
}

# ── Background status monitor (prints every 3 min) ───────────────────────────
status_monitor() {
    while true; do
        sleep 180
        local swe_p math_p
        swe_p=$(paired_count dpo_pairs/swe_accumulation.json)
        math_p=$(paired_count dpo_pairs/math_accumulation.json)
        local swe_bar math_bar
        swe_bar=$(progress_bar "$swe_p" "$TARGET")
        math_bar=$(progress_bar "$math_p" "$TARGET")
        echo ""
        echo "┌─ PROGRESS @ $(date '+%H:%M:%S') ────────────────────────────────┐"
        echo "│  SWE  $swe_bar  │"
        echo "│  MATH $math_bar  │"
        echo "└──────────────────────────────────────────────────────────┘"
    done
}

status_monitor &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null; echo ''; echo 'Monitor stopped.'" EXIT

# ── SWE loop ──────────────────────────────────────────────────────────────────
RUN=3
for TEMP in "${TEMPS[@]}"; do
    PAIRED=$(paired_count dpo_pairs/swe_accumulation.json)
    echo ""
    echo "=== SWE Run ${RUN} | temp=${TEMP} | paired: ${PAIRED}/${TARGET} ==="
    if [ "${PAIRED}" -ge "${TARGET}" ]; then
        echo "SWE target reached. Moving to Math."
        break
    fi
    PYTHONUNBUFFERED=1 python harness.py \
        --endpoint http://localhost:9001 \
        --model swe \
        --cycles 3 \
        --queries seeded_contradictions.json \
        --export-dpo dpo_pairs/swe_accumulation.json \
        --field software_engineering \
        --temperature "${TEMP}" \
        --append \
        --out "harness_results_swe_run${RUN}.json" \
        2>&1 | tee "logs/swe_run${RUN}.log"
    RUN=$((RUN + 1))
done

SWE_FINAL=$(paired_count dpo_pairs/swe_accumulation.json)
echo ""
echo "=== SWE DONE: ${SWE_FINAL}/250 paired entries ==="

# ── Math loop ─────────────────────────────────────────────────────────────────
RUN=3
for TEMP in "${TEMPS[@]}"; do
    PAIRED=$(paired_count dpo_pairs/math_accumulation.json)
    echo ""
    echo "=== MATH Run ${RUN} | temp=${TEMP} | paired: ${PAIRED}/${TARGET} ==="
    if [ "${PAIRED}" -ge "${TARGET}" ]; then
        echo "Math target reached."
        break
    fi
    PYTHONUNBUFFERED=1 python harness.py \
        --endpoint http://localhost:9002 \
        --model math \
        --cycles 3 \
        --queries seeded_contradictions.json \
        --export-dpo dpo_pairs/math_accumulation.json \
        --field mathematics \
        --temperature "${TEMP}" \
        --append \
        --out "harness_results_math_run${RUN}.json" \
        2>&1 | tee "logs/math_run${RUN}.log"
    RUN=$((RUN + 1))
done

MATH_FINAL=$(paired_count dpo_pairs/math_accumulation.json)
echo ""
echo "=== MATH DONE: ${MATH_FINAL}/250 paired entries ==="
echo ""
echo "══════════════════════════════════════════"
echo "  ALL DONE — SWE: ${SWE_FINAL}  MATH: ${MATH_FINAL}"
echo "══════════════════════════════════════════"
