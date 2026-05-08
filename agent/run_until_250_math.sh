#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Adaptive-Utility-Agent/agent

TARGET=250
RUN=3
TEMPS=(0.6 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8)

for TEMP in "${TEMPS[@]}"; do
    PAIRED=$(python -c "
import json, os
path = 'dpo_pairs/math_accumulation.json'
if os.path.exists(path):
    d = json.load(open(path))
    e = d if isinstance(d, list) else d.get('pairs', d.get('entries', []))
    print(len([x for x in e if x.get('chosen') and x.get('rejected')]))
else:
    print(0)
")
    echo ""
    echo "=== MATH Run ${RUN} | temp=${TEMP} | paired so far: ${PAIRED}/${TARGET} ==="
    if [ "${PAIRED}" -ge "${TARGET}" ]; then
        echo "Target reached. Stopping."
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

FINAL=$(python -c "
import json, os
path = 'dpo_pairs/math_accumulation.json'
d = json.load(open(path))
e = d if isinstance(d, list) else d.get('pairs', d.get('entries', []))
print(len([x for x in e if x.get('chosen') and x.get('rejected')]))
")
echo ""
echo "=== MATH LOOP DONE: ${FINAL} paired entries ==="
