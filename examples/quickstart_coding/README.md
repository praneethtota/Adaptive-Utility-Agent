# AUA Quickstart — Coding Preset

The fastest path to a running AUA system: two specialists (swe + math), Ollama backend, macbook tier.

## Setup

```bash
cd examples/quickstart_coding
pyenv local 3.11.10
pip install "adaptive-utility-agent[dev]"

aua init . --preset coding --tier macbook
aua doctor --strict
aua serve --dry-run
aua serve    # Terminal 1

# Terminal 2:
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Write binary search in Python. State time complexity."}' \
  | python3 -m json.tool
```

## Expected output

```json
{
  "routing_mode": "single",
  "primary_domain": "software_engineering",
  "u_score": 0.48,
  "latency_ms": 2800
}
```
