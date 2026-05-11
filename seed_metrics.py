#!/usr/bin/env python3
"""Send a burst of queries to generate live Grafana data."""
import json, time, urllib.request

ROUTER = "http://localhost:8000"
QUERIES = [
    "Write binary search in Python. State the time complexity.",
    "What is the derivative of x^3 + 2x^2 - 5?",
    "Explain the Kalman filter and when to use it.",
    "What are the SOLID principles in software engineering?",
    "Write a merge sort implementation in Python.",
    "What is the time complexity of Dijkstra's algorithm?",
    "Explain gradient descent in machine learning.",
    "Write a function to check if a binary tree is balanced.",
    "What is a P-NP problem? Give an example.",
    "Implement a LRU cache in Python.",
]

def check_health():
    try:
        with urllib.request.urlopen(f"{ROUTER}/health/live", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

def send_query(q):
    req = urllib.request.Request(
        f"{ROUTER}/query",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())

if __name__ == "__main__":
    if not check_health():
        print("✗ Router not reachable. Run 'aua serve' first.")
        raise SystemExit(1)
    print(f"✓ Router up. Sending {len(QUERIES)} queries...\n")
    for i, q in enumerate(QUERIES, 1):
        try:
            r = send_query(q)
            print(f"  [{i:2d}] {r.get('primary_domain','?'):25s} U={r.get('u_score',0):.3f}  {r.get('latency_ms',0):.0f}ms")
        except Exception as e:
            print(f"  [{i:2d}] ERROR: {e}")
        time.sleep(0.5)
    print("\n✓ Done. Open http://localhost:3000  (admin / aua-admin)")
