# Adaptive Utility Agent — MVP

A wrapper around frontier language models that governs behavior through a
mathematically grounded utility function. The core idea is a utility function that the agent actively maximizes, composed of measurable, updatable components. The agent does not just answer questions — it tracks how well it answers them, where its knowledge is contradictory or thin, and where the highest leverage for improvement lies.

## Core Idea

Instead of using an LLM as a static oracle, this agent tracks:
- **Efficacy (E)**: how well it performs vs. human baseline
- **Confidence (C)**: internal consistency score, penalized by detected contradictions
- **Curiosity (K)**: pull toward high-upside unexplored domains

```
U = w_e(field) × E + w_c(field) × C + w_k(field) × K
subject to: C ≥ C_min(field), E ≥ E_min(field)
```

Field weights and minimum bounds are derived from existing societal standards
(medical licensing, aviation certification, engineering requirements).

## Project Structure

```
agent/
├── config.py                # Field weights, bounds, penalty multipliers
├── field_classifier.py      # Classifies task into domain distribution
├── contradiction_detector.py # Detects logical/math/cross-session contradictions
├── utility_scorer.py        # Computes E, C, K, and final U
├── personality_manager.py   # Trait weights, situational activation, evolution
├── agent.py                 # Main UtilityAgent class
└── harness.py               # MVP test harness (LeetCode-style problems)

whitepaper.md                # Full theoretical writeup
```

## Quick Start

```bash
pip install httpx
cd agent
python harness.py
```

## What the Harness Does

Runs 5 LeetCode-style problems through the agent with varying difficulty
and novelty. After each problem:
1. Classifies the field → loads weights and bounds
2. Builds system prompt with personality traits
3. Calls Claude → gets solution
4. Runs contradiction detection (syntax, logical, complexity, cross-session)
5. Scores U = w_e×E + w_c×C + w_k×K
6. Logs all components

After every 3 interactions: personality evolution step runs and adjusts
trait weights based on utility trend and contradiction rate.

## Key Design Decisions

- **Wrapper not replacement**: builds on Claude, doesn't reinvent LLMs
- **Societal standards as bounds**: C_min and E_min derived from real licensing requirements
- **Conservative under ambiguity**: blended field bounds are always >= component bounds
- **Abstain rather than fail**: agent refuses to act when confidence is below domain minimum
- **Personality evolves**: traits shift based on accumulated performance data

## Roadmap

See whitepaper.md §8 for the full roadmap from MVP → multi-domain STEM →
creative fields → fine-tuning feedback loop.
