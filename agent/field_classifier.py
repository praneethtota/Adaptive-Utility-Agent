"""
Field classifier: determines domain distribution of a given task.
Uses a lightweight Claude call to classify before the main call.
"""

import json
import re
import httpx
from typing import Dict


FIELD_CLASSIFIER_PROMPT = """You are a domain classifier. Given a task or question, 
return a JSON object with the probability that it belongs to each field.
Probabilities must sum to 1.0. Only include fields with probability > 0.05.

Available fields:
- surgery
- aviation  
- law
- structural_engineering
- software_engineering
- stem_research
- education
- art
- creative_writing
- general

Return ONLY valid JSON, no explanation. Example:
{"software_engineering": 0.85, "stem_research": 0.15}
"""


async def classify_field(task: str) -> Dict[str, float]:
    """
    Classify a task into a field distribution.
    Falls back to {"general": 1.0} on any error.

    Args:
        task: the task or question to classify

    Returns:
        dict of {field_name: probability}
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-haiku-4-5-20251001",  # fast + cheap for classification
                    "max_tokens": 200,
                    "system": FIELD_CLASSIFIER_PROMPT,
                    "messages": [{"role": "user", "content": task}]
                },
                timeout=10.0
            )
            data = response.json()
            raw = data["content"][0]["text"].strip()

            # Strip markdown fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            distribution = json.loads(raw)

            # Validate and normalize
            total = sum(distribution.values())
            if total == 0:
                return {"general": 1.0}

            return {k: v / total for k, v in distribution.items()}

    except Exception as e:
        print(f"[FieldClassifier] Error: {e}, defaulting to general")
        return {"general": 1.0}
