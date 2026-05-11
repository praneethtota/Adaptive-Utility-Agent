"""
aua/correction_loop.py — Online DPO correction loop.

CorrectionLoop drives the online learning cycle:
  1. Collect contradiction events from the Router's assertions store
  2. Format (chosen, rejected) pairs for DPO training
  3. Trigger LoRA fine-tuning on the affected specialist
  4. Evaluate the resulting checkpoint for promotion

Status: v0.6-alpha stub — interface defined, DPO pair collection is operational
via aua.router (POST /corrections). The training trigger and LoRA harness
will be added in v0.7 (roadmap #12 / #13).

Usage:
    from aua import CorrectionLoop

    loop = CorrectionLoop(config, router_url="http://localhost:8000")
    pairs = await loop.collect_pairs(min_confidence=0.8)
    summary = loop.export_pairs(pairs, output_dir="dpo_pairs/")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DPOPair:
    """A single (chosen, rejected) training pair for DPO fine-tuning."""

    prompt: str
    chosen: str  # correct output (verified by Arbiter or expert)
    rejected: str  # incorrect output (to be suppressed)
    domain: str
    confidence: float
    source: str = "arbiter"  # "arbiter" | "expert" | "correction"


@dataclass
class CollectionSummary:
    """Summary of a DPO pair collection run."""

    n_pairs: int = 0
    n_domains: int = 0
    domains: list[str] = field(default_factory=list)
    output_path: str = ""
    ready_for_training: bool = False  # True when LoRA harness is available (v0.7)


class CorrectionLoop:
    """
    Online DPO correction loop — closes the error→learning→deployment cycle.

    Workflow (full loop, v0.7+):
        1. collect_pairs()   — pull contradiction events from Router
        2. export_pairs()    — write JSONL for DPO trainer
        3. train()           — run LoRA fine-tune (roadmap #12)
        4. evaluate()        — run BlueGreenDeployment.evaluate() on checkpoint
        5. promote()         — swap model if ΔU ≥ threshold

    Current capabilities (v0.6-alpha):
        - collect_pairs() hits GET /corrections — operational
        - export_pairs() writes JSONL — operational
        - train() / promote() — stub, returns immediately

    Example:
        loop = CorrectionLoop(config, router_url="http://localhost:8000")
        pairs = await loop.collect_pairs()
        summary = loop.export_pairs(pairs, output_dir="dpo_pairs/")
        print(f"Exported {summary.n_pairs} pairs")
    """

    def __init__(
        self,
        config: Any,  # AUAConfig
        router_url: str = "http://localhost:8000",
        project_dir: str = ".",
    ) -> None:
        self._config = config
        self._router_url = router_url
        self._project_dir = project_dir

    async def collect_pairs(
        self,
        min_confidence: float = 0.7,
        domain: str | None = None,
        limit: int = 100,
    ) -> list[DPOPair]:
        """
        Collect DPO pairs from the Router's corrections store.

        Hits GET /corrections on the running router and returns pairs
        whose confidence exceeds min_confidence.

        Args:
            min_confidence: minimum correction confidence threshold
            domain:         filter by domain (None = all domains)
            limit:          maximum number of pairs to return

        Returns:
            list of DPOPair ready for export
        """
        import httpx

        try:
            params: dict[str, Any] = {"limit": limit}
            if domain:
                params["domain"] = domain
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self._router_url}/corrections", params=params)
                r.raise_for_status()
                data = r.json()
        except Exception:
            # Router not running — return empty list (not an error before aua serve)
            return []

        pairs = []
        for item in data.get("corrections", []):
            if item.get("effective_confidence", 0) >= min_confidence:
                pairs.append(
                    DPOPair(
                        prompt=item.get("subject", ""),
                        chosen=item.get("claim", ""),
                        rejected="",  # populated by Arbiter in v0.7
                        domain=item.get("domain", "general"),
                        confidence=float(item.get("effective_confidence", 0)),
                        source=item.get("source", "arbiter"),
                    )
                )
        return pairs[:limit]

    def export_pairs(
        self,
        pairs: list[DPOPair],
        output_dir: str = "dpo_pairs",
    ) -> CollectionSummary:
        """
        Write DPO pairs to JSONL file for training.

        Args:
            pairs:      list of DPOPair to export
            output_dir: directory to write JSONL files

        Returns:
            CollectionSummary with export details
        """
        import json
        from datetime import datetime, timezone

        out = Path(self._project_dir) / output_dir
        out.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = out / f"dpo_pairs_{timestamp}.jsonl"

        domains: set[str] = set()
        with output_path.open("w") as f:
            for pair in pairs:
                f.write(
                    json.dumps(
                        {
                            "prompt": pair.prompt,
                            "chosen": pair.chosen,
                            "rejected": pair.rejected,
                            "domain": pair.domain,
                            "confidence": pair.confidence,
                            "source": pair.source,
                        }
                    )
                    + "\n"
                )
                domains.add(pair.domain)

        return CollectionSummary(
            n_pairs=len(pairs),
            n_domains=len(domains),
            domains=sorted(domains),
            output_path=str(output_path),
            ready_for_training=False,  # LoRA harness in v0.7
        )

    async def train(
        self,
        pairs_path: str,
        specialist_name: str,
        epochs: int = 1,
    ) -> None:
        """
        Trigger LoRA fine-tuning on the affected specialist.

        Not yet implemented — roadmap #12 (v0.7).
        """
        raise NotImplementedError(
            "CorrectionLoop.train() is roadmap #12 (v0.7). "
            "Export pairs with export_pairs() and run training manually for now."
        )
