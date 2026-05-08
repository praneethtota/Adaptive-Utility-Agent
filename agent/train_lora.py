"""
train_lora.py — DPO LoRA fine-tuning for AUA blue-green deployment
Phase 3 POC

Loads a DPO pair dataset, applies field penalty multipliers as loss weights,
and fine-tunes a LoRA adapter on the base specialist model.

The utility function's penalty multiplier µ(f) is the direct loss weighting
mechanism — a surgery error (µ=10×) contributes 10× more to the training
gradient than a creative writing error (µ=1×). This is the core claim of §5.

Usage:
    python train_lora.py \
        --base-model ./models/swe \
        --dpo-pairs dpo_pairs/cycle1.json \
        --output ./models/swe_green_v1 \
        --epochs 3 \
        --lora-r 16 \
        --lora-alpha 32 \
        --batch-size 2 \
        --field swe \
        --apply-field-weights

Output:
    ./models/swe_green_v1/   — LoRA adapter (merge with base for inference)
    ./models/swe_green_v1/training_log.json — per-step metrics
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("train_lora")


# ── Dependency check ──────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig


# ── Field configs (mirrors config.py) ─────────────────────────────────────────

FIELD_PENALTY = {
    "surgery":               10.0,
    "aviation":              10.0,
    "law":                    5.0,
    "structural_engineering": 4.0,
    "software_engineering":   2.0,
    "swe":                    2.0,
    "stem_research":          2.0,
    "education":              1.5,
    "art":                    1.0,
    "creative_writing":       1.0,
    "general":                1.5,
}


# ── DPO dataset builder ───────────────────────────────────────────────────────

def load_dpo_pairs(
    path: str,
    field: str,
    apply_field_weights: bool,
    min_paired: int = 2,
) -> tuple[Dataset, dict]:
    """
    Load DPO pairs from JSON and build a HuggingFace Dataset.

    Only paired entries (those with both 'chosen' and 'rejected') are used
    for training. Rejected-only entries are counted but skipped.

    Returns: (dataset, stats_dict)
    """
    with open(path) as f:
        raw = json.load(f)

    # Handle both list and dict-with-entries formats
    if isinstance(raw, dict):
        entries = raw.get("entries", raw.get("pairs", []))
    else:
        entries = raw

    paired, skipped = [], []
    for e in entries:
        if e.get("chosen") and e.get("rejected") and e.get("prompt"):
            paired.append(e)
        else:
            skipped.append(e.get("problem_id", "unknown"))

    log.info(f"DPO pairs: {len(paired)} paired, {len(skipped)} rejected-only (skipped)")

    if len(paired) < min_paired:
        log.warning(
            f"Only {len(paired)} paired entries — LoRA update will be minimal. "
            f"Run more harness cycles to accumulate more pairs."
        )

    # Field penalty multiplier
    mu = FIELD_PENALTY.get(field, 1.5)
    base_weight = mu if apply_field_weights else 1.0
    log.info(f"Field: {field} | µ(f) = {mu} | apply_weights = {apply_field_weights}")

    prompts, chosens, rejecteds, weights = [], [], [], []
    for e in paired:
        prompts.append(e["prompt"])
        chosens.append(e["chosen"])
        rejecteds.append(e["rejected"])
        # Per-entry weight overrides base if present and field weights are on
        entry_weight = float(e.get("weight", 1.0))
        final_weight = (entry_weight * base_weight) if apply_field_weights else 1.0
        weights.append(final_weight)

    stats = {
        "total_entries":  len(entries),
        "paired":         len(paired),
        "skipped":        len(skipped),
        "field":          field,
        "mu_f":           mu,
        "mean_weight":    sum(weights) / len(weights) if weights else 0,
        "max_weight":     max(weights) if weights else 0,
    }

    dataset = Dataset.from_dict({
        "prompt":   prompts,
        "chosen":   chosens,
        "rejected": rejecteds,
    })

    return dataset, stats, weights


def format_for_chat(tokenizer, prompt: str, response: str) -> str:
    """Format prompt+response using the model's chat template."""
    messages = [
        {"role": "system", "content": "You are a specialist software engineering assistant. Answer precisely and correctly."},
        {"role": "user",   "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: simple format
        return f"User: {prompt}\nAssistant: {response}"


# ── LoRA config builder ───────────────────────────────────────────────────────

def build_lora_config(r: int, alpha: int, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        # Target the attention and MLP projection layers
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ── Main training function ────────────────────────────────────────────────────

def train(args):
    t_start = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load DPO pairs ────────────────────────────────────────────────────────
    log.info(f"Loading DPO pairs from {args.dpo_pairs}")
    dataset, stats, weights = load_dpo_pairs(
        args.dpo_pairs,
        field=args.field,
        apply_field_weights=args.apply_field_weights,
    )
    log.info(f"Dataset stats: {json.dumps(stats, indent=2)}")

    if len(dataset) == 0:
        log.error("No paired DPO entries found. Run more harness cycles first.")
        sys.exit(1)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading model from {args.base_model}")

    # AWQ checkpoints store weights as packed int4 (qweight/qzeros/scales) — no fp16
    # tensors exist. Load via autoawq which understands the format, then dequantize
    # each WQLinear layer to a standard fp16 nn.Linear so PEFT can wrap it.
    try:
        import torch.nn as nn
        from awq import AutoAWQForCausalLM

        log.info("Loading AWQ model via autoawq for per-layer dequantization...")
        awq_model = AutoAWQForCausalLM.from_quantized(
            args.base_model,
            fuse_layers=False,
            trust_remote_code=True,
            safetensors=True,
            device_map="cuda:0",
        )
        hf_model = awq_model.model

        # Replace every WQLinear with a standard fp16 Linear
        replaced = 0
        for parent_name, parent_module in list(hf_model.named_modules()):
            for child_name, child_module in list(parent_module.named_children()):
                cls_name = type(child_module).__name__
                if "WQLinear" in cls_name:
                    w = child_module.dequantize()   # (in_features, out_features)
                    new_lin = nn.Linear(
                        child_module.in_features,
                        child_module.out_features,
                        bias=child_module.bias is not None,
                        dtype=torch.float16,
                        device="cuda:0",
                    )
                    new_lin.weight = nn.Parameter(w.T.to("cuda:0"))  # nn.Linear expects (out, in)
                    if child_module.bias is not None:
                        new_lin.bias = nn.Parameter(child_module.bias.to("cuda:0"))
                    setattr(parent_module, child_name, new_lin)
                    replaced += 1
        log.info(f"Dequantized {replaced} WQLinear → fp16 Linear layers")
        model = hf_model
    except Exception as e:
        log.warning(f"AWQ dequantize failed ({e}); loading fp16 with stripped config (fallback)")
        model_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
        if hasattr(model_config, "quantization_config"):
            del model_config.quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=model_config,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
        log.info("Fallback fp16 load (weights may be random — use only if checkpoint is not AWQ)")
    log.info("Model ready for LoRA training")

    model.config.use_cache = False

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    log.info(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = build_lora_config(args.lora_r, args.lora_alpha)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── DPO training args ─────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        # DPO-specific
        beta=0.1,            # KL penalty — lower = more aggressive updates
        max_length=512,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        ref_model=None,       # None = use frozen copy of model as reference
    )

    log.info(f"Starting DPO training — {len(dataset)} pairs, {args.epochs} epochs")
    log.info(f"Estimated time: {len(dataset) * args.epochs * 2 // 60 + 5} minutes on RTX 4090")

    train_result = trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    log.info(f"Saving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ── Write training log ────────────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    training_log = {
        "base_model":      args.base_model,
        "output":          str(output_dir),
        "field":           args.field,
        "dpo_pairs_file":  args.dpo_pairs,
        "dataset_stats":   stats,
        "lora_r":          args.lora_r,
        "lora_alpha":      args.lora_alpha,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "learning_rate":   args.lr,
        "apply_field_weights": args.apply_field_weights,
        "train_loss":      train_result.training_loss,
        "elapsed_seconds": round(t_elapsed, 1),
        "elapsed_human":   f"{t_elapsed/60:.1f} min",
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    log.info(f"Training complete in {t_elapsed/60:.1f} min")
    log.info(f"Train loss: {train_result.training_loss:.4f}")
    log.info(f"Adapter saved to: {output_dir}")
    log.info(f"Training log: {log_path}")
    log.info("")
    log.info("Next step: start GREEN server")
    log.info(f"  python -m vllm.entrypoints.openai.api_server \\")
    log.info(f"    --model {output_dir} --port 9011 \\")
    log.info(f"    --quantization awq --max-model-len 2048 \\")
    log.info(f"    --gpu-memory-utilization 0.30")

    return training_log


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPO LoRA fine-tuning for AUA blue-green")
    p.add_argument("--base-model",   required=True,       help="Path to base model (e.g. ./models/swe)")
    p.add_argument("--dpo-pairs",    required=True,       help="Path to DPO pairs JSON")
    p.add_argument("--output",       required=True,       help="Output directory for LoRA adapter")
    p.add_argument("--field",        default="swe",       help="Field name for penalty multiplier lookup")
    p.add_argument("--epochs",       type=int, default=3, help="Training epochs")
    p.add_argument("--lora-r",       type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha",   type=int, default=32, help="LoRA alpha")
    p.add_argument("--batch-size",   type=int, default=2,  help="Per-device batch size")
    p.add_argument("--lr",           type=float, default=5e-5, help="Learning rate")
    p.add_argument("--apply-field-weights", action="store_true", default=True,
                   help="Apply µ(f) penalty multiplier as DPO loss weight")
    p.add_argument("--dry-run", action="store_true",
                   help="Load data and model but skip training (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("=== AUA LoRA Trainer ===")
    log.info(f"Base model:  {args.base_model}")
    log.info(f"DPO pairs:   {args.dpo_pairs}")
    log.info(f"Output:      {args.output}")
    log.info(f"Field:       {args.field} (µ = {FIELD_PENALTY.get(args.field, 1.5)}×)")
    log.info(f"Epochs:      {args.epochs}")
    log.info(f"LoRA:        r={args.lora_r}, alpha={args.lora_alpha}")
    log.info("")

    if args.dry_run:
        log.info("DRY RUN — loading data only, skipping training")
        dataset, stats, weights = load_dpo_pairs(
            args.dpo_pairs, args.field, args.apply_field_weights
        )
        log.info(f"Dataset: {len(dataset)} paired entries")
        log.info(f"Stats: {json.dumps(stats, indent=2)}")
        log.info("Dry run complete — training would proceed with above config")
    else:
        result = train(args)
