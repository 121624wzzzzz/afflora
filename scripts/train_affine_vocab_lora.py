#!/usr/bin/env python
"""Train hidden LoRA and/or affine vocab adapters for the main SFT task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affine_vocab_lora import AffineVocabConfig, apply_affine_vocab_adapters, load_affine_vocab_adapter, save_affine_vocab_adapter


MASTER_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


PROMPT_TEMPLATE = """Question:
{question}

Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "full_finetune",
            "hidden_lora",
            "affine_input",
            "affine_lm_head",
            "affine_input_lm_head",
            "affine_input_plus_hidden_lora",
            "affine_lm_head_plus_hidden_lora",
            "affine_input_lm_head_plus_hidden_lora",
        ],
        required=True,
    )
    parser.add_argument("--hidden-lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument(
        "--hidden-lora-layers-to-transform",
        default=None,
        help=(
            "Comma-separated layer indices to apply LoRA to (e.g. '14' or '0,13,27')."
            " If omitted, LoRA is applied to every decoder layer (default behavior)."
            " Useful for matched-capacity comparisons against affine adapters."
        ),
    )
    parser.add_argument("--affine-rank", type=int, default=16)
    parser.add_argument("--affine-alpha", type=float, default=32.0)
    parser.add_argument(
        "--affine-bias-scale",
        type=float,
        default=1.0,
        help="s2 in W' = W (I + s1 AB) + s2 b. Default 1.0 keeps prior behavior.",
    )
    parser.add_argument("--affine-dropout", type=float, default=0.0)
    parser.add_argument(
        "--affine-learning-rate-scale",
        type=float,
        default=1.0,
        help=(
            "Learning-rate multiplier applied only to A-LoRA parameters. "
            "Hidden LoRA and all other trainable parameters retain --learning-rate."
        ),
    )
    parser.add_argument(
        "--affine-bias-learning-rate-scale",
        type=float,
        default=1.0,
        help=(
            "Additional learning-rate multiplier for affine bias parameters, relative "
            "to the affine learning rate. Default 1.0 preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--affine-energy-lambda",
        type=float,
        default=0.0,
        help=(
            "Weight for the A-LoRA energy trust-region penalty. Input embedding "
            "terms use the actual uncentered row-codebook update; independent "
            "output terms use the centered output-codebook update."
        ),
    )
    parser.add_argument(
        "--affine-energy-tau",
        type=float,
        default=0.0,
        help="Allowed relative codebook-update norm before the hinge penalty activates.",
    )
    parser.add_argument(
        "--affine-energy-exclude-bias",
        action="store_true",
        help=(
            "Apply the global energy trust region only to the low-rank linear "
            "codebook update, excluding the affine translation/bias."
        ),
    )
    parser.add_argument(
        "--affine-energy-bias-weight",
        type=float,
        default=1.0,
        help=(
            "Relative weight of the affine translation inside the global energy "
            "norm. 1.0 is the exact codebook update and 0.0 matches "
            "--affine-energy-exclude-bias."
        ),
    )
    parser.add_argument(
        "--affine-energy-start-step",
        type=int,
        default=0,
        help=(
            "Keep the global affine-energy penalty inactive before this optimizer "
            "step; 0 applies it throughout training."
        ),
    )
    parser.add_argument(
        "--affine-energy-end-step",
        type=int,
        default=-1,
        help=(
            "Deactivate the global affine-energy penalty at this optimizer step. "
            "-1 keeps it active through training; together with --affine-energy-"
            "start-step this selects a half-open [start, end) schedule."
        ),
    )
    parser.add_argument(
        "--affine-bias-energy-lambda",
        type=float,
        default=0.0,
        help="Weight of an independent affine-bias trust-region hinge penalty.",
    )
    parser.add_argument(
        "--affine-bias-energy-tau",
        type=float,
        default=0.0,
        help="Allowed relative codebook translation norm sqrt(V)||beta||/||W||_F.",
    )
    parser.add_argument("--no-affine-input-bias", action="store_true")
    parser.add_argument("--affine-lm-head-bias", action="store_true")
    parser.add_argument(
        "--initial-affine-adapter",
        default=None,
        help=(
            "Directory containing an affine_vocab_adapter.safetensors file to load "
            "after constructing the requested affine topology and before training."
        ),
    )
    parser.add_argument(
        "--tie-affine-input-lm-head-adapters",
        action="store_true",
        help=(
            "For tied-embedding models, share one affine adapter between input"
            " embeddings and lm_head so the learned adapter can be merged back"
            " into the single tied embedding matrix. Uses transpose affine on"
            " the lm_head side for merge equivalence. Input and lm_head bias"
            " settings must match."
        ),
    )
    parser.add_argument("--hidden-lora-rank", type=int, default=16)
    parser.add_argument("--hidden-lora-alpha", type=int, default=32)
    parser.add_argument("--hidden-lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--initial-hidden-lora-adapter",
        default=None,
        help="Load an existing hidden LoRA adapter instead of initializing a new one.",
    )
    parser.add_argument(
        "--freeze-initial-hidden-lora",
        action="store_true",
        help="Freeze a loaded hidden LoRA so only newly added affine parameters train.",
    )
    parser.add_argument(
        "--include-emb-lmh-lora-rank",
        type=int,
        default=0,
        help=(
            "Claim 1b control: if > 0, additionally place vocab-dim LoRA on"
            " embed_tokens and lm_head at the given rank, on top of the normal"
            " hidden_lora variant. Default 0 disables this. Most useful at"
            " rank=1 since vocab-dim LoRA cost scales with vocab_size * rank."
        ),
    )
    parser.add_argument(
        "--emb-lmh-lora-alpha",
        type=int,
        default=None,
        help="Alpha for the emb/lm_head LoRA. Defaults to 2*include_emb_lmh_lora_rank.",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--anchor-data",
        default=None,
        help=(
            "Optional assistant-conversation corpus mixed into training as a behavior anchor. "
            "It is sampled independently of --train-data and must use the same row schema."
        ),
    )
    parser.add_argument(
        "--anchor-fraction",
        type=float,
        default=0.0,
        help="Fraction of the final fixed-size training set drawn from --anchor-data.",
    )
    parser.add_argument(
        "--anchor-sampling-seed",
        type=int,
        default=1729,
        help="Fixed source-selection seed for --anchor-data; training order still uses --seed.",
    )
    parser.add_argument(
        "--auxiliary-anchor-data",
        default=None,
        help=(
            "Optional assistant-conversation corpus used as an additional supervised loss. "
            "Unlike --anchor-data, it never replaces rows in --train-data."
        ),
    )
    parser.add_argument(
        "--auxiliary-anchor-lambda",
        type=float,
        default=0.0,
        help="Weight of the additional anchor cross-entropy loss (0 disables it).",
    )
    parser.add_argument(
        "--auxiliary-anchor-samples",
        type=int,
        default=0,
        help="Fixed number of anchor rows to cycle through; 0 uses the full anchor corpus.",
    )
    parser.add_argument(
        "--auxiliary-anchor-batch-size",
        type=int,
        default=1,
        help="Microbatch size of the additional anchor forward pass.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--master-dtype",
        choices=list(MASTER_DTYPE_MAP),
        default="fp32",
        help=(
            "Storage dtype for *trainable* adapter parameters. Defaults to fp32 to avoid"
            " the bf16-master-weight pitfall documented in cpt/实验记录.md §9: with bf16"
            " params + bf16 AdamW state, lr=2e-4 updates get rounded away and training"
            " silently stalls."
        ),
    )
    parser.add_argument(
        "--base-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Storage dtype for the *frozen* base model. 'auto' uses config.torch_dtype.",
    )
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--skip-final-model-save", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--eval-data", default=None, help="Optional held-out JSONL/HF path for eval loss.")
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reference-run-dir", default=None)
    parser.add_argument("--reference-kl-lambda", type=float, default=0.0)
    parser.add_argument("--reference-kl-temperature", type=float, default=1.0)
    parser.add_argument(
        "--reference-kl-on-auxiliary-anchor",
        action="store_true",
        help=(
            "Compute teacher KL on the separate auxiliary-anchor batch instead of the "
            "main SFT batch. Requires both --reference-run-dir and an auxiliary anchor."
        ),
    )
    parser.add_argument(
        "--reference-kl-on-main-and-auxiliary-anchor",
        action="store_true",
        help=(
            "Average teacher KL equally over the main SFT and separate auxiliary-anchor "
            "batches. Requires both --reference-run-dir and an auxiliary anchor; "
            "mutually exclusive with --reference-kl-on-auxiliary-anchor."
        ),
    )
    return parser.parse_args()


def load_any_dataset(data_ref: str, split: str) -> Dataset:
    path = Path(data_ref)
    if path.exists() and path.is_dir():
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local data suffix: {suffix}")
    return load_dataset(data_ref, split=split)


def mix_anchor_dataset(train_ds: Dataset, args: argparse.Namespace) -> Dataset:
    """Make a fixed-size primary/anchor mixture without changing epoch budget.

    Source rows are selected with a fixed seed so matched runs differ only in their
    model objective.  The final shuffle uses the training seed and is therefore
    paired between treatment and control of a given seed.
    """

    if args.anchor_fraction == 0.0:
        if args.anchor_data:
            raise ValueError("--anchor-data requires --anchor-fraction > 0")
        return train_ds
    if not args.anchor_data:
        raise ValueError("--anchor-fraction > 0 requires --anchor-data")
    if not 0.0 < args.anchor_fraction < 1.0:
        raise ValueError("--anchor-fraction must be strictly between 0 and 1")

    total = args.max_train_samples or len(train_ds)
    anchor_count = round(total * args.anchor_fraction)
    primary_count = total - anchor_count
    anchor_ds = load_any_dataset(args.anchor_data, args.dataset_split)
    if primary_count > len(train_ds) or anchor_count > len(anchor_ds):
        raise ValueError(
            "Requested anchor mixture exceeds an input corpus: "
            f"primary={primary_count}/{len(train_ds)}, anchor={anchor_count}/{len(anchor_ds)}"
        )

    # Keep only the shared training payload before concatenating heterogeneous
    # JSON metadata columns from the two corpora.
    primary_ds = train_ds.select_columns(["conversations"])
    anchor_ds = anchor_ds.select_columns(["conversations"])
    primary_ds = primary_ds.shuffle(seed=args.anchor_sampling_seed).select(range(primary_count))
    anchor_ds = anchor_ds.shuffle(seed=args.anchor_sampling_seed + 1).select(range(anchor_count))
    mixed = concatenate_datasets([primary_ds, anchor_ds]).shuffle(seed=args.seed)
    print(
        "[anchor_mix] "
        f"total={len(mixed)} primary={primary_count} anchor={anchor_count} "
        f"fraction={args.anchor_fraction:g} source_seed={args.anchor_sampling_seed}",
        flush=True,
    )
    return mixed


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def normalize_example(row: dict[str, Any]) -> tuple[str, str]:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        user_parts = []
        assistant_parts = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = str(turn.get("content") or "")
            if role == "user" and content:
                user_parts.append(content)
            elif role == "assistant" and content:
                assistant_parts.append(content)
        if user_parts and assistant_parts:
            return "\n".join(user_parts), "\n".join(assistant_parts)

    question = first_present(row, ["query", "question", "instruction", "problem", "input", "prompt"])
    answer = first_present(row, ["response", "answer", "output", "solution", "target", "completion"])
    if not question or not answer:
        raise ValueError(
            "Each training row must contain a question-like field and an answer-like field."
        )
    return question, answer


def tokenize_row(row: dict[str, Any], tokenizer: Any, max_seq_len: int) -> dict[str, list[int]]:
    question, answer = normalize_example(row)
    prompt = PROMPT_TEMPLATE.format(question=question)
    # Tokenize prompt and response independently, then concatenate their ids.  Using
    # len(tokenizer(prompt)) as a mask boundary inside tokenizer(prompt + response)
    # is unsafe: BPE tokenizers may merge across that string boundary.  For Qwen3,
    # this used to shift the response mask on answers beginning with a newline.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_seq_len]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def variant_uses_affine(variant: str) -> tuple[bool, bool]:
    return "affine_input" in variant, "lm_head" in variant


def variant_uses_hidden_lora(variant: str) -> bool:
    return variant == "hidden_lora" or variant.endswith("plus_hidden_lora")


def variant_uses_full_finetune(variant: str) -> bool:
    return variant == "full_finetune"


def trainable_summary(model: Any) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "pct": trainable / total * 100}


def enable_affine_trainable(model: Any) -> int:
    count = 0
    for name, param in model.named_parameters():
        if ".affine." in name:
            param.requires_grad_(True)
            count += param.numel()
    return count


def cast_trainable_to_master_dtype(model: Any, master_dtype: torch.dtype) -> dict[str, int]:
    """Force every trainable param to ``master_dtype`` (typically fp32).

    PEFT 0.19's ``_move_adapter_to_device_of_base_layer`` casts LoRA params to the base
    layer's dtype. If the base is loaded in bf16, LoRA + AdamW state end up bf16, which
    silently kills small lr updates (cpt/实验记录.md §9). Call this after PEFT wrapping
    and after enable_affine_trainable to make sure all trainable storage is fp32.
    """

    cast = 0
    already = 0
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dtype == master_dtype:
            already += 1
            continue
        param.data = param.data.to(dtype=master_dtype)
        cast += 1
    return {"cast": cast, "already": already}


def precision_sanity_report(model: Any, master_dtype_arg: str) -> dict[str, Any]:
    trainable_dtypes: dict[str, int] = {}
    sample = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = str(param.dtype)
        trainable_dtypes[key] = trainable_dtypes.get(key, 0) + 1
        if sample is None:
            sample = (name, key)
    frozen_dtypes: dict[str, int] = {}
    for _, param in model.named_parameters():
        if param.requires_grad:
            continue
        key = str(param.dtype)
        frozen_dtypes[key] = frozen_dtypes.get(key, 0) + 1
    return {
        "master_dtype_arg": master_dtype_arg,
        "trainable_param_dtypes": trainable_dtypes,
        "frozen_param_dtypes": frozen_dtypes,
        "sample_trainable_param": sample,
    }


class SaveAffineAdapterCallback(TrainerCallback):
    """Mirror Trainer's intermediate checkpoint saves for the affine vocab adapter.

    Trainer's ``save_model`` only serialises PEFT adapter files. Without this callback,
    a job that dies mid-training would lose every affine vocab update.

    For AffLoRA-only runs (no PEFT/hidden LoRA), Trainer also writes the full frozen
    base model into the checkpoint (~16 GB for 8B). We remove it after saving the
    lightweight affine adapter, since the base model is already on disk at --model-path.
    This does break resume-from-checkpoint for AffLoRA-only runs (just restart from
    scratch with the same seed — the adapter is tiny).
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def _resolve_base(self) -> Any:
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def on_save(self, args, state, control, **kwargs):  # noqa: D401, ANN001
        if not state.is_world_process_zero:
            return
        if getattr(self._resolve_base(), "affine_vocab_config", None) is None:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_affine_vocab_adapter(self._resolve_base(), ckpt_dir)

        # Remove full model checkpoint saved by Trainer for non-PEFT AffLoRA runs.
        # PEFT-wrapped models only write adapter_model.safetensors (small); this
        # cleanup only fires when Trainer wrote the full model.safetensors (16 GB).
        full_model = ckpt_dir / "model.safetensors"
        affine_adapter = ckpt_dir / "affine_vocab_adapter.safetensors"
        if full_model.exists() and affine_adapter.exists():
            full_model.unlink()
            # Also remove the full-model index if present
            for f in ckpt_dir.glob("model.safetensors.index.json"):
                f.unlink()


class AffineLearningRateTrainer(Trainer):
    """Trainer with a separate LR parameter group for A-LoRA parameters."""

    def __init__(
        self,
        *args: Any,
        affine_learning_rate_scale: float = 1.0,
        affine_bias_learning_rate_scale: float = 1.0,
        affine_energy_lambda: float = 0.0,
        affine_energy_tau: float = 0.0,
        affine_energy_exclude_bias: bool = False,
        affine_energy_bias_weight: float = 1.0,
        affine_energy_start_step: int = 0,
        affine_energy_end_step: int = -1,
        affine_bias_energy_lambda: float = 0.0,
        affine_bias_energy_tau: float = 0.0,
        reference_model: torch.nn.Module | None = None,
        reference_kl_lambda: float = 0.0,
        reference_kl_temperature: float = 1.0,
        reference_kl_on_auxiliary_anchor: bool = False,
        reference_kl_on_main_and_auxiliary_anchor: bool = False,
        auxiliary_anchor_dataset: Dataset | None = None,
        auxiliary_anchor_lambda: float = 0.0,
        auxiliary_anchor_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        if affine_learning_rate_scale <= 0:
            raise ValueError("affine_learning_rate_scale must be positive")
        if affine_bias_learning_rate_scale <= 0:
            raise ValueError("affine_bias_learning_rate_scale must be positive")
        if not 0.0 <= affine_energy_bias_weight <= 1.0:
            raise ValueError("affine_energy_bias_weight must be in [0, 1]")
        if affine_energy_start_step < 0:
            raise ValueError("affine_energy_start_step must be non-negative")
        if affine_energy_end_step >= 0 and affine_energy_end_step < affine_energy_start_step:
            raise ValueError("affine_energy_end_step must be -1 or >= affine_energy_start_step")
        self.affine_learning_rate_scale = float(affine_learning_rate_scale)
        self.affine_bias_learning_rate_scale = float(affine_bias_learning_rate_scale)
        self.affine_energy_lambda = float(affine_energy_lambda)
        self.affine_energy_tau = float(affine_energy_tau)
        self.affine_energy_exclude_bias = bool(affine_energy_exclude_bias)
        self.affine_energy_bias_weight = float(affine_energy_bias_weight)
        self.affine_energy_start_step = int(affine_energy_start_step)
        self.affine_energy_end_step = int(affine_energy_end_step)
        self.affine_bias_energy_lambda = float(affine_bias_energy_lambda)
        self.affine_bias_energy_tau = float(affine_bias_energy_tau)
        if reference_kl_lambda < 0 or reference_kl_temperature <= 0:
            raise ValueError("reference KL lambda must be non-negative and temperature positive")
        if auxiliary_anchor_lambda < 0:
            raise ValueError("auxiliary anchor lambda must be non-negative")
        if auxiliary_anchor_batch_size <= 0:
            raise ValueError("auxiliary anchor batch size must be positive")
        if auxiliary_anchor_lambda > 0 and auxiliary_anchor_dataset is None:
            raise ValueError("auxiliary anchor lambda requires an auxiliary anchor dataset")
        if reference_kl_on_auxiliary_anchor and reference_kl_on_main_and_auxiliary_anchor:
            raise ValueError("reference KL source selectors are mutually exclusive")
        if (reference_kl_on_auxiliary_anchor or reference_kl_on_main_and_auxiliary_anchor) and (
            reference_model is None or auxiliary_anchor_dataset is None
        ):
            raise ValueError("auxiliary-anchor KL requires both teacher and auxiliary anchor data")
        self.reference_model = reference_model
        self.reference_kl_lambda = float(reference_kl_lambda)
        self.reference_kl_temperature = float(reference_kl_temperature)
        self.reference_kl_on_auxiliary_anchor = bool(reference_kl_on_auxiliary_anchor)
        self.reference_kl_on_main_and_auxiliary_anchor = bool(
            reference_kl_on_main_and_auxiliary_anchor
        )
        self.auxiliary_anchor_dataset = auxiliary_anchor_dataset
        self.auxiliary_anchor_lambda = float(auxiliary_anchor_lambda)
        self.auxiliary_anchor_batch_size = int(auxiliary_anchor_batch_size)
        self._auxiliary_anchor_iterator: Any = None
        self._auxiliary_last_logged_step = -1
        self._energy_terms: list[dict[str, Any]] | None = None
        self._energy_last_logged_step = -1
        super().__init__(*args, **kwargs)

    @staticmethod
    def _energy_statistics(
        weight: torch.Tensor,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return W^T W, 1^T W, ||W||_F^2, and row count for an energy metric."""
        vocab, hidden = weight.shape
        covariance = torch.zeros((hidden, hidden), dtype=torch.float32, device=weight.device)
        row_sum = torch.zeros(hidden, dtype=torch.float32, device=weight.device)
        with torch.no_grad():
            for begin in range(0, vocab, 4096):
                chunk = weight[begin : begin + 4096].float()
                covariance.addmm_(chunk.T, chunk)
                row_sum.add_(chunk.sum(dim=0))
            if centered:
                covariance.add_(torch.outer(row_sum, row_sum), alpha=-1.0 / float(vocab))
                row_sum.zero_()
            denominator = covariance.diagonal().sum().clamp_min(1e-12)
        return covariance, row_sum, denominator, vocab

    def _prepare_affine_energy(self) -> None:
        if (
            self.affine_energy_lambda <= 0
            and self.affine_bias_energy_lambda <= 0
        ) or self._energy_terms is not None:
            return
        embeddings = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "AffineEmbedding"
            and hasattr(module, "base_embedding")
            and hasattr(module, "affine")
        ]
        heads = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "AffineLMHead"
            and hasattr(module, "base_head")
            and hasattr(module, "affine")
        ]
        terms: list[dict[str, Any]] = []
        input_affine_ids: set[int] = set()
        for index, embedding in enumerate(embeddings):
            covariance, row_sum, denominator, vocab = self._energy_statistics(
                embedding.base_embedding.weight.detach(), centered=False
            )
            terms.append({
                "name": f"input{index}", "affine": embedding.affine,
                "covariance": covariance, "row_sum": row_sum,
                "denominator": denominator, "vocab": vocab,
                # AffineEmbedding maps a row x as x + s * x D^T U^T + beta,
                # hence its actual codebook update is W D^T U^T + 1 beta^T.
                "geometry": "input_row_codebook",
            })
            input_affine_ids.add(id(embedding.affine))
        for index, head in enumerate(heads):
            if id(head.affine) in input_affine_ids:
                continue
            if getattr(head.affine, "bias", None) is not None:
                raise RuntimeError(
                    "An independent output-side energy term expects beta disabled because a "
                    "common logit shift is softmax-invariant."
                )
            covariance, row_sum, denominator, vocab = self._energy_statistics(
                head.base_head.weight.detach(), centered=True
            )
            terms.append({
                "name": f"output{index}", "affine": head.affine,
                "covariance": covariance, "row_sum": row_sum,
                "denominator": denominator, "vocab": vocab,
                # AffineLMHead maps a hidden row h as h + s * h D^T U^T;
                # its induced output codebook update is W U D.
                "geometry": "output_codebook",
            })
        if not terms:
            raise RuntimeError("No supported affine input/output module found for energy penalty")
        self._energy_terms = terms
        for term in terms:
            print(
                f"[energy] prepared {term['name']} covariance "
                f"shape={tuple(term['covariance'].shape)} "
                f"denominator={float(term['denominator']):.8e} "
                f"bias={getattr(term['affine'], 'bias', None) is not None} "
                f"tau={self.affine_energy_tau:g} lambda={self.affine_energy_lambda:g} "
                f"bias_weight={self.affine_energy_bias_weight:g} "
                f"bias_tau={self.affine_bias_energy_tau:g} "
                f"bias_lambda={self.affine_bias_energy_lambda:g}"
            )

    @staticmethod
    def _affine_energy_rho(
        term: dict[str, Any], *, bias_weight: float = 1.0
    ) -> torch.Tensor:
        affine = term["affine"]
        up = affine.up.weight.float()
        down = affine.down.weight.float()
        scale = float(affine.scale)
        if term["geometry"] == "input_row_codebook":
            # Input rows are transformed by D^T U^T, not U D. This also covers
            # a tied-transpose head because its exact merged codebook is the
            # same W D^T U^T transformation.
            small_cov = down @ term["covariance"] @ down.T
            up_gram = up.T @ up
            numerator = (small_cov * up_gram).sum() * scale**2
        elif term["geometry"] == "output_codebook":
            small_cov = up.T @ term["covariance"] @ up
            down_gram = down @ down.T
            numerator = (small_cov * down_gram).sum() * scale**2
        else:
            raise RuntimeError(f"Unknown affine-energy geometry: {term['geometry']!r}")
        bias = getattr(affine, "bias", None)
        if bias_weight and bias is not None:
            beta = bias.float() * float(affine.bias_scale) * bias_weight
            if term["geometry"] == "input_row_codebook":
                summed_multiplicative_update = (term["row_sum"] @ down.T) @ up.T
            else:
                summed_multiplicative_update = (term["row_sum"] @ up) @ down
            numerator = numerator + 2.0 * scale * (summed_multiplicative_update * beta).sum()
            numerator = numerator + float(term["vocab"]) * beta.square().sum()
        return torch.sqrt((numerator / term["denominator"]).clamp_min(0.0) + 1e-20)

    @staticmethod
    def _affine_bias_energy_rho(term: dict[str, Any]) -> torch.Tensor:
        """Relative norm of the additive codebook translation alone."""
        affine = term["affine"]
        bias = getattr(affine, "bias", None)
        if bias is None:
            return torch.zeros((), dtype=torch.float32, device=term["denominator"].device)
        beta = bias.float() * float(affine.bias_scale)
        numerator = float(term["vocab"]) * beta.square().sum()
        return torch.sqrt((numerator / term["denominator"]).clamp_min(0.0) + 1e-20)

    def _next_auxiliary_anchor_inputs(self) -> dict[str, Any]:
        """Cycle a deterministic anchor loader without changing main-dataset sampling."""
        if self.auxiliary_anchor_dataset is None:
            raise RuntimeError("Auxiliary anchor loader requested without a dataset")
        if self._auxiliary_anchor_iterator is None:
            loader = torch.utils.data.DataLoader(
                self.auxiliary_anchor_dataset,
                batch_size=self.auxiliary_anchor_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
            self._auxiliary_anchor_iterator = iter(loader)
        try:
            batch = next(self._auxiliary_anchor_iterator)
        except StopIteration:
            loader = torch.utils.data.DataLoader(
                self.auxiliary_anchor_dataset,
                batch_size=self.auxiliary_anchor_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
            self._auxiliary_anchor_iterator = iter(loader)
            batch = next(self._auxiliary_anchor_iterator)
        return self._prepare_inputs(batch)

    def _teacher_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Assistant-token KL for either the main SFT or independent anchor batch."""
        temperature = self.reference_kl_temperature
        token_kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits.float() / temperature, dim=-1),
            torch.nn.functional.softmax(teacher_logits.float() / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        mask = labels[:, 1:].ne(-100)
        return token_kl[:, :-1][mask].mean() * temperature**2

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        step = int(self.state.global_step)
        energy_is_active = step >= self.affine_energy_start_step and (
            self.affine_energy_end_step < 0 or step < self.affine_energy_end_step
        )
        active_energy_lambda = self.affine_energy_lambda if energy_is_active else 0.0
        if (
            active_energy_lambda <= 0
            and self.affine_bias_energy_lambda <= 0
            and self.reference_model is None
            and self.auxiliary_anchor_lambda <= 0
        ):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        task_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        kl_penalty = torch.zeros((), device=task_loss.device, dtype=torch.float32)
        auxiliary_anchor_loss = torch.zeros((), device=task_loss.device, dtype=torch.float32)
        auxiliary_inputs = None
        auxiliary_outputs = None
        if self.auxiliary_anchor_lambda > 0:
            auxiliary_inputs = self._next_auxiliary_anchor_inputs()
            if (
                self.reference_kl_on_auxiliary_anchor
                or self.reference_kl_on_main_and_auxiliary_anchor
            ):
                auxiliary_anchor_loss, auxiliary_outputs = super().compute_loss(
                    model, auxiliary_inputs, return_outputs=True
                )
                auxiliary_anchor_loss = auxiliary_anchor_loss.float()
            else:
                auxiliary_anchor_loss = super().compute_loss(
                    model, auxiliary_inputs, return_outputs=False
                ).float()
        if self.reference_model is not None and self.reference_kl_lambda > 0:
            if self.reference_kl_on_main_and_auxiliary_anchor:
                if auxiliary_inputs is None or auxiliary_outputs is None:
                    raise RuntimeError("Main-plus-anchor KL was requested without an anchor batch")
                with torch.no_grad():
                    main_reference_logits = self.reference_model(
                        input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
                    ).logits.float()
                    auxiliary_reference_logits = self.reference_model(
                        input_ids=auxiliary_inputs["input_ids"],
                        attention_mask=auxiliary_inputs.get("attention_mask"),
                    ).logits.float()
                kl_penalty = 0.5 * (
                    self._teacher_kl(outputs.logits, main_reference_logits, inputs["labels"])
                    + self._teacher_kl(
                        auxiliary_outputs.logits,
                        auxiliary_reference_logits,
                        auxiliary_inputs["labels"],
                    )
                )
            else:
                kl_inputs = auxiliary_inputs if self.reference_kl_on_auxiliary_anchor else inputs
                student_logits = auxiliary_outputs.logits if self.reference_kl_on_auxiliary_anchor else outputs.logits
                if kl_inputs is None:
                    raise RuntimeError("Auxiliary-anchor KL was requested without an anchor batch")
                with torch.no_grad():
                    reference_logits = self.reference_model(
                        input_ids=kl_inputs["input_ids"], attention_mask=kl_inputs.get("attention_mask")
                    ).logits.float()
                kl_penalty = self._teacher_kl(student_logits, reference_logits, kl_inputs["labels"])
        if active_energy_lambda > 0 or self.affine_bias_energy_lambda > 0:
            self._prepare_affine_energy()
        rhos = [
            (term["name"], self._affine_energy_rho(
                term,
                bias_weight=(
                    0.0 if self.affine_energy_exclude_bias
                    else self.affine_energy_bias_weight
                ),
            ))
            for term in (self._energy_terms or [])
        ]
        excesses = [(name, torch.relu(rho - self.affine_energy_tau)) for name, rho in rhos]
        bias_rhos = [
            (term["name"], self._affine_bias_energy_rho(term)) for term in (self._energy_terms or [])
        ]
        bias_excesses = [
            (name, torch.relu(rho - self.affine_bias_energy_tau)) for name, rho in bias_rhos
        ]
        penalty = active_energy_lambda * sum(excess.square() for _, excess in excesses)
        bias_penalty = self.affine_bias_energy_lambda * sum(
            excess.square() for _, excess in bias_excesses
        )
        loss = task_loss + (
            penalty
            + bias_penalty
            + self.reference_kl_lambda * kl_penalty
            + self.auxiliary_anchor_lambda * auxiliary_anchor_loss
        ).to(dtype=task_loss.dtype)
        if (active_energy_lambda > 0 or self.affine_bias_energy_lambda > 0) and step % max(1, int(self.args.logging_steps)) == 0 and step != self._energy_last_logged_step:
            self._energy_last_logged_step = step
            values = " ".join(
                f"{name}_rho={float(rho.detach()):.8f} "
                f"{name}_excess={float(dict(excesses)[name].detach()):.8f}"
                for name, rho in rhos
            )
            bias_values = " ".join(
                f"{name}_bias_rho={float(rho.detach()):.8f} "
                f"{name}_bias_excess={float(dict(bias_excesses)[name].detach()):.8f}"
                for name, rho in bias_rhos
            )
            print(
                f"[energy] step={step} {values} {bias_values} "
                f"active_lambda={active_energy_lambda:g} "
                f"penalty={float(penalty.detach()):.8e} "
                f"bias_penalty={float(bias_penalty.detach()):.8e}"
            )
        if self.reference_model is not None and step % max(1, int(self.args.logging_steps)) == 0:
            source = (
                "main+auxiliary_anchor"
                if self.reference_kl_on_main_and_auxiliary_anchor
                else "auxiliary_anchor" if self.reference_kl_on_auxiliary_anchor else "main"
            )
            print(
                f"[reference_kl] step={step} value={float(kl_penalty.detach()):.8e} "
                f"lambda={self.reference_kl_lambda:g} source={source}"
            )
        if (
            self.auxiliary_anchor_lambda > 0
            and step % max(1, int(self.args.logging_steps)) == 0
            and step != self._auxiliary_last_logged_step
        ):
            self._auxiliary_last_logged_step = step
            print(
                f"[auxiliary_anchor] step={step} loss={float(auxiliary_anchor_loss.detach()):.8e} "
                f"lambda={self.auxiliary_anchor_lambda:g}",
                flush=True,
            )
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
        if (
            self.affine_learning_rate_scale == 1.0
            and self.affine_bias_learning_rate_scale == 1.0
        ):
            return super().create_optimizer(model)
        opt_model = self.model if model is None else model
        if self.optimizer is not None:
            return self.optimizer

        decay_names = self.get_decay_parameter_names(opt_model)
        base_lr = float(self.args.learning_rate)
        affine_lr = base_lr * self.affine_learning_rate_scale
        affine_bias_lr = affine_lr * self.affine_bias_learning_rate_scale
        grouped: list[dict[str, Any]] = []
        counts = {"hidden_or_other": 0, "affine_weight": 0, "affine_bias": 0}

        def is_affine_name(name: str) -> bool:
            return name.startswith("affine.") or ".affine." in name

        def parameter_group(name: str) -> str:
            if not is_affine_name(name):
                return "hidden_or_other"
            if name.endswith(".affine.bias"):
                return "affine_bias"
            return "affine_weight"

        for group_name, group_lr in (
            ("hidden_or_other", base_lr),
            ("affine_weight", affine_lr),
            ("affine_bias", affine_bias_lr),
        ):
            for use_decay in (True, False):
                params = [
                    param
                    for name, param in opt_model.named_parameters()
                    if param.requires_grad
                    and parameter_group(name) == group_name
                    and (name in decay_names) == use_decay
                ]
                if not params:
                    continue
                grouped.append(
                    {
                        "params": params,
                        "weight_decay": self.args.weight_decay if use_decay else 0.0,
                        "lr": group_lr,
                    }
                )
                counts[group_name] += sum(param.numel() for param in params)
        if counts["affine_weight"] == 0:
            raise RuntimeError(
                "A separate affine learning rate was requested, but no trainable "
                "affine weight parameters were found"
            )
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        print(
            "[optimizer] separate_affine_lr "
            f"base_lr={base_lr:g} affine_lr={affine_lr:g} "
            f"affine_scale={self.affine_learning_rate_scale:g} "
            f"affine_bias_lr={affine_bias_lr:g} "
            f"bias_scale={self.affine_bias_learning_rate_scale:g} counts={counts}"
        )
        return self.optimizer


def main() -> None:
    args = parse_args()
    # Trainer seeds samplers/dropout in Trainer.__init__, but adapters are created
    # before that point.  Seed here so LoRA/AffLoRA initialization is reproducible
    # and the seed recorded in run_args.json describes the whole run.
    set_seed(args.seed)
    print(f"[reproducibility] initialization_seed={args.seed}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_dtype_arg = args.base_dtype
    base_dtype: Any
    if base_dtype_arg == "auto":
        base_dtype = "auto"
    else:
        base_dtype = MASTER_DTYPE_MAP[base_dtype_arg]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=base_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    use_full_finetune = variant_uses_full_finetune(args.variant)
    use_input, use_lm_head = variant_uses_affine(args.variant)
    if use_input or use_lm_head:
        cfg = AffineVocabConfig(
            hidden_size=model.get_input_embeddings().embedding_dim,
            rank=args.affine_rank,
            alpha=args.affine_alpha,
            bias_scale=args.affine_bias_scale,
            dropout=args.affine_dropout,
            use_input=use_input,
            use_lm_head=use_lm_head,
            use_input_bias=not args.no_affine_input_bias,
            use_lm_head_bias=args.affine_lm_head_bias,
            tie_input_lm_head_adapters=args.tie_affine_input_lm_head_adapters,
        )
        apply_affine_vocab_adapters(model, cfg)
        if args.initial_affine_adapter:
            initial_affine_dir = Path(args.initial_affine_adapter)
            state_path = initial_affine_dir / "affine_vocab_adapter.safetensors"
            if not state_path.exists():
                raise FileNotFoundError(state_path)
            state = load_file(str(state_path), device="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            bad_unexpected = [key for key in unexpected if ".affine." in key]
            required = {
                name for name, _ in model.named_parameters() if ".affine." in name
            }
            absent = sorted(required - set(state))
            if bad_unexpected or absent:
                raise RuntimeError(
                    "Initial affine adapter is incompatible with the requested topology: "
                    f"unexpected={bad_unexpected}, missing={absent}, load_missing={missing}"
                )
            print(f"[staged] loaded_initial_affine_adapter={initial_affine_dir}")

    if use_full_finetune:
        for param in model.parameters():
            param.requires_grad_(True)
        print(json.dumps(trainable_summary(model), indent=2))
    elif variant_uses_hidden_lora(args.variant):
        if args.initial_hidden_lora_adapter:
            print(
                "[staged] loading_hidden_lora_adapter="
                f"{args.initial_hidden_lora_adapter} "
                f"freeze={args.freeze_initial_hidden_lora}"
            )
            model = PeftModel.from_pretrained(
                model,
                args.initial_hidden_lora_adapter,
                is_trainable=not args.freeze_initial_hidden_lora,
            )
        else:
            if args.freeze_initial_hidden_lora:
                raise ValueError(
                    "--freeze-initial-hidden-lora requires --initial-hidden-lora-adapter"
                )
            # Affine adapters are constructed before hidden LoRA and consume random
            # numbers. Reset the component seed so paired runs start identically.
            set_seed(args.seed)
            print(f"[reproducibility] hidden_lora_initialization_seed={args.seed}")
            layers_to_transform: list[int] | None = None
            if args.hidden_lora_layers_to_transform:
                layers_to_transform = [
                    int(idx.strip())
                    for idx in args.hidden_lora_layers_to_transform.split(",")
                    if idx.strip()
                ]
            target_modules = [
                m.strip() for m in args.hidden_lora_target_modules.split(",") if m.strip()
            ]
            rank_pattern: dict[str, int] = {}
            alpha_pattern: dict[str, int] = {}
            if args.include_emb_lmh_lora_rank > 0:
                emb_rank = args.include_emb_lmh_lora_rank
                emb_alpha = args.emb_lmh_lora_alpha if args.emb_lmh_lora_alpha is not None else (2 * emb_rank)
                for m in ("embed_tokens", "lm_head"):
                    if m not in target_modules:
                        target_modules.append(m)
                    rank_pattern[m] = emb_rank
                    alpha_pattern[m] = emb_alpha
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.hidden_lora_rank,
                lora_alpha=args.hidden_lora_alpha,
                lora_dropout=args.hidden_lora_dropout,
                target_modules=target_modules,
                layers_to_transform=layers_to_transform,
                rank_pattern=rank_pattern,
                alpha_pattern=alpha_pattern,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        if use_input or use_lm_head:
            affine_count = enable_affine_trainable(model)
            print(f"re-enabled affine vocab trainable params: {affine_count}")
        model.print_trainable_parameters()
    else:
        print(json.dumps(trainable_summary(model), indent=2))

    master_dtype = MASTER_DTYPE_MAP[args.master_dtype]
    cast_stats = cast_trainable_to_master_dtype(model, master_dtype)
    print(f"[precision] cast_to_master_dtype={args.master_dtype} stats={cast_stats}")
    print(json.dumps({"precision_sanity": precision_sanity_report(model, args.master_dtype)}, indent=2))

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_ds = load_any_dataset(args.train_data, args.dataset_split)
    if args.anchor_fraction and args.auxiliary_anchor_lambda > 0:
        raise ValueError(
            "Use either replacement --anchor-data mixing or an auxiliary anchor loss, not both."
        )
    if args.anchor_fraction:
        train_ds = mix_anchor_dataset(train_ds, args)
    elif args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    tokenized = train_ds.map(
        lambda row: tokenize_row(row, tokenizer, args.max_seq_len),
        remove_columns=train_ds.column_names,
        desc="Tokenizing",
    )

    auxiliary_anchor_tokenized = None
    if args.auxiliary_anchor_lambda > 0:
        if not args.auxiliary_anchor_data:
            raise ValueError("--auxiliary-anchor-lambda > 0 requires --auxiliary-anchor-data")
        if args.auxiliary_anchor_batch_size <= 0:
            raise ValueError("--auxiliary-anchor-batch-size must be positive")
        anchor_ds = load_any_dataset(args.auxiliary_anchor_data, args.dataset_split)
        anchor_count = args.auxiliary_anchor_samples or len(anchor_ds)
        if anchor_count > len(anchor_ds):
            raise ValueError(
                "--auxiliary-anchor-samples exceeds the available anchor corpus: "
                f"{anchor_count}/{len(anchor_ds)}"
            )
        anchor_ds = anchor_ds.select_columns(["conversations"])
        anchor_ds = anchor_ds.shuffle(seed=args.anchor_sampling_seed).select(range(anchor_count))
        auxiliary_anchor_tokenized = anchor_ds.map(
            lambda row: tokenize_row(row, tokenizer, args.max_seq_len),
            remove_columns=anchor_ds.column_names,
            desc="Tokenizing auxiliary anchor",
        )
        print(
            "[auxiliary_anchor] "
            f"main_rows={len(tokenized)} anchor_rows={len(auxiliary_anchor_tokenized)} "
            f"lambda={args.auxiliary_anchor_lambda:g} "
            f"batch_size={args.auxiliary_anchor_batch_size} "
            f"source_seed={args.anchor_sampling_seed}",
            flush=True,
        )
    elif args.auxiliary_anchor_data:
        raise ValueError("--auxiliary-anchor-data requires --auxiliary-anchor-lambda > 0")

    eval_tokenized = None
    eval_strategy = "no"
    if args.eval_data and args.eval_samples > 0:
        eval_ds = load_any_dataset(args.eval_data, "test")
        eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))
        eval_tokenized = eval_ds.map(
            lambda row: tokenize_row(row, tokenizer, args.max_seq_len),
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval",
        )
        eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy != "no" else None,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    callbacks: list[TrainerCallback] = []
    if (use_input or use_lm_head) and args.save_strategy != "no":
        callbacks.append(SaveAffineAdapterCallback(model))

    reference_model = None
    if args.reference_run_dir:
        if args.reference_kl_lambda <= 0:
            raise ValueError("--reference-run-dir requires --reference-kl-lambda > 0")
        if (
            args.reference_kl_on_auxiliary_anchor
            or args.reference_kl_on_main_and_auxiliary_anchor
        ) and args.auxiliary_anchor_lambda <= 0:
            raise ValueError(
                "auxiliary-anchor reference KL requires --auxiliary-anchor-lambda > 0"
            )
        reference_dir = Path(args.reference_run_dir)
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=base_dtype, trust_remote_code=args.trust_remote_code
        )
        reference_model = load_affine_vocab_adapter(reference_model, reference_dir)
        if (reference_dir / "adapter_config.json").exists():
            reference_model = PeftModel.from_pretrained(reference_model, reference_dir)
        reference_model.to(training_args.device).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        print(f"[reference_kl] loaded_teacher={reference_dir} lambda={args.reference_kl_lambda:g}")
    elif args.reference_kl_on_auxiliary_anchor or args.reference_kl_on_main_and_auxiliary_anchor:
        raise ValueError("auxiliary-anchor reference KL requires --reference-run-dir")

    trainer = AffineLearningRateTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
        callbacks=callbacks,
        affine_learning_rate_scale=args.affine_learning_rate_scale,
        affine_bias_learning_rate_scale=args.affine_bias_learning_rate_scale,
        affine_energy_lambda=args.affine_energy_lambda,
        affine_energy_tau=args.affine_energy_tau,
        affine_energy_exclude_bias=args.affine_energy_exclude_bias,
        affine_energy_bias_weight=args.affine_energy_bias_weight,
        affine_energy_start_step=args.affine_energy_start_step,
        affine_energy_end_step=args.affine_energy_end_step,
        affine_bias_energy_lambda=args.affine_bias_energy_lambda,
        affine_bias_energy_tau=args.affine_bias_energy_tau,
        reference_model=reference_model,
        reference_kl_lambda=args.reference_kl_lambda,
        reference_kl_temperature=args.reference_kl_temperature,
        reference_kl_on_auxiliary_anchor=args.reference_kl_on_auxiliary_anchor,
        reference_kl_on_main_and_auxiliary_anchor=args.reference_kl_on_main_and_auxiliary_anchor,
        auxiliary_anchor_dataset=auxiliary_anchor_tokenized,
        auxiliary_anchor_lambda=args.auxiliary_anchor_lambda,
        auxiliary_anchor_batch_size=args.auxiliary_anchor_batch_size,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if trainer.is_world_process_zero() and trainer.optimizer is not None:
        adam_state_dtypes: dict[str, int] = {}
        for state in trainer.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                t = state.get(key)
                if isinstance(t, torch.Tensor):
                    name = f"{key}:{t.dtype}"
                    adam_state_dtypes[name] = adam_state_dtypes.get(name, 0) + 1
        print(f"[precision] post_train adam_state_dtypes={adam_state_dtypes}")

    if trainer.is_world_process_zero():
        if args.skip_final_model_save:
            pass
        elif use_input or use_lm_head:
            target = model.get_base_model() if hasattr(model, "get_base_model") else model
            save_affine_vocab_adapter(target, output_dir)
        if use_full_finetune:
            model.save_pretrained(str(output_dir))
        elif variant_uses_hidden_lora(args.variant):
            model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        with (output_dir / "trainable_summary.json").open("w", encoding="utf-8") as f:
            target = model.get_base_model() if hasattr(model, "get_base_model") else model
            json.dump(trainable_summary(target), f, indent=2)


if __name__ == "__main__":
    main()
