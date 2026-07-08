#!/usr/bin/env python
"""Run the existing trainer with corrected multi-turn tokenization."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import train_affine_vocab_lora as legacy  # noqa: E402
from data_pipeline import tokenize_conversation  # noqa: E402


def corrected_tokenize_row(row, tokenizer, max_seq_len):  # noqa: ANN001, ANN201
    return tokenize_conversation(row, tokenizer, max_seq_len)


def tensor_group_sha256(model, marker: str) -> str:  # noqa: ANN001
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(model.named_parameters()):
        if marker not in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        matched += 1
    if matched == 0:
        raise RuntimeError(f"No parameters matched initialization checksum marker {marker!r}.")
    return digest.hexdigest()


def install_reproducible_initialization(seed: int) -> None:
    """Seed before adapter creation and isolate AffLoRA RNG from PEFT RNG."""

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    original_apply_affine = legacy.apply_affine_vocab_adapters
    original_get_peft_model = legacy.get_peft_model

    def isolated_apply_affine(model, config):  # noqa: ANN001, ANN201
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            # AffLoRA gets a deterministic, separate stream. Restoring the global
            # stream ensures paired baseline/treatment runs create identical LoRA.
            torch.manual_seed(seed + 1_000_003)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 1_000_003)
            return original_apply_affine(model, config)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def checksummed_get_peft_model(model, config):  # noqa: ANN001, ANN201
        wrapped = original_get_peft_model(model, config)
        checksum = tensor_group_sha256(wrapped, "lora_")
        print(f"[repro] seed={seed} hidden_lora_init_sha256={checksum}", flush=True)
        return wrapped

    legacy.apply_affine_vocab_adapters = isolated_apply_affine
    legacy.get_peft_model = checksummed_get_peft_model


class FinalEvalTrainer(legacy.Trainer):
    """Persist a final dev evaluation in addition to scheduled evaluations."""

    def train(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        result = super().train(*args, **kwargs)
        if self.eval_dataset is not None:
            metrics = self.evaluate(metric_key_prefix="final_eval")
            self.log_metrics("final_eval", metrics)
            self.save_metrics("final_eval", metrics)
        return result


def argument_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_pipeline_metadata() -> None:
    output_value = argument_value("--output-dir")
    if not output_value:
        return
    output_dir = Path(output_value)
    run_args_path = output_dir / "run_args.json"
    if not run_args_path.exists():
        return
    run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
    run_args["corrected_data_pipeline"] = {
        "version": 1,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(HERE / "data_pipeline.py"),
        "chat_template": "tokenizer_native",
        "loss_mask": "assistant_content_plus_im_end",
        "initialization": "seed_before_model_build_and_isolated_afflora_rng",
    }
    run_args_path.write_text(
        json.dumps(run_args, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    seed = int(argument_value("--seed") or 42)
    install_reproducible_initialization(seed)
    legacy.tokenize_row = corrected_tokenize_row
    legacy.Trainer = FinalEvalTrainer
    legacy.main()
    add_pipeline_metadata()


if __name__ == "__main__":
    main()
