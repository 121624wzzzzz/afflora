#!/usr/bin/env python
"""Corrected SFT with vocab-LoRA RNG isolated from hidden-LoRA initialization."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected_sft_experiment"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_affine_vocab_lora as legacy  # noqa: E402
import train_corrected_sft as corrected  # noqa: E402


def argument_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def hidden_lora_sha256(model) -> str:  # noqa: ANN001
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name or "embed_tokens" in name or "lm_head" in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        matched += 1
    if not matched:
        raise RuntimeError("No hidden LoRA parameters found for checksum.")
    print(f"[repro] hidden_only_lora_init_sha256={digest.hexdigest()} tensors={matched}", flush=True)
    return digest.hexdigest()


def install_isolated_vocab_initialization() -> None:
    rank = int(argument_value("--include-emb-lmh-lora-rank") or 0)
    if rank <= 0:
        return
    seed = int(argument_value("--seed") or 42)
    original_get_peft_model = legacy.get_peft_model

    def wrapped_get_peft_model(model, config):  # noqa: ANN001, ANN202
        hidden = int(model.get_input_embeddings().embedding_dim)
        vocab = int(model.get_input_embeddings().num_embeddings)
        original_kaiming = torch.nn.init.kaiming_uniform_
        original_normal = torch.nn.init.normal_
        original_randn = torch.randn

        def isolated_call(function, tensor, *args, **kwargs):  # noqa: ANN001, ANN202
            cpu_state = torch.random.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                torch.manual_seed(seed + 2_000_003)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + 2_000_003)
                return function(tensor, *args, **kwargs)
            finally:
                torch.random.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)

        def kaiming(tensor, *args, **kwargs):  # noqa: ANN001, ANN202
            if tuple(tensor.shape) == (rank, hidden):
                return isolated_call(original_kaiming, tensor, *args, **kwargs)
            return original_kaiming(tensor, *args, **kwargs)

        def normal(tensor, *args, **kwargs):  # noqa: ANN001, ANN202
            if tuple(tensor.shape) == (hidden, rank):
                return isolated_call(original_normal, tensor, *args, **kwargs)
            return original_normal(tensor, *args, **kwargs)

        def randn(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
            if shape in ((rank, vocab), (hidden, rank)):
                cpu_state = torch.random.get_rng_state()
                cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                try:
                    torch.manual_seed(seed + 2_000_003)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + 2_000_003)
                    return original_randn(*args, **kwargs)
                finally:
                    torch.random.set_rng_state(cpu_state)
                    if cuda_states is not None:
                        torch.cuda.set_rng_state_all(cuda_states)
            return original_randn(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming
        torch.nn.init.normal_ = normal
        torch.randn = randn
        try:
            wrapped = original_get_peft_model(model, config)
        finally:
            torch.nn.init.kaiming_uniform_ = original_kaiming
            torch.nn.init.normal_ = original_normal
            torch.randn = original_randn
        hidden_lora_sha256(wrapped)
        return wrapped

    legacy.get_peft_model = wrapped_get_peft_model


def main() -> None:
    install_isolated_vocab_initialization()
    corrected.main()


if __name__ == "__main__":
    main()
