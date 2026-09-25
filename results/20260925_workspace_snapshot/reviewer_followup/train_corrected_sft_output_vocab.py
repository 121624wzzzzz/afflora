#!/usr/bin/env python
"""Corrected SFT with output-only vocabulary LoRA and isolated initialization.

The legacy ``--include-emb-lmh-lora-rank`` control targets both input
embeddings and the language-model head.  This wrapper keeps the same corrected
data pipeline and initialization controls while restricting that optional
vocabulary-sized LoRA to ``lm_head``.  Hidden-layer LoRA remains unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected_sft_experiment"))
sys.path.insert(0, str(ROOT / "reviewer_followup"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_affine_vocab_lora as legacy  # noqa: E402
import train_corrected_sft as corrected  # noqa: E402
from train_corrected_sft_isolated_vocab import (  # noqa: E402
    install_isolated_vocab_initialization,
)


def install_output_only_vocab_target() -> None:
    """Keep only ``lm_head`` and set its boundary dropout to zero."""

    original_lora_config = legacy.LoraConfig
    original_get_peft_model = legacy.get_peft_model

    def output_only_lora_config(*args: Any, **kwargs: Any):  # noqa: ANN202
        target_modules = kwargs.get("target_modules")
        if target_modules is not None and "lm_head" in target_modules:
            kwargs["target_modules"] = [
                name for name in target_modules if name != "embed_tokens"
            ]
            for pattern_name in ("rank_pattern", "alpha_pattern"):
                pattern = kwargs.get(pattern_name)
                if pattern is not None:
                    kwargs[pattern_name] = {
                        name: value
                        for name, value in pattern.items()
                        if name != "embed_tokens"
                    }
        return original_lora_config(*args, **kwargs)

    def output_only_get_peft_model(model, config):  # noqa: ANN001, ANN202
        wrapped = original_get_peft_model(model, config)
        trainable_names = [
            name for name, parameter in wrapped.named_parameters() if parameter.requires_grad
        ]
        if not any("lm_head" in name and "lora_" in name for name in trainable_names):
            raise RuntimeError("Output-only Vocab-LoRA did not create lm_head LoRA parameters.")
        bad_input = [
            name
            for name in trainable_names
            if "embed_tokens" in name and "lora_" in name
        ]
        if bad_input:
            raise RuntimeError(
                f"Output-only Vocab-LoRA unexpectedly targeted embed_tokens: {bad_input}"
            )

        dropout_modules = []
        for name, module in wrapped.named_modules():
            if not name.endswith("lm_head") or not hasattr(module, "lora_dropout"):
                continue
            for adapter_name in list(module.lora_dropout.keys()):
                module.lora_dropout[adapter_name] = nn.Identity()
            dropout_modules.append(name)
        if len(dropout_modules) != 1:
            raise RuntimeError(
                "Expected exactly one lm_head LoRA dropout module, "
                f"found {dropout_modules}."
            )
        boundary_count = sum(
            parameter.numel()
            for name, parameter in wrapped.named_parameters()
            if parameter.requires_grad and "lm_head" in name and "lora_" in name
        )
        print(
            "[output_only_vocab] "
            f"targets=lm_head boundary_dropout=0 boundary_params={boundary_count} "
            f"dropout_modules={dropout_modules}",
            flush=True,
        )
        return wrapped

    legacy.LoraConfig = output_only_lora_config
    legacy.get_peft_model = output_only_get_peft_model


def main() -> None:
    install_isolated_vocab_initialization()
    install_output_only_vocab_target()
    corrected.main()


if __name__ == "__main__":
    main()
