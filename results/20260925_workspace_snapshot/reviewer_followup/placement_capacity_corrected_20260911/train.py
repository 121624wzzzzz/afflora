#!/usr/bin/env python3
"""Audit P0 initialization and persist final training metrics; use frozen sources."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source/corrected_sft_experiment"))
import train_corrected_sft as corrected
import torch

legacy = corrected.legacy
OriginalTrainer = legacy.AffineLearningRateTrainer


def affine_digest(model):
    digest = hashlib.sha256()
    entries = []
    for name, parameter in model.named_parameters():
        if ".affine." in name and name.endswith(("down.weight", "up.weight")):
            entries.append((name.rsplit(".affine.", 1)[1], parameter))
    for suffix, parameter in sorted(entries):
        digest.update(suffix.encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest() if entries else None


class AuditedTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        variant = corrected.argument_value("--variant")
        hidden_rank = int(corrected.argument_value("--hidden-lora-rank") or 0)
        expected_hidden = legacy.variant_uses_hidden_lora(variant)
        use_input, use_output = legacy.variant_uses_affine(variant)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        embed = base.get_input_embeddings()
        head = base.get_output_embeddings()
        assert hasattr(embed, "affine") == use_input
        assert hasattr(head, "affine") == use_output
        counts = {"hidden": 0, "boundary": 0}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lora_" in name:
                group = "hidden"
            elif ".affine." in name:
                group = "boundary"
            else:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            if parameter.dtype != torch.float32:
                raise RuntimeError(f"Trainable parameter is not FP32: {name}")
            counts[group] += parameter.numel()
        assert bool(counts["hidden"]) == expected_hidden
        if not expected_hidden:
            assert hidden_rank == 0
        d = int(base.config.hidden_size)
        expected_boundary = (32 * d + d if use_input else 32 * d if use_output else 0)
        assert counts["boundary"] == expected_boundary, (counts, expected_boundary)
        audit = {
            "variant": variant, "seed": int(self.args.seed), "hidden_rank": hidden_rank,
            "hidden_init_sha256": corrected.tensor_group_sha256(self.model, "lora_") if expected_hidden else None,
            "affine_init_sha256": affine_digest(self.model), "trainable": counts,
            "total_trainable": sum(counts.values()), "base_tied": bool(base.config.tie_word_embeddings),
            "train_rows": len(self.train_dataset), "torch": torch.__version__,
            "train_batch": self.args.per_device_train_batch_size,
            "gradient_accumulation": self.args.gradient_accumulation_steps,
        }
        self.audit = audit
        Path(self.args.output_dir, "initialization_audit.json").write_text(json.dumps(audit, indent=2))
        print("[p0_audit] " + json.dumps(audit), flush=True)

    def train(self, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = super().train(*args, **kwargs)
        metrics = dict(result.metrics)
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        metrics["global_step"] = self.state.global_step
        self.save_metrics("train", metrics)
        self.save_state()
        return result


if __name__ == "__main__":
    torch.set_num_threads(4)
    legacy.AffineLearningRateTrainer = AuditedTrainer
    corrected.main()
