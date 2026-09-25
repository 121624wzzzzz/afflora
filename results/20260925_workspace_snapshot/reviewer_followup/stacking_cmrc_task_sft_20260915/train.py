"""Frozen corrected trainer with paired initialization and budget controls."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source/corrected_sft_experiment"))
import train_corrected_sft as corrected
import torch
from budget import add_hidden_budget, hidden_digest

legacy = corrected.legacy
OriginalTrainer = legacy.AffineLearningRateTrainer
BUDGET = {}


def affine_digest(module):
    digest = hashlib.sha256()
    for suffix in ("down.weight", "up.weight"):
        value = module.get_parameter(suffix)
        digest.update(suffix.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def install_controls():
    original_apply = legacy.apply_affine_vocab_adapters
    original_get = legacy.get_peft_model

    def apply(model, config):
        result = original_apply(model, config)
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        if hasattr(embed, "affine") and hasattr(head, "affine"):
            # Independent modules, identical initialization to each P0 one-sided arm.
            with torch.no_grad():
                head.affine.down.weight.copy_(embed.affine.down.weight)
                head.affine.up.weight.copy_(embed.affine.up.weight)
            assert head.affine.down.weight is not embed.affine.down.weight
        return result

    def get(model, config):
        wrapped = original_get(model, config)
        if os.environ.get("STACKING_ARM") == "hidden_budget":
            BUDGET.update(add_hidden_budget(wrapped, config.r, int(corrected.argument_value("--seed"))))
        return wrapped

    legacy.apply_affine_vocab_adapters = apply
    legacy.get_peft_model = get


class AuditedTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rank = int(corrected.argument_value("--hidden-lora-rank"))
        arm = os.environ["STACKING_ARM"]
        variant = corrected.argument_value("--variant")
        inp, out = legacy.variant_uses_affine(variant)
        base = self.model.get_base_model()
        embed, head = base.get_input_embeddings(), base.get_output_embeddings()
        assert hasattr(embed, "affine") == inp and hasattr(head, "affine") == out
        counts = {"hidden": 0, "boundary": 0}
        for name, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            group = "hidden" if "lora_" in name else "boundary" if ".affine." in name else None
            assert group is not None, name
            assert value.dtype == torch.float32, name
            counts[group] += value.numel()
        d = int(base.config.hidden_size)
        assert counts["boundary"] == ((33 * d if inp else 0) + (32 * d if out else 0))
        one_rank = 0
        for layer in self.model.modules():
            if hasattr(layer, "lora_A") and "default" in layer.lora_A:
                one_rank += layer.in_features + layer.out_features
        assert counts["hidden"] == rank * one_rank + BUDGET.get("actual_extra_parameters", 0)
        shared = hidden_digest(self.model, rank)
        if BUDGET:
            assert shared == BUDGET["shared_initialization_sha256"]
        audit = {"arm": arm, "variant": variant, "seed": int(self.args.seed),
                 "hidden_rank": rank, "hidden_init_sha256": hidden_digest(self.model),
                 "shared_hidden_init_sha256": shared, "budget_control": BUDGET,
                 "affine_components": {key: affine_digest(mod.affine) for key, mod in
                     (("input", embed), ("output", head)) if hasattr(mod, "affine")},
                 "trainable": counts, "total_trainable": sum(counts.values()),
                 "base_tied": bool(base.config.tie_word_embeddings), "train_rows": len(self.train_dataset),
                 "train_batch": self.args.per_device_train_batch_size,
                 "gradient_accumulation": self.args.gradient_accumulation_steps,
                 "torch": torch.__version__}
        Path(self.args.output_dir, "initialization_audit.json").write_text(json.dumps(audit, indent=2))
        print("[stacking_audit] " + json.dumps(audit), flush=True)

    def train(self, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = super().train(*args, **kwargs)
        metrics = dict(result.metrics)
        metrics.update(global_step=self.state.global_step,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        self.save_metrics("train", metrics)
        self.save_state()
        if self.args.max_steps == 2:
            item=self.train_dataset[0]
            ids=torch.tensor([item['input_ids']],device=self.args.device)
            attention=torch.ones_like(ids)
            self.model.eval()
            with torch.inference_mode():
                amp_logits=self.model(input_ids=ids,attention_mask=attention).logits[:,-3:,:].float().cpu()
            # Remove the trainer's implicit AMP wrapper before recording the
            # ordinary independently loaded inference path. No parameters change.
            plain_model=self.accelerator.unwrap_model(self.model,keep_fp32_wrapper=False)
            with torch.inference_mode():
                logits=plain_model(input_ids=ids,attention_mask=attention).logits[:,-3:,:].float().cpu()
            digest=hashlib.sha256()
            for name,value in sorted(plain_model.named_parameters()):
                if 'lora_' in name or '.affine.' in name:
                    digest.update(name.encode());digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
            torch.save({'input_ids':ids.cpu(),'attention_mask':attention.cpu(),'last_logits':logits,
                        'amp_logits':amp_logits,'adapter_tensor_sha256':digest.hexdigest()},
                       Path(self.args.output_dir)/'smoke_forward.pt')
        return result


if __name__ == "__main__":
    torch.set_num_threads(4)
    install_controls()
    legacy.AffineLearningRateTrainer = AuditedTrainer
    corrected.main()
