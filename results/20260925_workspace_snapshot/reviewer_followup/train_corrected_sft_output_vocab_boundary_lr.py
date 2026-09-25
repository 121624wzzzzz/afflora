#!/usr/bin/env python
"""Output-only Vocab-LoRA SFT with an independently scaled boundary LR.

This is a deliberately narrow wrapper for the strict output-only equal-budget
experiment.  It composes the existing corrected-data, isolated-initialization,
and output-only-target wrappers without modifying the legacy trainer.

``--learning-rate`` remains the hidden-LoRA learning rate and is required to be
exactly 2e-4.  The additional ``--output-vocab-lr-scale`` flag multiplies only
the two trainable ``lm_head`` LoRA matrices.  Optimizer construction performs
strict name, shape, coverage, and parameter-group assertions before training.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected_sft_experiment"))
sys.path.insert(0, str(ROOT / "reviewer_followup"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_affine_vocab_lora as legacy  # noqa: E402
import train_corrected_sft as corrected  # noqa: E402
from train_corrected_sft_isolated_vocab import (  # noqa: E402
    install_isolated_vocab_initialization,
)
from train_corrected_sft_output_vocab import (  # noqa: E402
    install_output_only_vocab_target,
)


BOUNDARY_LR_FLAG = "--output-vocab-lr-scale"
REQUIRED_HIDDEN_LR = 2e-4
OPTIMIZER_AUDIT: dict[str, Any] = {}


def pop_float_flag(flag: str, default: float) -> float:
    """Remove one wrapper-only float flag before the legacy parser runs."""

    values: list[str] = []
    retained = [sys.argv[0]]
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        if argument == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a value")
            values.append(sys.argv[index + 1])
            index += 2
            continue
        if argument.startswith(f"{flag}="):
            values.append(argument.split("=", 1)[1])
            index += 1
            continue
        retained.append(argument)
        index += 1
    if len(values) > 1:
        raise ValueError(f"{flag} was specified more than once: {values}")
    sys.argv[:] = retained
    value = float(values[0]) if values else float(default)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{flag} must be finite and positive, got {value}")
    return value


def _is_output_boundary_lora(name: str) -> bool:
    is_lm_head = ".lm_head." in name or name.startswith("lm_head.")
    return is_lm_head and (
        ".lora_A." in name
        or ".lora_B." in name
        or name.endswith(".lora_A.weight")
        or name.endswith(".lora_B.weight")
    )


def _boundary_role(name: str) -> str | None:
    if ".lora_A." in name or name.endswith(".lora_A.weight"):
        return "A"
    if ".lora_B." in name or name.endswith(".lora_B.weight"):
        return "B"
    return None


def install_output_boundary_lr_trainer(boundary_lr_scale: float) -> None:
    """Monkeypatch the trainer class used by ``legacy.main``."""

    original_trainer = legacy.AffineLearningRateTrainer

    class OutputBoundaryLearningRateTrainer(original_trainer):
        """Create disjoint hidden and output-boundary optimizer groups."""

        def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
            if self.optimizer is not None:
                return self.optimizer

            opt_model = self.model if model is None else model
            base_lr = float(self.args.learning_rate)
            if not math.isclose(
                base_lr, REQUIRED_HIDDEN_LR, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    "This experiment requires hidden LoRA LR=2e-4, "
                    f"but --learning-rate resolved to {base_lr:g}"
                )
            if (
                self.affine_learning_rate_scale != 1.0
                or self.affine_bias_learning_rate_scale != 1.0
            ):
                raise RuntimeError(
                    "Output-only Vocab-LoRA must not also request affine LR scales"
                )

            trainable = [
                (name, parameter)
                for name, parameter in opt_model.named_parameters()
                if parameter.requires_grad
            ]
            if not trainable:
                raise RuntimeError("No trainable parameters were found")

            boundary = [
                (name, parameter)
                for name, parameter in trainable
                if _is_output_boundary_lora(name)
            ]
            hidden = [
                (name, parameter)
                for name, parameter in trainable
                if not _is_output_boundary_lora(name)
            ]

            input_boundary = [
                name
                for name, _ in trainable
                if "embed_tokens" in name and "lora_" in name
            ]
            if input_boundary:
                raise RuntimeError(
                    "Output-only run contains input-embedding LoRA parameters: "
                    f"{input_boundary}"
                )

            unexpected_hidden = [
                name
                for name, _ in hidden
                if "lora_" not in name
                or ".lm_head." in name
                or name.startswith("lm_head.")
                or "embed_tokens" in name
            ]
            if unexpected_hidden:
                raise RuntimeError(
                    "Trainable parameters outside hidden LoRA were found: "
                    f"{unexpected_hidden}"
                )

            roles: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
                "A": [],
                "B": [],
            }
            for item in boundary:
                role = _boundary_role(item[0])
                if role is None:
                    raise RuntimeError(
                        f"Unrecognized output-boundary parameter: {item[0]}"
                    )
                roles[role].append(item)
            if len(roles["A"]) != 1 or len(roles["B"]) != 1:
                raise RuntimeError(
                    "Expected exactly one output LoRA A and one output LoRA B; "
                    f"found A={[name for name, _ in roles['A']]} "
                    f"B={[name for name, _ in roles['B']]}"
                )
            if not hidden:
                raise RuntimeError("No hidden LoRA parameters were found")

            a_name, a_parameter = roles["A"][0]
            b_name, b_parameter = roles["B"][0]
            if a_parameter.ndim != 2 or b_parameter.ndim != 2:
                raise RuntimeError(
                    "Output LoRA A/B must both be matrices: "
                    f"{a_name}={tuple(a_parameter.shape)}, "
                    f"{b_name}={tuple(b_parameter.shape)}"
                )
            if a_parameter.shape[0] != b_parameter.shape[1]:
                raise RuntimeError(
                    "Output LoRA rank mismatch: "
                    f"{a_name}={tuple(a_parameter.shape)}, "
                    f"{b_name}={tuple(b_parameter.shape)}"
                )

            decay_names = set(self.get_decay_parameter_names(opt_model))
            boundary_lr = base_lr * boundary_lr_scale
            grouped: list[dict[str, Any]] = []
            group_counts: dict[str, int] = {}

            for group_name, members, group_lr in (
                ("hidden_lora", hidden, base_lr),
                ("output_vocab_lora", boundary, boundary_lr),
            ):
                for use_decay in (True, False):
                    parameters = [
                        parameter
                        for name, parameter in members
                        if (name in decay_names) == use_decay
                    ]
                    if not parameters:
                        continue
                    optimizer_group_name = (
                        f"{group_name}_{'decay' if use_decay else 'no_decay'}"
                    )
                    grouped.append(
                        {
                            "params": parameters,
                            "weight_decay": (
                                self.args.weight_decay if use_decay else 0.0
                            ),
                            "lr": group_lr,
                            "group_name": optimizer_group_name,
                        }
                    )
                    group_counts[optimizer_group_name] = sum(
                        parameter.numel() for parameter in parameters
                    )

            expected_ids = {id(parameter) for _, parameter in trainable}
            grouped_parameters = [
                parameter
                for group in grouped
                for parameter in group["params"]
            ]
            grouped_ids = [id(parameter) for parameter in grouped_parameters]
            if len(grouped_ids) != len(set(grouped_ids)):
                raise RuntimeError("An optimizer parameter appears in multiple groups")
            if set(grouped_ids) != expected_ids:
                missing = [
                    name
                    for name, parameter in trainable
                    if id(parameter) not in set(grouped_ids)
                ]
                raise RuntimeError(
                    "Optimizer groups do not exactly cover trainable parameters; "
                    f"missing={missing} expected={len(expected_ids)} "
                    f"grouped={len(set(grouped_ids))}"
                )

            hidden_count = sum(parameter.numel() for _, parameter in hidden)
            boundary_count = sum(parameter.numel() for _, parameter in boundary)
            total_count = sum(parameter.numel() for _, parameter in trainable)
            if hidden_count + boundary_count != total_count:
                raise RuntimeError(
                    "Hidden/boundary parameter counts do not sum to total: "
                    f"{hidden_count}+{boundary_count}!={total_count}"
                )

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)

            OPTIMIZER_AUDIT.clear()
            OPTIMIZER_AUDIT.update(
                {
                    "hidden_learning_rate": base_lr,
                    "output_vocab_learning_rate": boundary_lr,
                    "output_vocab_lr_scale": boundary_lr_scale,
                    "hidden_lora_params": hidden_count,
                    "output_vocab_lora_params": boundary_count,
                    "total_trainable_params": total_count,
                    "output_lora_rank": int(a_parameter.shape[0]),
                    "output_lora_a_shape": list(a_parameter.shape),
                    "output_lora_b_shape": list(b_parameter.shape),
                    "optimizer_group_params": group_counts,
                    "coverage_assertion": "passed",
                }
            )
            print(
                "[optimizer] output_vocab_boundary_lr "
                + json.dumps(OPTIMIZER_AUDIT, sort_keys=True),
                flush=True,
            )
            return self.optimizer

    legacy.AffineLearningRateTrainer = OutputBoundaryLearningRateTrainer


def _record_optimizer_audit() -> None:
    """Persist the wrapper-only argument and verified grouping beside run args."""

    output_value = corrected.argument_value("--output-dir")
    if not output_value:
        raise RuntimeError("--output-dir is required")
    if not OPTIMIZER_AUDIT:
        raise RuntimeError("Optimizer audit was not populated during training")
    output_dir = Path(output_value)
    audit_path = output_dir / "optimizer_groups.json"
    audit_path.write_text(
        json.dumps(OPTIMIZER_AUDIT, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_args_path = output_dir / "run_args.json"
    if not run_args_path.exists():
        raise RuntimeError(f"Missing completed-run metadata: {run_args_path}")
    run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
    run_args["output_vocab_lr"] = dict(OPTIMIZER_AUDIT)
    run_args["output_vocab_lr"]["implementation"] = str(Path(__file__).resolve())
    run_args_path.write_text(
        json.dumps(run_args, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    boundary_lr_scale = pop_float_flag(BOUNDARY_LR_FLAG, default=1.0)
    install_isolated_vocab_initialization()
    install_output_only_vocab_target()
    install_output_boundary_lr_trainer(boundary_lr_scale)
    corrected.main()
    _record_optimizer_audit()


if __name__ == "__main__":
    main()
