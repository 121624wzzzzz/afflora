#!/usr/bin/env python
"""Rank-generalized entry point for the audited fixed-hidden A-LoRA control.

The numerical implementation remains in
``train_corrected_sft_fixed_hidden_boundary_fp32.py``.  This wrapper changes
only the predeclared A-LoRA rank and keeps the functional scale fixed.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import train_corrected_sft_fixed_hidden_boundary_fp32 as base


RANK_FLAG = "--sweep-alora-rank"
SCALE_FLAG = "--sweep-alora-scale"
MARKER = "fixed_hidden_output_boundary_fp32_rank_sweep_v1"
ALLOWED_RANKS = {2, 4, 8, 16, 32, 50}


def pop_value(flag: str, cast):  # noqa: ANN001, ANN201
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
    if len(values) != 1:
        raise ValueError(f"{flag} must be specified exactly once")
    sys.argv[:] = retained
    return cast(values[0])


def main() -> None:
    rank = pop_value(RANK_FLAG, int)
    scale = pop_value(SCALE_FLAG, float)
    if rank not in ALLOWED_RANKS:
        raise ValueError(f"rank must be one of {sorted(ALLOWED_RANKS)}, got {rank}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {scale}")

    expected_count = 2 * base.EXPECTED_HIDDEN_SIZE * rank
    original_validate_args = base._validate_args

    def validate_args(args, kind: str) -> None:  # noqa: ANN001
        if kind != "alora":
            raise RuntimeError("The rank sweep supports only A-LoRA")
        # Reuse every safety check in the audited entry point.  Its sole
        # rank-specific check expects 50, so satisfy that check on a copy and
        # validate the real sweep rank separately below.
        proxy = copy.copy(args)
        proxy.affine_rank = 50
        original_validate_args(proxy, kind)
        if args.affine_rank != rank:
            raise RuntimeError(
                f"--affine-rank={args.affine_rank}, expected sweep rank {rank}"
            )
        actual_scale = float(args.affine_alpha) / rank
        if not math.isclose(actual_scale, scale, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"functional scale={actual_scale}, expected predeclared {scale}"
            )

    def assert_sweep_budget(boundary) -> int:  # noqa: ANN001
        if boundary.kind != "alora":
            raise RuntimeError(f"Expected A-LoRA boundary, got {boundary.kind}")
        if boundary.hidden_size != base.EXPECTED_HIDDEN_SIZE:
            raise RuntimeError("Unexpected hidden size")
        if boundary.vocab_size != base.EXPECTED_VOCAB_SIZE:
            raise RuntimeError("Unexpected vocabulary size")
        if boundary.rank != rank:
            raise RuntimeError(f"Boundary rank={boundary.rank}, expected {rank}")
        count = sum(parameter.numel() for parameter in boundary.affine.parameters())
        if count != expected_count:
            raise RuntimeError(
                f"Boundary count={count}, expected {expected_count}"
            )
        return count

    base.EXPECTED_COUNTS["alora"] = expected_count
    base.CONTROL_MARKER = MARKER
    base._validate_args = validate_args
    base._assert_qwen_budget = assert_sweep_budget
    base.WRAPPER_OPTIONS.update(
        {
            "sweep_rank": rank,
            "sweep_scale": scale,
            "sweep_expected_count": expected_count,
        }
    )
    # The terminal audit must bind to this wrapper as well as the reused
    # numerical implementation.
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
