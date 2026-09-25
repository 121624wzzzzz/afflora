#!/usr/bin/env python
"""Exact centered output-energy diagnostics for Llama-3.1 A-LoRA checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
MODEL = ROOT.parent / "models/Llama-3.1-8B-Base"
RANK_SWEEP = ROOT / "reviewer_followup/llama_affine_rank_sweep"
CROSS = ROOT / "reviewer_followup/llama_cross_version"
DOWNSTREAM = ROOT / "reviewer_followup/llama31_affine_rank_downstream/RESULTS.md"
RANKS = (4, 8, 16, 32)


def checkpoint(rank: int) -> Path:
    root = CROSS / "checkpoints" if rank == 16 else RANK_SWEEP / "checkpoints"
    return root / f"llama31_8b_lmhead_ar{rank}_s1_hr4_seed42"


def task_metrics(rank: int) -> tuple[float, float]:
    name = f"llama31_8b_lmhead_ar{rank}_s1_hr4_seed42_full.json"
    root = CROSS / "outputs" if rank == 16 else ROOT / "reviewer_followup/llama31_affine_rank_downstream/outputs"
    math_report = json.loads((root / "math" / name).read_text())
    gsm_report = json.loads((root / "gsm8k" / name).read_text())
    return float(math_report["clean"]["accuracy_pct"]), float(gsm_report["full"]["accuracy_pct"])


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    weight = model.lm_head.weight.detach()
    vocab, hidden = weight.shape
    mean = weight.float().mean(dim=0)
    denom = 0.0
    chunk_size = 4096
    for begin in range(0, vocab, chunk_size):
        centered = weight[begin : begin + chunk_size].float() - mean
        denom += float(centered.square().sum().item())

    rows = []
    for rank in RANKS:
        state = load_file(str(checkpoint(rank) / "affine_vocab_adapter.safetensors"))
        up = state["lm_head.affine.up.weight"].float().cuda()       # d x r
        down = state["lm_head.affine.down.weight"].float().cuda()  # r x d
        ata = torch.zeros((rank, rank), dtype=torch.float64, device="cuda")
        cross = 0.0
        for begin in range(0, vocab, chunk_size):
            centered = weight[begin : begin + chunk_size].float() - mean
            a = centered @ up
            b = centered @ down.T
            ata += (a.T.double() @ a.double())
            cross += float((a * b).sum().item())
        ddt = down.double() @ down.double().T
        update_energy = float((ata * ddt).sum().item())
        rho_sq = update_energy / denom
        cross_relative = cross / denom
        energy_change = (2.0 * cross + update_energy) / denom

        # Singular values of U D from two thin QR decompositions.
        _, ru = torch.linalg.qr(up.double(), mode="reduced")
        _, rd = torch.linalg.qr(down.T.double(), mode="reduced")
        singular = torch.linalg.svdvals(ru @ rd.T)
        g_fro = float(torch.linalg.vector_norm(singular).item())
        g_spectral = float(singular.max().item())
        math_acc, gsm_acc = task_metrics(rank)
        rows.append({
            "rank": rank,
            "centered_base_energy": denom,
            "centered_update_energy": update_energy,
            "rho_squared": rho_sq,
            "rho": math.sqrt(max(rho_sq, 0.0)),
            "cross_relative": cross_relative,
            "relative_total_energy_change": energy_change,
            "relative_total_energy": 1.0 + energy_change,
            "g_frobenius": g_fro,
            "g_spectral": g_spectral,
            "math_accuracy": math_acc,
            "gsm8k_accuracy": gsm_acc,
        })
    (HERE / "llama31_energy.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# Llama-3.1 A-LoRA centered output-energy diagnostics", "",
        "| rank | rho | rho² | cross/base | total energy Δ | ||G||F | ||G||2 | MATH | GSM8K |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['rho']:.8f} | {row['rho_squared']:.8e} | "
            f"{row['cross_relative']:.8e} | {row['relative_total_energy_change']:+.8e} | "
            f"{row['g_frobenius']:.8f} | {row['g_spectral']:.8f} | "
            f"{row['math_accuracy']:.4f}% | {row['gsm8k_accuracy']:.4f}% |"
        )
    (HERE / "LLAMA31_ENERGY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
