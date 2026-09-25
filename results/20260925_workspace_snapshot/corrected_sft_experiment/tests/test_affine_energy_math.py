"""Regression tests for the codebook geometry used by the energy penalty."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from affine_vocab_lora.adapter import LowRankAffineMap  # noqa: E402
from train_affine_vocab_lora import AffineLearningRateTrainer  # noqa: E402


def _term(weight: torch.Tensor, affine: LowRankAffineMap, geometry: str) -> dict:
    covariance = weight.T @ weight
    row_sum = weight.sum(dim=0)
    return {
        "affine": affine,
        "covariance": covariance,
        "row_sum": row_sum,
        "denominator": covariance.diagonal().sum(),
        "vocab": weight.shape[0],
        "geometry": geometry,
    }


def test_input_energy_matches_actual_row_affine_codebook_update() -> None:
    torch.manual_seed(7)
    weight = torch.randn(19, 11)
    affine = LowRankAffineMap(11, rank=3, alpha=6.0, dropout=0.0, use_bias=True)
    with torch.no_grad():
        affine.up.weight.normal_(std=0.2)
        affine.bias.normal_(std=0.1)
    expected = (affine(weight) - weight).norm() / weight.norm()
    actual = AffineLearningRateTrainer._affine_energy_rho(
        _term(weight, affine, "input_row_codebook")
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_input_energy_can_exclude_affine_translation() -> None:
    torch.manual_seed(17)
    weight = torch.randn(19, 11)
    affine = LowRankAffineMap(11, rank=3, alpha=6.0, dropout=0.0, use_bias=True)
    with torch.no_grad():
        affine.up.weight.normal_(std=0.2)
        affine.bias.normal_(std=0.3)
    linear_update = affine.scale * weight @ affine.down.weight.T @ affine.up.weight.T
    expected = linear_update.norm() / weight.norm()
    actual = AffineLearningRateTrainer._affine_energy_rho(
        _term(weight, affine, "input_row_codebook"), bias_weight=0.0
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_input_energy_can_downweight_affine_translation() -> None:
    torch.manual_seed(27)
    weight = torch.randn(19, 11)
    affine = LowRankAffineMap(11, rank=3, alpha=6.0, dropout=0.0, use_bias=True)
    with torch.no_grad():
        affine.up.weight.normal_(std=0.2)
        affine.bias.normal_(std=0.3)
    weight_factor = 0.5
    linear_update = affine.scale * weight @ affine.down.weight.T @ affine.up.weight.T
    translation = affine.bias_scale * affine.bias
    expected = (linear_update + weight_factor * translation).norm() / weight.norm()
    actual = AffineLearningRateTrainer._affine_energy_rho(
        _term(weight, affine, "input_row_codebook"), bias_weight=weight_factor
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_output_energy_matches_induced_output_codebook_update() -> None:
    torch.manual_seed(8)
    weight = torch.randn(23, 11)
    affine = LowRankAffineMap(11, rank=3, alpha=6.0, dropout=0.0, use_bias=False)
    with torch.no_grad():
        affine.up.weight.normal_(std=0.2)
    induced = weight + affine.scale * weight @ affine.up.weight @ affine.down.weight
    expected = (induced - weight).norm() / weight.norm()
    actual = AffineLearningRateTrainer._affine_energy_rho(
        _term(weight, affine, "output_codebook")
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
