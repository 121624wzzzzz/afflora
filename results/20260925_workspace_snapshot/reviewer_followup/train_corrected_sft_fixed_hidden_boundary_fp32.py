#!/usr/bin/env python
"""Fixed-hidden, output-boundary-only FP32 control for corrected SFT.

This deliberately narrow entry point reuses the corrected SFT data pipeline
while removing three possible confounds from the equal-budget comparison:

* a *completed* hidden-LoRA checkpoint is loaded and tensor-audited;
* every hidden parameter is frozen and every hidden LoRA dropout is replaced
  by ``Identity`` before the boundary stage starts;
* A-LoRA and direct Vocab-LoRA use the same native-precision frozen base-logit
  path, plus separately computed FP32 residuals with autocast disabled.

The two supported boundaries are:

``alora``
    ``z0 = linear(x_native, W_native).float()``
    ``logits = z0 + linear(scale * U(D(x_fp32)), W_fp32)`` (rank 50)

``vocab_lora``
    ``z0 = linear(x_native, W_native).float()``
    ``logits = z0 + scale * B(A(x_fp32))`` (rank 1)

For Qwen2.5-1.5B (hidden=1536, vocab=151936), these contain exactly 153600
and 153472 trainable parameters respectively.  There is no boundary bias,
dropout, input adapter, gradient clipping, or trainable hidden parameter.

The implementation monkeypatches only symbols imported by the legacy trainer;
the shared training and adapter implementations are not modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file, save_file
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected_sft_experiment"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_affine_vocab_lora as legacy  # noqa: E402
import train_corrected_sft as corrected  # noqa: E402


KIND_FLAG = "--fixed-boundary-kind"
TF32_FLAG = "--fixed-boundary-tf32"
CONTROL_MARKER = "fixed_hidden_output_boundary_fp32_v1"
EXPECTED_HIDDEN_SIZE = 1536
EXPECTED_VOCAB_SIZE = 151936
EXPECTED_COUNTS = {"alora": 153600, "vocab_lora": 153472}
BOUNDARY_FILENAME = "fixed_boundary_adapter.safetensors"
CONFIG_FILENAME = "fixed_boundary_config.json"
AUDIT_FILENAME = "fixed_hidden_boundary_audit.json"

BOUNDARY_MODULE: "FixedOutputBoundaryLMHead | None" = None
CONTROL_AUDIT: dict[str, Any] = {}
WRAPPER_OPTIONS: dict[str, Any] = {}


def _pop_choice_flag(flag: str, choices: Iterable[str], default: str | None = None) -> str:
    """Remove one wrapper-only flag before the legacy argparse parser runs."""

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
    if not values and default is None:
        raise ValueError(f"{flag} is required")
    value = values[0] if values else str(default)
    allowed = set(choices)
    if value not in allowed:
        raise ValueError(f"{flag} must be one of {sorted(allowed)}, got {value!r}")
    sys.argv[:] = retained
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hidden_key(name: str) -> str:
    """Normalize PEFT checkpoint/model prefixes for tensor-exact comparison."""

    if "layers." not in name:
        raise RuntimeError(f"Unexpected hidden-LoRA tensor name without layers.*: {name}")
    key = "layers." + name.split("layers.", 1)[1]
    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")
    key = key.replace(".lora_embedding_A.default", ".lora_embedding_A")
    key = key.replace(".lora_embedding_B.default", ".lora_embedding_B")
    return key


def _tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sha256_tensor_rows(tensor: torch.Tensor, rows_per_chunk: int = 4096) -> str:
    """Hash a potentially huge matrix without materializing one giant bytes copy."""

    value = tensor.detach()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    if value.ndim == 0:
        chunks = [value.reshape(1)]
    else:
        chunks = (
            value[begin : begin + rows_per_chunk]
            for begin in range(0, value.shape[0], rows_per_chunk)
        )
    for chunk in chunks:
        digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hidden_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    state_path = adapter_dir / "adapter_model.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = load_file(str(state_path), device="cpu")
    bad = [
        name
        for name in state
        if "lora_" not in name or "lm_head" in name or "embed_tokens" in name
    ]
    if bad:
        raise RuntimeError(
            "The fixed-hidden source must be a pure hidden-LoRA checkpoint; "
            f"unexpected tensors={bad[:8]}"
        )
    canonical = {_canonical_hidden_key(name): tensor for name, tensor in state.items()}
    if len(canonical) != len(state):
        raise RuntimeError("Canonical hidden-LoRA checkpoint names collided")
    return canonical


def model_hidden_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    selected = {
        _canonical_hidden_key(name): parameter.detach()
        for name, parameter in model.named_parameters()
        if "lora_" in name and "lm_head" not in name and "embed_tokens" not in name
    }
    if not selected:
        raise RuntimeError("No hidden-LoRA parameters were loaded")
    return selected


def audit_hidden_tensor_identity(model: nn.Module, adapter_dir: Path) -> dict[str, Any]:
    """Fail closed unless the runtime hidden adapter equals its source tensor-wise."""

    source = source_hidden_tensors(adapter_dir)
    runtime = model_hidden_tensors(model)
    if source.keys() != runtime.keys():
        raise RuntimeError(
            "Loaded hidden-LoRA tensor names differ from source: "
            f"missing={sorted(source.keys() - runtime.keys())[:8]} "
            f"unexpected={sorted(runtime.keys() - source.keys())[:8]}"
        )
    mismatches = [
        name
        for name in source
        if source[name].shape != runtime[name].shape
        or not torch.equal(source[name].float().cpu(), runtime[name].float().cpu())
    ]
    if mismatches:
        raise RuntimeError(
            "Loaded hidden-LoRA tensors are not source-identical: "
            f"{mismatches[:8]}"
        )
    source_digest = _tensor_digest(source)
    runtime_digest = _tensor_digest(runtime)
    if source_digest != runtime_digest:
        raise RuntimeError(
            f"Hidden tensor digest mismatch: source={source_digest} runtime={runtime_digest}"
        )
    return {
        "source_adapter_dir": str(adapter_dir.resolve()),
        "source_adapter_file_sha256": sha256_file(
            adapter_dir / "adapter_model.safetensors"
        ),
        "canonical_tensor_sha256": source_digest,
        "tensor_count": len(source),
        "parameter_count": sum(tensor.numel() for tensor in source.values()),
        "tensor_exact_assertion": "passed",
    }


def disable_hidden_lora_dropout(model: nn.Module) -> list[str]:
    """Make a frozen hidden feature extractor deterministic during Trainer.train()."""

    replaced: list[str] = []
    for module_name, module in model.named_modules():
        dropouts = getattr(module, "lora_dropout", None)
        if not isinstance(dropouts, nn.ModuleDict):
            continue
        if "lm_head" in module_name or "embed_tokens" in module_name:
            raise RuntimeError(
                "The pure hidden checkpoint unexpectedly contains a vocabulary LoRA: "
                f"{module_name}"
            )
        for adapter_name in list(dropouts.keys()):
            dropouts[adapter_name] = nn.Identity()
            replaced.append(f"{module_name}.lora_dropout.{adapter_name}")
    if not replaced:
        raise RuntimeError("No hidden LoRA dropout modules were found to disable")
    remaining = []
    for module_name, module in model.named_modules():
        dropouts = getattr(module, "lora_dropout", None)
        if not isinstance(dropouts, nn.ModuleDict):
            continue
        remaining.extend(
            f"{module_name}.lora_dropout.{adapter_name}"
            for adapter_name, dropout in dropouts.items()
            if not isinstance(dropout, nn.Identity)
        )
    if remaining:
        raise RuntimeError(f"Non-identity hidden LoRA dropout remains: {remaining[:8]}")
    return replaced


@dataclass(frozen=True)
class FixedBoundaryConfig:
    marker: str
    kind: str
    hidden_size: int
    vocab_size: int
    rank: int
    alpha: float
    scale: float
    trainable_parameters: int
    boundary_dtype: str
    output_dtype: str
    autocast_disabled: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    input_adapter: bool
    boundary_bias: bool
    boundary_dropout: float
    frozen_lm_head_weight_dtype: str
    frozen_native_weight_sha256: str
    frozen_fp32_weight_sha256: str
    common_base_path: str
    residual_path: str
    zero_initialized_residual: bool


class BoundaryFactors(nn.Module):
    """Two bias-free FP32 matrices with standard LoRA initialization."""

    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int,
        rank: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(
            hidden_size, rank, bias=False, device=device, dtype=torch.float32
        )
        self.up = nn.Linear(
            rank, output_size, bias=False, device=device, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)


class FixedOutputBoundaryLMHead(nn.Module):
    """Common frozen FP32 base projection plus one trainable output boundary."""

    def __init__(
        self,
        base_head: nn.Linear,
        *,
        kind: str,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if kind not in EXPECTED_COUNTS:
            raise ValueError(f"Unsupported boundary kind: {kind}")
        if base_head.bias is not None:
            raise RuntimeError("This control requires a bias-free frozen base lm_head")
        self.kind = kind
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.hidden_size = int(base_head.in_features)
        self.vocab_size = int(base_head.out_features)
        self.frozen_lm_head_weight_dtype = str(base_head.weight.dtype)
        # Both buffers are non-persistent: they are deterministic frozen views of
        # the base model and must never inflate adapter checkpoints.  Keeping a
        # native clone makes the common base projection literally identical in
        # both treatments; the FP32 shadow is used only by A-LoRA's residual.
        self.register_buffer(
            "_frozen_weight_native",
            base_head.weight.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "_frozen_weight_fp32",
            self._frozen_weight_native.detach().to(dtype=torch.float32).clone(),
            persistent=False,
        )
        self.frozen_native_weight_sha256 = _sha256_tensor_rows(
            self._frozen_weight_native
        )
        self.frozen_fp32_weight_sha256 = _sha256_tensor_rows(
            self._frozen_weight_fp32
        )
        output_size = self.hidden_size if kind == "alora" else self.vocab_size
        self.affine = BoundaryFactors(
            hidden_size=self.hidden_size,
            output_size=output_size,
            rank=self.rank,
            device=base_head.weight.device,
        )
        self.zero_initialized_residual_at_construction = bool(
            torch.count_nonzero(self.affine.up.weight.detach()).item() == 0
        )
        if not self.zero_initialized_residual_at_construction:
            raise RuntimeError("Boundary residual did not initialize to exact zero")
        # Do not retain the original nn.Linear as a submodule: its weight is tied to
        # input embeddings, and the non-persistent FP32 buffer is the sole output
        # projection used by both treatments.
        base_head.weight.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility with transformers' output-embedding interface."""

        return self._frozen_weight_native

    @property
    def bias(self) -> None:
        return None

    @property
    def in_features(self) -> int:
        return self.hidden_size

    @property
    def out_features(self) -> int:
        return self.vocab_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The common native base is intentionally evaluated by the exact same
        # operation in both treatments.  Explicitly disabling autocast prevents
        # ambient Trainer state from changing either this path or the FP32
        # residual path.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            base_logits = F.linear(
                hidden_states.to(dtype=self._frozen_weight_native.dtype),
                self._frozen_weight_native,
            ).float()
            x = hidden_states.float()
            if self.kind == "alora":
                delta_hidden = self.affine.up(self.affine.down(x))
                delta_logits = F.linear(
                    self.scale * delta_hidden, self._frozen_weight_fp32
                )
            else:
                delta_logits = self.scale * self.affine.up(self.affine.down(x))
            return base_logits + delta_logits


def _base_causal_model(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def find_fixed_boundary(model: nn.Module) -> FixedOutputBoundaryLMHead:
    matches = [
        module
        for module in model.modules()
        if isinstance(module, FixedOutputBoundaryLMHead)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one fixed output boundary module, found {len(matches)}"
        )
    return matches[0]


def _assert_qwen_budget(boundary: FixedOutputBoundaryLMHead) -> int:
    if boundary.hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"Expected hidden size {EXPECTED_HIDDEN_SIZE}, got {boundary.hidden_size}"
        )
    if boundary.vocab_size != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocab size {EXPECTED_VOCAB_SIZE}, got {boundary.vocab_size}"
        )
    expected_rank = 50 if boundary.kind == "alora" else 1
    if boundary.rank != expected_rank:
        raise RuntimeError(
            f"{boundary.kind} requires rank {expected_rank}, got {boundary.rank}"
        )
    count = sum(parameter.numel() for parameter in boundary.affine.parameters())
    expected_count = EXPECTED_COUNTS[boundary.kind]
    if count != expected_count:
        raise RuntimeError(
            f"{boundary.kind} boundary count mismatch: {count} != {expected_count}"
        )
    return count


def install_fixed_boundary(
    model: nn.Module,
    *,
    kind: str,
    rank: int,
    alpha: float,
    enforce_qwen_budget: bool = True,
) -> FixedOutputBoundaryLMHead:
    target = _base_causal_model(model)
    base_head = target.get_output_embeddings()
    if not isinstance(base_head, nn.Linear):
        raise TypeError(
            "Expected the unmodified output embedding to be nn.Linear, "
            f"got {type(base_head).__name__}"
        )
    input_before = target.get_input_embeddings()
    input_weight_ptr = input_before.weight.data_ptr()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    boundary = FixedOutputBoundaryLMHead(
        base_head, kind=kind, rank=rank, alpha=alpha
    )
    target.set_output_embeddings(boundary)
    # Retain this attribute only so the legacy callback recognizes an affine run.
    target.affine_vocab_config = {
        "marker": CONTROL_MARKER,
        "kind": kind,
    }
    if target.get_input_embeddings() is not input_before:
        raise RuntimeError("Installing the output boundary changed input embeddings")
    if target.get_input_embeddings().weight.data_ptr() != input_weight_ptr:
        raise RuntimeError("Installing the output boundary changed input-embedding storage")
    if enforce_qwen_budget:
        _assert_qwen_budget(boundary)
    return boundary


def boundary_config(boundary: FixedOutputBoundaryLMHead) -> FixedBoundaryConfig:
    count = sum(parameter.numel() for parameter in boundary.affine.parameters())
    return FixedBoundaryConfig(
        marker=CONTROL_MARKER,
        kind=boundary.kind,
        hidden_size=boundary.hidden_size,
        vocab_size=boundary.vocab_size,
        rank=boundary.rank,
        alpha=boundary.alpha,
        scale=boundary.scale,
        trainable_parameters=count,
        boundary_dtype="torch.float32",
        output_dtype="torch.float32",
        autocast_disabled=True,
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        input_adapter=False,
        boundary_bias=False,
        boundary_dropout=0.0,
        frozen_lm_head_weight_dtype=boundary.frozen_lm_head_weight_dtype,
        frozen_native_weight_sha256=boundary.frozen_native_weight_sha256,
        frozen_fp32_weight_sha256=boundary.frozen_fp32_weight_sha256,
        common_base_path="autocast_off_F.linear(hidden_native,W_native).float",
        residual_path=(
            "autocast_off_FP32_F.linear(scale*U(D(hidden_fp32)),W_fp32)"
            if boundary.kind == "alora"
            else "autocast_off_FP32_scale*B(A(hidden_fp32))"
        ),
        zero_initialized_residual=boundary.zero_initialized_residual_at_construction,
    )


def save_fixed_boundary_adapter(model: nn.Module, output_dir: str | Path) -> None:
    """Save only the two boundary matrices plus a fail-closed manifest."""

    boundary = find_fixed_boundary(model)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        f"affine.{name}": tensor.detach().float().cpu().contiguous()
        for name, tensor in boundary.affine.state_dict().items()
    }
    if set(state) != {"affine.down.weight", "affine.up.weight"}:
        raise RuntimeError(f"Unexpected fixed-boundary state keys: {sorted(state)}")
    save_file(state, str(output / BOUNDARY_FILENAME))
    payload = asdict(boundary_config(boundary))
    payload["hidden_checkpoint_audit"] = CONTROL_AUDIT.get("hidden_checkpoint")
    (output / CONFIG_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_fixed_boundary_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    trainable: bool = False,
) -> nn.Module:
    """Attach a saved fixed boundary to a model that already has hidden PEFT."""

    directory = Path(adapter_dir)
    raw = json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
    if raw.get("marker") != CONTROL_MARKER:
        raise RuntimeError(f"Invalid fixed-boundary marker in {directory}")
    boundary = install_fixed_boundary(
        model,
        kind=str(raw["kind"]),
        rank=int(raw["rank"]),
        alpha=float(raw["alpha"]),
        enforce_qwen_budget=True,
    )
    manifest_checks = {
        "hidden_size": boundary.hidden_size,
        "vocab_size": boundary.vocab_size,
        "trainable_parameters": sum(
            parameter.numel() for parameter in boundary.affine.parameters()
        ),
        "boundary_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "frozen_native_weight_sha256": boundary.frozen_native_weight_sha256,
        "frozen_fp32_weight_sha256": boundary.frozen_fp32_weight_sha256,
    }
    mismatched_manifest = {
        key: {"saved": raw.get(key), "runtime": value}
        for key, value in manifest_checks.items()
        if raw.get(key) != value
    }
    if mismatched_manifest:
        raise RuntimeError(
            "Fixed-boundary manifest does not match reconstructed model: "
            f"{mismatched_manifest}"
        )
    state = load_file(str(directory / BOUNDARY_FILENAME), device="cpu")
    expected = {"affine.down.weight", "affine.up.weight"}
    if set(state) != expected:
        raise RuntimeError(
            f"Fixed-boundary state mismatch: expected={sorted(expected)} "
            f"actual={sorted(state)}"
        )
    factors = {name.removeprefix("affine."): tensor for name, tensor in state.items()}
    missing, unexpected = boundary.affine.load_state_dict(factors, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load fixed boundary: missing={missing} unexpected={unexpected}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for parameter in boundary.affine.parameters():
            parameter.requires_grad_(True)
    return model


def load_fixed_hidden_boundary_run(
    model_path: str | Path,
    run_dir: str | Path,
    *,
    torch_dtype: Any = "auto",
    trust_remote_code: bool = False,
) -> nn.Module:
    """Independently reconstruct hidden PEFT + fixed boundary for evaluation."""

    from transformers import AutoModelForCausalLM

    directory = Path(run_dir)
    boundary_manifest = json.loads(
        (directory / CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(
        boundary_manifest["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(boundary_manifest["cudnn_allow_tf32"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, directory, is_trainable=False)
    disable_hidden_lora_dropout(model)
    model = load_fixed_boundary_adapter(model, directory, trainable=False)
    saved_hidden = source_hidden_tensors(directory)
    runtime_hidden = model_hidden_tensors(model)
    if saved_hidden.keys() != runtime_hidden.keys() or any(
        not torch.equal(saved_hidden[name].float(), runtime_hidden[name].float().cpu())
        for name in saved_hidden
    ):
        raise RuntimeError("Evaluation reconstruction changed the saved hidden adapter")
    model.eval()
    return model


def _validate_args(args: Any, kind: str) -> None:
    required_rank = 50 if kind == "alora" else 1
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(
        args.variant == "affine_lm_head_plus_hidden_lora",
        "--variant must be affine_lm_head_plus_hidden_lora",
    )
    require(bool(args.initial_hidden_lora_adapter), "--initial-hidden-lora-adapter is required")
    require(args.freeze_initial_hidden_lora, "--freeze-initial-hidden-lora is required")
    require(args.affine_rank == required_rank, f"--affine-rank must be {required_rank}")
    require(args.affine_dropout == 0.0, "--affine-dropout must be 0")
    require(not args.affine_lm_head_bias, "--affine-lm-head-bias is forbidden")
    require(args.no_affine_input_bias, "--no-affine-input-bias must be explicit")
    require(not args.tie_affine_input_lm_head_adapters, "tied input/output adapter is forbidden")
    require(args.initial_affine_adapter is None, "--initial-affine-adapter is forbidden")
    require(args.include_emb_lmh_lora_rank == 0, "--include-emb-lmh-lora-rank must be 0")
    require(args.master_dtype == "fp32", "--master-dtype must be fp32")
    require(args.max_grad_norm == 0.0, "--max-grad-norm must be 0 (no clipping)")
    require(args.affine_energy_lambda == 0.0, "affine energy penalty is forbidden")
    require(args.affine_bias_energy_lambda == 0.0, "affine bias penalty is forbidden")
    require(args.reference_run_dir is None, "reference KL is forbidden")
    require(args.reference_kl_lambda == 0.0, "reference KL is forbidden")
    require(args.anchor_fraction == 0.0, "anchor replacement is forbidden")
    require(args.auxiliary_anchor_lambda == 0.0, "auxiliary anchor loss is forbidden")
    require(
        args.affine_bias_learning_rate_scale == 1.0,
        "affine bias LR scale must remain 1",
    )
    require(math.isfinite(args.affine_alpha) and args.affine_alpha > 0, "alpha must be positive")
    if failures:
        raise RuntimeError("Invalid fixed-hidden control arguments:\n- " + "\n- ".join(failures))


def install_control_monkeypatches(kind: str) -> None:
    """Install narrow construction, loading, optimizer, and save hooks."""

    original_parse_args = legacy.parse_args
    original_peft_from_pretrained = legacy.PeftModel.from_pretrained
    original_trainer = legacy.AffineLearningRateTrainer

    def audited_parse_args():  # noqa: ANN202
        args = original_parse_args()
        _validate_args(args, kind)
        WRAPPER_OPTIONS["parsed_args"] = vars(args).copy()
        return args

    def custom_apply(model: nn.Module, cfg: Any) -> nn.Module:
        global BOUNDARY_MODULE
        if cfg.use_input or not cfg.use_lm_head:
            raise RuntimeError(
                "Fixed-hidden control must request output-only boundary topology"
            )
        BOUNDARY_MODULE = install_fixed_boundary(
            model,
            kind=kind,
            rank=int(cfg.rank),
            alpha=float(cfg.alpha),
            enforce_qwen_budget=True,
        )
        print(
            "[fixed_boundary] "
            + json.dumps(asdict(boundary_config(BOUNDARY_MODULE)), sort_keys=True),
            flush=True,
        )
        return model

    def audited_hidden_load(
        model: nn.Module,
        adapter_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        if kwargs.get("is_trainable", False):
            raise RuntimeError("The initial hidden LoRA must load with is_trainable=False")
        loaded = original_peft_from_pretrained(model, adapter_dir, *args, **kwargs)
        adapter_path = Path(adapter_dir)
        hidden_audit = audit_hidden_tensor_identity(loaded, adapter_path)
        replaced = disable_hidden_lora_dropout(loaded)
        hidden_audit["dropout_modules_replaced_with_identity"] = len(replaced)
        hidden_audit["dropout_identity_assertion"] = "passed"
        CONTROL_AUDIT["hidden_checkpoint"] = hidden_audit
        print(
            "[fixed_hidden] " + json.dumps(hidden_audit, sort_keys=True),
            flush=True,
        )
        return loaded

    class AuditedBoundaryOnlyTrainer(original_trainer):
        def _audit_trainable(self) -> dict[str, Any]:
            boundary = find_fixed_boundary(self.model)
            trainable = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            expected_ids = {id(parameter) for parameter in boundary.affine.parameters()}
            actual_ids = {id(parameter) for _, parameter in trainable}
            if actual_ids != expected_ids or len(trainable) != 2:
                raise RuntimeError(
                    "Only the two output-boundary matrices may be trainable; "
                    f"actual={[name for name, _ in trainable]}"
                )
            wrong_dtype = [
                f"{name}:{parameter.dtype}"
                for name, parameter in trainable
                if parameter.dtype != torch.float32
            ]
            if wrong_dtype:
                raise RuntimeError(f"Boundary parameters are not FP32: {wrong_dtype}")
            hidden_trainable = [
                name
                for name, parameter in self.model.named_parameters()
                if "lora_" in name and parameter.requires_grad
            ]
            if hidden_trainable:
                raise RuntimeError(f"Frozen hidden LoRA became trainable: {hidden_trainable[:8]}")
            count = sum(parameter.numel() for _, parameter in trainable)
            if count != EXPECTED_COUNTS[kind]:
                raise RuntimeError(
                    f"Trainable count mismatch for {kind}: "
                    f"{count} != {EXPECTED_COUNTS[kind]}"
                )
            return {
                "trainable_names": [name for name, _ in trainable],
                "trainable_shapes": {
                    name: list(parameter.shape) for name, parameter in trainable
                },
                "trainable_dtypes": {
                    name: str(parameter.dtype) for name, parameter in trainable
                },
                "trainable_parameters": count,
                "only_boundary_trainable_assertion": "passed",
                "hidden_frozen_assertion": "passed",
                "max_grad_norm": float(self.args.max_grad_norm),
            }

        def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
            before = self._audit_trainable()
            optimizer = super().create_optimizer(model)
            boundary = find_fixed_boundary(self.model)
            expected_ids = {id(parameter) for parameter in boundary.affine.parameters()}
            grouped = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            grouped_ids = [id(parameter) for parameter in grouped]
            if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != expected_ids:
                raise RuntimeError("Optimizer does not cover each boundary parameter exactly once")
            state_dtypes = sorted({str(parameter.dtype) for parameter in grouped})
            before.update(
                {
                    "optimizer_parameter_coverage_assertion": "passed",
                    "optimizer_group_count": len(optimizer.param_groups),
                    "optimizer_parameter_dtypes": state_dtypes,
                    "learning_rates": sorted(
                        {float(group["lr"]) for group in optimizer.param_groups}
                    ),
                }
            )
            CONTROL_AUDIT["boundary_optimizer"] = before
            print(
                "[fixed_boundary_optimizer] "
                + json.dumps(before, sort_keys=True),
                flush=True,
            )
            return optimizer

    legacy.parse_args = audited_parse_args
    legacy.apply_affine_vocab_adapters = custom_apply
    legacy.PeftModel = type(
        "_AuditedFrozenPeftLoader",
        (),
        {"from_pretrained": staticmethod(audited_hidden_load)},
    )
    legacy.AffineLearningRateTrainer = AuditedBoundaryOnlyTrainer
    legacy.save_affine_vocab_adapter = save_fixed_boundary_adapter


def _verify_completed_run(output_dir: Path) -> dict[str, Any]:
    required = [
        output_dir / "adapter_model.safetensors",
        output_dir / "adapter_config.json",
        output_dir / BOUNDARY_FILENAME,
        output_dir / CONFIG_FILENAME,
        output_dir / "run_args.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Completed fixed-boundary run is missing files: {missing}")
    boundary_state = load_file(str(output_dir / BOUNDARY_FILENAME), device="cpu")
    if set(boundary_state) != {"affine.down.weight", "affine.up.weight"}:
        raise RuntimeError("Saved boundary contains unexpected tensors")
    boundary_count = sum(tensor.numel() for tensor in boundary_state.values())
    if boundary_count != EXPECTED_COUNTS[WRAPPER_OPTIONS["kind"]]:
        raise RuntimeError(
            f"Saved boundary count mismatch: {boundary_count} != "
            f"{EXPECTED_COUNTS[WRAPPER_OPTIONS['kind']]}"
        )
    source_audit = CONTROL_AUDIT["hidden_checkpoint"]
    saved_hidden = source_hidden_tensors(output_dir)
    saved_digest = _tensor_digest(saved_hidden)
    if saved_digest != source_audit["canonical_tensor_sha256"]:
        raise RuntimeError(
            "Final saved hidden adapter differs from the frozen source: "
            f"{saved_digest} != {source_audit['canonical_tensor_sha256']}"
        )
    return {
        "marker": CONTROL_MARKER,
        "kind": WRAPPER_OPTIONS["kind"],
        "tf32_mode": WRAPPER_OPTIONS["tf32_mode"],
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "hidden_checkpoint": source_audit,
        "boundary_optimizer": CONTROL_AUDIT["boundary_optimizer"],
        "saved_boundary_file_sha256": sha256_file(output_dir / BOUNDARY_FILENAME),
        "saved_hidden_file_sha256": sha256_file(
            output_dir / "adapter_model.safetensors"
        ),
        "saved_hidden_canonical_tensor_sha256": saved_digest,
        "saved_hidden_unchanged_assertion": "passed",
        "saved_boundary_parameter_count": boundary_count,
        "completion_audit_assertion": "passed",
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }


def _persist_completed_audit(output_dir: Path, audit: dict[str, Any]) -> None:
    (output_dir / AUDIT_FILENAME).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_args_path = output_dir / "run_args.json"
    run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
    run_args["fixed_hidden_boundary_control"] = audit
    run_args_path.write_text(
        json.dumps(run_args, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    kind = _pop_choice_flag(KIND_FLAG, ("alora", "vocab_lora"))
    tf32_mode = _pop_choice_flag(TF32_FLAG, ("allow", "deny"), default="deny")
    allow_tf32 = tf32_mode == "allow"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    WRAPPER_OPTIONS.update({"kind": kind, "tf32_mode": tf32_mode})
    install_control_monkeypatches(kind)
    corrected.main()

    output_value = corrected.argument_value("--output-dir")
    if not output_value:
        raise RuntimeError("--output-dir is required")
    output_dir = Path(output_value)
    audit = _verify_completed_run(output_dir)
    _persist_completed_audit(output_dir, audit)
    print("[fixed_control_complete] " + json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
