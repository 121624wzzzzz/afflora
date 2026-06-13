from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass
class AffineVocabConfig:
    """Hyperparameters for the hidden-dimensional affine vocab adapter.

    Formula: W' = W (I + s1 * A @ B) + s2 * b
        s1 = alpha / rank   (LoRA-style scale, swept via ``alpha``)
        s2 = bias_scale     (extra multiplicative scale on the bias term)
    """

    hidden_size: int
    rank: int = 16
    alpha: float = 32.0
    bias_scale: float = 1.0
    dropout: float = 0.0
    use_input: bool = True
    use_lm_head: bool = False
    use_input_bias: bool = True
    use_lm_head_bias: bool = False
    tie_input_lm_head_adapters: bool = False

    @property
    def scale(self) -> float:
        return self.alpha / max(1, self.rank)


class LowRankAffineMap(nn.Module):
    """Row-wise affine map ``x -> x + s1 * U(D(x)) + s2 * bias``.

    ``s1 = alpha / rank`` follows the LoRA convention.
    ``s2 = bias_scale`` is exposed separately so that the bias contribution can be
    swept independently of the low-rank update.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: float,
        dropout: float,
        use_bias: bool,
        bias_scale: float = 1.0,
    ):
        super().__init__()
        self.scale = alpha / max(1, rank)
        self.bias_scale = bias_scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.to(dtype=self.down.weight.dtype)
        out = x + self.up(self.down(self.dropout(x))) * self.scale
        if self.bias is not None:
            out = out + self.bias_scale * self.bias.to(dtype=out.dtype)
        return out.to(dtype=out_dtype)


class AffineEmbedding(nn.Module):
    def __init__(
        self,
        base_embedding: nn.Embedding,
        cfg: AffineVocabConfig,
        affine: LowRankAffineMap | None = None,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.affine = affine or LowRankAffineMap(
            cfg.hidden_size,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.use_input_bias,
            bias_scale=cfg.bias_scale,
        )
        self.affine.to(device=base_embedding.weight.device)
        self.base_embedding.weight.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_embedding.weight

    @property
    def num_embeddings(self) -> int:
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base_embedding.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.affine(self.base_embedding(input_ids))


class AffineLMHead(nn.Module):
    def __init__(
        self,
        base_head: nn.Linear,
        cfg: AffineVocabConfig,
        affine: LowRankAffineMap | None = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.affine = affine or LowRankAffineMap(
            cfg.hidden_size,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.use_lm_head_bias,
            bias_scale=cfg.bias_scale,
        )
        self.affine.to(device=base_head.weight.device)
        self.base_head.weight.requires_grad_(False)
        if self.base_head.bias is not None:
            self.base_head.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_head.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_head.bias

    @property
    def in_features(self) -> int:
        return self.base_head.in_features

    @property
    def out_features(self) -> int:
        return self.base_head.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.affine(hidden_states).to(dtype=self.base_head.weight.dtype)
        return self.base_head(hidden_states)


class TiedTransposeAffineLMHead(nn.Module):
    """lm_head wrapper exactly matching a merged tied embedding.

    For input-side merge ``W' = W M + b``, the tied output logits are
    ``h W'^T = h M^T W^T + (h b^T) * 1``. The common scalar is included so raw
    logits match the merged model, although it is softmax-invariant.
    """

    def __init__(self, base_head: nn.Linear, cfg: AffineVocabConfig, affine: LowRankAffineMap):
        super().__init__()
        self.base_head = base_head
        self.affine = affine
        self.base_head.weight.requires_grad_(False)
        if self.base_head.bias is not None:
            self.base_head.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_head.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_head.bias

    @property
    def in_features(self) -> int:
        return self.base_head.in_features

    @property
    def out_features(self) -> int:
        return self.base_head.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out_dtype = hidden_states.dtype
        x = hidden_states.to(dtype=self.affine.down.weight.dtype)
        low_rank = F.linear(F.linear(x, self.affine.up.weight.T), self.affine.down.weight.T)
        projected = (x + self.affine.scale * low_rank).to(dtype=self.base_head.weight.dtype)
        logits = self.base_head(projected)
        if self.affine.bias is not None:
            common_shift = (
                x * (self.affine.bias_scale * self.affine.bias.to(dtype=x.dtype))
            ).sum(dim=-1, keepdim=True)
            logits = logits + common_shift.to(dtype=logits.dtype)
        return logits.to(dtype=out_dtype)


def _infer_hidden_size(model: nn.Module) -> int:
    emb = model.get_input_embeddings()
    return int(emb.embedding_dim)


def _freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _input_lm_head_weights_are_tied(model: nn.Module) -> bool:
    emb = model.get_input_embeddings()
    lm_head = getattr(model, "lm_head", None)
    return (
        emb is not None
        and lm_head is not None
        and hasattr(emb, "weight")
        and hasattr(lm_head, "weight")
        and emb.weight.data_ptr() == lm_head.weight.data_ptr()
    )


def apply_affine_vocab_adapters(
    model: nn.Module,
    cfg: AffineVocabConfig | None = None,
    **kwargs: Any,
) -> nn.Module:
    if cfg is None:
        cfg = AffineVocabConfig(hidden_size=_infer_hidden_size(model), **kwargs)
    else:
        cfg.hidden_size = cfg.hidden_size or _infer_hidden_size(model)

    _freeze_model(model)
    model.affine_vocab_config = cfg
    shared_affine: LowRankAffineMap | None = None
    if cfg.tie_input_lm_head_adapters:
        if not (cfg.use_input and cfg.use_lm_head):
            raise ValueError("tie_input_lm_head_adapters requires both use_input and use_lm_head.")
        if cfg.use_lm_head_bias != cfg.use_input_bias:
            raise ValueError("Tied input/lm_head adapters require matching input and lm_head bias settings.")
        if not cfg.use_lm_head_bias:
            raise ValueError("Tied input/lm_head adapters require affine lm_head bias to match input bias.")
        if not _input_lm_head_weights_are_tied(model):
            raise ValueError("tie_input_lm_head_adapters requires tied embed_tokens/lm_head weights.")
        shared_affine = LowRankAffineMap(
            cfg.hidden_size,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.use_input_bias,
            bias_scale=cfg.bias_scale,
        )

    if cfg.use_input:
        model.set_input_embeddings(AffineEmbedding(model.get_input_embeddings(), cfg, shared_affine))

    if cfg.use_lm_head:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None or not isinstance(lm_head, nn.Linear):
            raise TypeError("Expected model.lm_head to be an nn.Linear for lm_head affine adapter.")
        if shared_affine is not None:
            model.lm_head = TiedTransposeAffineLMHead(lm_head, cfg, shared_affine)
        else:
            model.lm_head = AffineLMHead(lm_head, cfg, shared_affine)

    return model


def affine_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".affine." in name
    }


def save_affine_vocab_adapter(model: nn.Module, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = getattr(model, "affine_vocab_config", None)
    if cfg is None:
        raise ValueError("Model does not have affine_vocab_config.")
    with (output / "affine_vocab_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    save_file(affine_state_dict(model), str(output / "affine_vocab_adapter.safetensors"))


def load_affine_vocab_adapter(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    adapter = Path(adapter_dir)
    with (adapter / "affine_vocab_config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
        # Filter to known fields (backward compat with older configs)
        import dataclasses
        known = {f.name for f in dataclasses.fields(AffineVocabConfig)}
        cfg = AffineVocabConfig(**{k: v for k, v in raw.items() if k in known})
    apply_affine_vocab_adapters(model, cfg)
    state = load_file(str(adapter / "affine_vocab_adapter.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_unexpected = [name for name in unexpected if ".affine." in name or "vocab_logit_bias" in name]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected affine adapter keys: {bad_unexpected}")
    loaded = set(state)
    not_loaded = [
        name
        for name, _ in model.named_parameters()
        if (".affine." in name or "vocab_logit_bias" in name) and name not in loaded
    ]
    if not_loaded:
        raise RuntimeError(f"Missing affine adapter keys: {not_loaded}; load_state missing={missing}")
    return model
