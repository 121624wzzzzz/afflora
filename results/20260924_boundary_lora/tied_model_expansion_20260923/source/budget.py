"""Spend the bilateral boundary budget on extra hidden-LoRA rank instead."""
from __future__ import annotations

import hashlib
import math
import re

import torch
from torch import nn


def hidden_digest(model, base_rank=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        if base_rank is not None:
            if ".lora_A." in name:
                value = value[:base_rank, :]
            elif ".lora_B." in name:
                value = value[:, :base_rank]
            else:
                raise ValueError(name)
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evenly_spaced(items, count):
    assert 0 <= count <= len(items)
    if not count:
        return []
    return [items[(i * len(items)) // count] for i in range(count)]


def add_hidden_budget(model, base_rank, seed):
    """Keep every shared initialization and scaling factor exactly unchanged.

    Add one rank to evenly spaced q_proj layers first (at most all layers),
    then k_proj layers until reaching the bilateral A-LoRA r16 budget.
    The 7B model overshoots by 512 parameters because of rank granularity.
    """
    target = 65 * int(model.config.hidden_size)
    before = hidden_digest(model)
    groups = {}
    for kind in ("q_proj", "k_proj"):
        candidates = [(name, layer) for name, layer in model.named_modules()
                      if name.endswith("." + kind) and hasattr(layer, "lora_A")]
        candidates.sort(key=lambda pair: int(re.search(r"layers\.(\d+)\.", pair[0])[1]))
        assert candidates
        groups[kind] = candidates
    costs = {kind: {layer.in_features + layer.out_features for _,layer in values}
             for kind,values in groups.items()}
    assert all(len(values)==1 for values in costs.values())
    qcost,kcost = costs['q_proj'].pop(),costs['k_proj'].pop()
    candidates = [(q*qcost+k*kcost-target,-q,k,q) for q in range(len(groups['q_proj'])+1)
                  for k in range(len(groups['k_proj'])+1) if q*qcost+k*kcost>=target]
    excess,_,nk,nq = min(candidates)
    spent = target+excess
    chosen = evenly_spaced(groups['q_proj'],nq)+evenly_spaced(groups['k_proj'],nk)
    assert target <= spent <= target * 1.003, (target, spent)
    cfg = model.peft_config["default"]
    assert not cfg.rank_pattern and not cfg.alpha_pattern
    pattern = {}
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator; manual_seed also changes CUDA RNGs.
        torch.random.default_generator.manual_seed(seed + 3_000_017)
        for name, layer in chosen:
            a, b = layer.lora_A["default"], layer.lora_B["default"]
            assert a.weight.device.type == "cpu"
            assert layer.r["default"] == base_rank
            new_rank = base_rank + 1
            new_a = nn.Linear(a.in_features, new_rank, bias=False, dtype=a.weight.dtype)
            new_b = nn.Linear(new_rank, b.out_features, bias=False, dtype=b.weight.dtype)
            with torch.no_grad():
                new_a.weight[:base_rank].copy_(a.weight)
                new_b.weight.zero_()
                new_b.weight[:, :base_rank].copy_(b.weight)
            layer.lora_A["default"] = new_a
            layer.lora_B["default"] = new_b
            layer.r["default"] = new_rank
            layer.lora_alpha["default"] = 2 * new_rank
            layer.scaling["default"] = 2.0
            # Full model path avoids ambiguities between q/k modules and layers.
            config_name = name.removeprefix("base_model.model.")
            cfg.rank_pattern[config_name] = new_rank
            cfg.alpha_pattern[config_name] = 2 * new_rank
            pattern[config_name] = new_rank
    assert hidden_digest(model, base_rank) == before
    return {"target_extra_parameters": target, "actual_extra_parameters": spent,
            "rank_pattern": pattern, "shared_initialization_sha256": before}
