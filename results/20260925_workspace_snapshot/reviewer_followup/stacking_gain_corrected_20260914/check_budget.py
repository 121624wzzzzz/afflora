"""CPU structural check: extra capacity must learn and survive PEFT reload."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from budget import add_hidden_budget, hidden_digest


def main():
    torch.set_num_threads(2)
    records = []
    cases = [
        (Qwen3Config(hidden_size=8, intermediate_size=24, num_hidden_layers=28,
                     num_attention_heads=8, num_key_value_heads=4, head_dim=2,
                     vocab_size=32, tie_word_embeddings=True), Qwen3ForCausalLM),
        (Qwen2Config(hidden_size=56, intermediate_size=296, num_hidden_layers=28,
                     num_attention_heads=28, num_key_value_heads=4,
                     vocab_size=32, tie_word_embeddings=False), Qwen2ForCausalLM),
    ]
    for cfg, cls in cases:
        for rank in (2, 64):
            torch.manual_seed(42)
            base = cls(cfg)
            state = copy.deepcopy(base.state_dict())
            model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=rank,
                lora_alpha=2*rank, lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            model.eval()
            inputs = torch.tensor([[1, 4, 2, 5, 6, 8]])
            with torch.no_grad(): before = model(inputs).logits
            old_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            old_digest = hidden_digest(model)
            rng = torch.random.get_rng_state().clone()
            with patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA seed must remain unchanged")):
                detail = add_hidden_budget(model, rank, 42)
            assert torch.equal(rng, torch.random.get_rng_state())
            assert hidden_digest(model, rank) == old_digest
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count-old_count == detail["actual_extra_parameters"]
            with torch.no_grad(): assert torch.equal(before, model(inputs).logits)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            model.train()
            loss = model(inputs, labels=inputs).loss
            loss.backward()
            gradients = [m.lora_B["default"].weight.grad[:, -1].abs().sum().item()
                         for m in model.modules() if hasattr(m, "r") and isinstance(m.r, dict)
                         and m.r.get("default") == rank+1]
            assert gradients and all(g > 0 for g in gradients)
            optimizer.step()
            model.eval()
            with torch.no_grad(): after = model(inputs).logits
            assert not torch.equal(before, after)
            with tempfile.TemporaryDirectory(prefix="stacking_budget_") as tmp:
                model.save_pretrained(tmp)
                restored_base = cls(cfg)
                restored_base.load_state_dict(state)
                restored = PeftModel.from_pretrained(restored_base, tmp).eval()
                with torch.no_grad(): actual = restored(inputs).logits
                error = (after-actual).abs().max().item()
                assert error < 1e-6, error
                assert hidden_digest(model) == hidden_digest(restored)
            records.append({"model": cls.__name__, "rank": rank,
                "extra": count-old_count, "new_rank_slices_learn": True,
                "shared_init_preserved": True, "rng_preserved": True,
                "reload_max_abs_error": error})
    out = Path(__file__).resolve().parent / "budget_check.json"
    out.write_text(json.dumps({"status":"passed", "cases":records},indent=2)+"\n")
    print(out.read_text())


if __name__ == "__main__": main()
