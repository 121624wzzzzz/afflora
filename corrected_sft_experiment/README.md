# Corrected multi-turn SFT experiment

This directory reruns the AffLoRA headline comparison with a corrected data
pipeline.  The historical code and outputs outside this directory are not
modified.

The corrected pipeline:

- preserves the original `user` / `assistant` turn order;
- renders each conversation with the model tokenizer's chat template;
- computes loss only on assistant content and `<|im_end|>`;
- rejects malformed conversations instead of silently flattening them;
- uses deterministic unseen-first-prompt dev/test splits;
- records source checksums, filtering decisions, and overlap audits.

Prepare and verify the data:

```bash
cd /path/to/im_exp/lora
# Optional: source "$PWD/../set"
python corrected_sft_experiment/prepare_data.py
python -m unittest discover -s corrected_sft_experiment/tests -v
```

Training continues to use `scripts/train_affine_vocab_lora.py` through a thin
wrapper that replaces only its tokenizer/data function.  AffLoRA and PEFT
implementation code remains shared with the original project.

See `RESULTS.md` for the experiment matrix and results.

