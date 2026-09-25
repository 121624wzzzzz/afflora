# Near-equal boundary-budget comparison

This comparison uses the corrected multi-turn SFT pipeline on Qwen2.5-1.5B.

## Budget

The base model has hidden size 1,536 and vocabulary size 151,936.

- decoupled input/output A-LoRA rank 48:
  `4 * 1536 * 48 + 1536 = 296,448` boundary parameters;
- input/output Vocab LoRA rank 1:
  `2 * (151936 + 1536) = 306,944` boundary parameters.

Both are added to hidden LoRA rank 8. Their boundary budgets differ by 10,496
parameters, while the resulting total trainable budgets differ by about 0.11%.
A-LoRA uses alpha 384 (scale 8), matching the scale of the existing rank-16
configuration; Vocab LoRA uses alpha 2.

## Runs

- seeds: 42, 43, 44;
- baseline checkpoints: existing corrected-SFT hidden LoRA rank 8 runs;
- treatments: hidden + A-LoRA rank 48 and hidden + Vocab LoRA rank 1;
- final evaluation: untouched corrected test split with per-example NLL and
  paired item bootstrap.

## Initialization control

Standard PEFT construction of embedding LoRA consumes the global random stream
before hidden LoRA modules are initialized. A direct audit showed that this
changes the hidden-LoRA initialization at the same nominal seed.

The Vocab-LoRA runs therefore use
`reviewer_followup/train_corrected_sft_isolated_vocab.py`, which initializes the
embedding/lm-head adapters in an isolated RNG stream. For seed 42, the hidden
LoRA SHA256 is identical with and without Vocab LoRA:

`12de5ac2e5a9fa8032228014a6da12226c6132ea4b62799b4a5f8e1d26e9c469`.
