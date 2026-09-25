# Output-only equal-budget final IFEval summary

The metric is official IFEval prompt-level accuracy (`follow_all_instructions`). The sign convention is **A-LoRA − Vocab-LoRA**; positive differences favor A-LoRA.

## Frozen configurations

| Method | Scale | Boundary-LR multiplier | Boundary LR |
|---|---:|---:|---:|
| output-only A-LoRA r50 | 16 | 1 | 0.0002000 |
| output-only Vocab-LoRA r1 | 32 | 1 | 0.0002000 |

## Inferential units

- **Prompt/item level:** Within each trained seed, 541 paired prompts provide the contingency table, exact McNemar p-value, and paired-prompt bootstrap CI.
- **Training-seed level:** The primary paired-t CI uses only independently trained seeds 43–45. It does not pool prompts across seeds.
- Seed 42 selected the hyperparameters on corrected dev and is reported separately.

## Strict: per-seed paired prompt analysis

| Seed | Role | A acc. | V acc. | A−V | Direction | Both ✓ | A-only ✓ | V-only ✓ | Both ✗ | Bootstrap 95% CI | Exact p |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | selected | 22.736% | 20.702% | +2.033 pp | aff_better | 75 | 48 | 37 | 381 | [-1.294 pp, +5.360 pp] | 0.277999 |
| 43 | primary | 23.290% | 21.627% | +1.664 pp | aff_better | 75 | 51 | 42 | 373 | [-1.848 pp, +5.176 pp] | 0.406924 |
| 44 | primary | 20.887% | 20.887% | +0.000 pp | tie | 72 | 41 | 41 | 387 | [-3.327 pp, +3.327 pp] | 1 |
| 45 | primary | 21.627% | 22.921% | -1.294 pp | vocab_better | 71 | 46 | 53 | 371 | [-4.991 pp, +2.218 pp] | 0.546713 |

## Loose: per-seed paired prompt analysis

| Seed | Role | A acc. | V acc. | A−V | Direction | Both ✓ | A-only ✓ | V-only ✓ | Both ✗ | Bootstrap 95% CI | Exact p |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | selected | 25.508% | 23.660% | +1.848 pp | aff_better | 82 | 56 | 46 | 357 | [-1.848 pp, +5.545 pp] | 0.372944 |
| 43 | primary | 25.139% | 24.584% | +0.555 pp | aff_better | 81 | 55 | 52 | 353 | [-3.142 pp, +4.251 pp] | 0.846802 |
| 44 | primary | 23.290% | 22.921% | +0.370 pp | aff_better | 77 | 49 | 47 | 368 | [-3.142 pp, +3.882 pp] | 0.918778 |
| 45 | primary | 24.769% | 26.063% | -1.294 pp | vocab_better | 85 | 49 | 56 | 351 | [-4.991 pp, +2.403 pp] | 0.558394 |

## Primary seed-level summary: seeds 43–45

### Strict

- A-LoRA mean `21.935%`, sample SD `1.231%`.
- Vocab-LoRA mean `21.811%`, sample SD `1.029%`.
- Mean A−V `+0.123 pp`, sample SD `+1.483 pp`.
- Seed-level paired-t 95% CI (df=2): `[-3.560 pp, +3.806 pp]`.
- Directions: A-LoRA better `1/3`; Vocab-LoRA better `1/3`; ties `1/3`.

### Loose

- A-LoRA mean `24.399%`, sample SD `0.978%`.
- Vocab-LoRA mean `24.522%`, sample SD `1.572%`.
- Mean A−V `-0.123 pp`, sample SD `+1.018 pp`.
- Seed-level paired-t 95% CI (df=2): `[-2.652 pp, +2.406 pp]`.
- Directions: A-LoRA better `2/3`; Vocab-LoRA better `1/3`; ties `0/3`.

## Seed 42, reported separately

| Mode | A acc. | V acc. | A−V | Direction |
|---|---:|---:|---:|---|
| strict | 22.736% | 20.702% | +2.033 pp | aff_better |
| loose | 25.508% | 23.660% | +1.848 pp | aff_better |

Seed 42 is excluded from the primary paired-t interval because its corrected-dev results selected the frozen configurations.

## Four-seed descriptive view

| Mode | Mean A acc. | Mean V acc. | Mean A−V | Delta sample SD | Directions A/V/tie |
|---|---:|---:|---:|---:|---:|
| strict | 22.135% | 21.534% | +0.601 pp | +1.542 pp | 2/1/1 |
| loose | 24.677% | 24.307% | +0.370 pp | +1.289 pp | 3/1/0 |

The four-seed view is descriptive only.

## Artifact validation

Validated the exact frozen set of 8 response artifacts, 8 strict score artifacts, and 8 loose score artifacts. Every artifact contains exactly 541 rows aligned in canonical Google IFEval prompt order; response, instruction-list, checkpoint, method, scale, boundary-LR, and seed metadata checks all passed.
