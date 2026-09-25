# Output-only equal-budget final confirmation

The sign convention throughout is **A-LoRA CE − Vocab-LoRA CE**; negative values favor A-LoRA.

## Frozen configurations

| Method | Scale | Boundary-LR multiplier | Boundary LR |
|---|---:|---:|---:|
| output-only A-LoRA r50 | 16 | 1 | 0.0002000 |
| output-only Vocab-LoRA r1 | 32 | 1 | 0.0002000 |

## Primary test confirmation: independent seeds 43–45

| Seed | A-LoRA CE | Vocab-LoRA CE | A−V | Direction |
|---:|---:|---:|---:|---|
| 43 | 1.122098913 | 1.019337400 | +0.102761513 | vocab_better |
| 44 | 1.122133085 | 1.017254804 | +0.104878281 | vocab_better |
| 45 | 1.122227789 | 1.019543274 | +0.102684515 | vocab_better |

- Mean A-LoRA CE: `1.122153263` (sample SD `0.000066765`)
- Mean Vocab-LoRA CE: `1.018711826` (sample SD `0.001266009`)
- Mean A−V: `+0.103441437` (sample SD `0.001244939`)
- Seed-level paired-t 95% CI (df=2): `[+0.100348837, +0.106534036]`
- Directions: A-LoRA better `0/3`; Vocab-LoRA better `3/3`; ties `0/3`.

## Seed 42 test, reported separately

| Seed | A-LoRA CE | Vocab-LoRA CE | A−V | Direction |
|---:|---:|---:|---:|---|
| 42 | 1.122571314 | 1.019268387 | +0.103302928 | vocab_better |

Seed 42 is excluded from the primary interval because its dev result was used to choose the frozen hyperparameters.

## Four-seed descriptive view

- Mean A-LoRA CE: `1.122257775`
- Mean Vocab-LoRA CE: `1.018850966`
- Mean A−V: `+0.103406809` (sample SD `0.001018845`)

This four-seed mean is descriptive only and is not the confirmatory estimate.

## Artifact validation

Validated 16 reports (dev/test × two methods × four seeds) and recomputed all eight 10,000-sample paired-bootstrap comparisons. All configuration, path, seed, record-ID, supervised-token, CE, and comparison checks passed.
