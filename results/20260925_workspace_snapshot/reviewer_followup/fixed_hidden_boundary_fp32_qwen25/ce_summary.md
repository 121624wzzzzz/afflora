# Fixed-hidden FP32 boundary CE summary

The sign convention is **A-LoRA CE − Vocab-LoRA CE**; negative values favor A-LoRA. Hidden-baseline improvement is **hidden-only zero-boundary CE − method CE**; positive values are improvements.

## Primary corrected test (seeds 42–44)

| Seed | Hidden zero-boundary CE | A-LoRA CE | Vocab CE | A−V | Hidden→A improvement | Hidden→V improvement | Item-bootstrap 95% CI | P(A better) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.132984093 | 1.126872773 | 1.026656346 | +0.100216427 | +0.006111320 (+0.539%) | +0.106327747 (+9.385%) | [+0.097034736, +0.103463050] | 0.0000 |
| 43 | 1.132497420 | 1.126132620 | 1.026239502 | +0.099893118 | +0.006364800 (+0.562%) | +0.106257918 (+9.383%) | [+0.096717457, +0.103044653] | 0.0000 |
| 44 | 1.132312386 | 1.126273465 | 1.025983548 | +0.100289917 | +0.006038921 (+0.533%) | +0.106328838 (+9.390%) | [+0.097184709, +0.103558105] | 0.0000 |

- Mean hidden CE: `1.132597966` (sample SD `0.000346958`).
- Mean A-LoRA CE: `1.126426286` (sample SD `0.000393029`).
- Mean Vocab-LoRA CE: `1.026293132` (sample SD `0.000339590`).
- Mean A−V: `+0.100133154` (sample SD `0.000211100`).
- Seed-level paired-t 95% CI (df=2): `[+0.099608753, +0.100657555]`.
- Directions: A-LoRA `0/3`; Vocab-LoRA `3/3`; ties `0/3`.
- Mean CE reduction versus hidden: A-LoRA `+0.006171680`; Vocab-LoRA `+0.106304835`.

## Corrected dev, descriptive (seeds 42–44)

| Seed | Hidden zero-boundary CE | A-LoRA CE | Vocab CE | A−V | Hidden→A improvement | Hidden→V improvement | Item-bootstrap 95% CI | P(A better) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.113355350 | 1.107084474 | 1.007713106 | +0.099371368 | +0.006270876 (+0.563%) | +0.105642244 (+9.489%) | [+0.096228387, +0.102659712] | 0.0000 |
| 43 | 1.113147717 | 1.107164661 | 1.007346833 | +0.099817829 | +0.005983055 (+0.537%) | +0.105800884 (+9.505%) | [+0.096637766, +0.103084294] | 0.0000 |
| 44 | 1.112989157 | 1.106891574 | 1.007388584 | +0.099502990 | +0.006097583 (+0.548%) | +0.105600573 (+9.488%) | [+0.096313512, +0.102794143] | 0.0000 |

- Mean hidden CE: `1.113164075` (sample SD `0.000183644`).
- Mean A-LoRA CE: `1.107046903` (sample SD `0.000140367`).
- Mean Vocab-LoRA CE: `1.007482841` (sample SD `0.000200505`).
- Mean A−V: `+0.099564062` (sample SD `0.000229411`).
- Seed-level paired-t 95% CI (df=2): `[+0.098994174, +0.100133950]`.
- Directions: A-LoRA `0/3`; Vocab-LoRA `3/3`; ties `0/3`.
- Mean CE reduction versus hidden: A-LoRA `+0.006117172`; Vocab-LoRA `+0.105681234`.

## Uncertainty interpretation

The paired-t interval treats the three independently trained seed pairs as the inferential units. Each 10,000-sample item bootstrap instead conditions on one already-trained seed and measures evaluation-set sampling uncertainty. Narrow item intervals do not establish robustness across training seeds.

## Seed-42 common-scale sensitivity

This is a diagnostic only; it is not part of the primary three-seed inference.

### test

| Common scale | A-LoRA CE | Vocab CE | A−V | Direction |
|---:|---:|---:|---:|---|
| 16 | 1.126872773 | 1.026400554 | +0.100472218 | vocab_better |
| 32 | 1.127506775 | 1.026656346 | +0.100850429 | vocab_better |

### dev

| Common scale | A-LoRA CE | Vocab CE | A−V | Direction |
|---:|---:|---:|---:|---|
| 16 | 1.107084474 | 1.007383875 | +0.099700599 | vocab_better |
| 32 | 1.107765918 | 1.007713106 | +0.100052812 | vocab_better |

## Validation

Validated `12` fixed-boundary reports and `6` formal hidden-only reports. Completion-manifest digests, report arithmetic, source record-ID order, and supervised-token pairing all passed.
