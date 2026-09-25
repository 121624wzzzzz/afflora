# Paired test analysis against r50

Positive deltas favor r50. Bootstrap unit: paired corrected-SFT test example.

| rank | test CE | rank−r50 CE | paired bootstrap 95% CI |
|---:|---:|---:|---:|
| 2 | 1.132496473 | +0.005623700 | [+0.005101486, +0.006151701] |
| 4 | 1.132136936 | +0.005264164 | [+0.004749737, +0.005793174] |
| 8 | 1.131511148 | +0.004638376 | [+0.004109603, +0.005167034] |
| 16 | 1.130232467 | +0.003359695 | [+0.002825433, +0.003883511] |
| 32 | 1.128488106 | +0.001615334 | [+0.001066770, +0.002156709] |
| 50 | 1.126872773 | +0.000000000 | [+0.000000000, +0.000000000] |

- r2→r50 train1000 CE reduction: `0.012674487`.
- r2→r50 dev CE reduction: `0.005888892` (46.5% of train1000 gain).
- r2→r50 test CE reduction: `0.005623700` (44.4% of train1000 gain).
- Diagnosis: all three curves improve monotonically through r50. The generalization gap widens, but there is no held-out reversal or lower-rank sweet spot within r2–r50.
