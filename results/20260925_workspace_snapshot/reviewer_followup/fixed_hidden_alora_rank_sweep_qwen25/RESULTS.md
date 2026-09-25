# Fixed-hidden A-LoRA rank sweep: final result

## Result

Within the corrected-SFT, frozen-hidden, output-only FP32 protocol, increasing
A-LoRA rank from 2 through 50 improves train-subset, dev, and test CE
monotonically. Rank 50 is the best tested endpoint. Thus the rank-50
disadvantage against direct Vocab-LoRA is not explained by a lower-rank
held-out sweet spot.

All runs use hidden seed 42, functional scale 16, one fixed epoch, and the same
frozen hidden checkpoint and example order.

| rank | boundary parameters | train1000 CE | dev CE | test CE |
|---:|---:|---:|---:|---:|
| 2 | 6,144 | 1.117070340 | 1.112973366 | 1.132496473 |
| 4 | 12,288 | 1.116460634 | 1.112711213 | 1.132136936 |
| 8 | 24,576 | 1.115131792 | 1.112059818 | 1.131511148 |
| 16 | 49,152 | 1.113038024 | 1.111045905 | 1.130232467 |
| 32 | 98,304 | 1.109090478 | 1.108875296 | 1.128488106 |
| 50 | 153,600 | 1.104395854 | 1.107084474 | 1.126872773 |

## Paired test evidence

The sign below is each rank's test CE minus r50 CE. Positive values favor r50.

| rank | rank−r50 test CE | paired-item bootstrap 95% CI |
|---:|---:|---:|
| 2 | +0.005623700 | [+0.005101486, +0.006151701] |
| 4 | +0.005264164 | [+0.004749737, +0.005793174] |
| 8 | +0.004638376 | [+0.004109603, +0.005167034] |
| 16 | +0.003359695 | [+0.002825433, +0.003883511] |
| 32 | +0.001615334 | [+0.001066770, +0.002156709] |
| 50 | 0 | [0, 0] |

From r2 to r50, train1000 CE improves by 0.012674487, while dev and test
improve by 0.005888892 and 0.005623700. The widening train-to-held-out gap
shows that only about 44–47% of the extra in-sample fit transfers, so there is
a partial overfitting component. It does not produce a held-out reversal:
dev and test still improve at every tested rank.

## Interpretation

This result rules out the simple explanation that forcing A-LoRA to rank 50
caused its recent fixed-hidden output-only CE failure. Lower ranks make that
endpoint worse, not better.

Relative to the same zero-boundary hidden checkpoint (test CE 1.132984093),
r2 improves by only 0.000487620 and r50 improves by 0.006111320. The direct
rank-1 Vocab-LoRA endpoint remains much lower at 1.026656346. Therefore the
large A-LoRA/Vocab-LoRA gap is primarily associated with the output function
class and topology, not an r50 overfitting sweet spot.

This does not contradict the historical low-rank AffLoRA parameter-efficiency
results. Those used input-side or bilateral AffLoRA, often jointly trained
with hidden LoRA, and legacy SFT data. This sweep isolates an output-only
boundary on a frozen corrected-SFT hidden representation. The most valuable
remaining reproduction is the historical `affine_input r2` versus
`vocab-only LoRA r2/r4` topology on the corrected data pipeline.

## Scope

This is a seed-42 mechanism diagnostic on previously inspected data. The
paired-item intervals condition on this trained seed and do not establish a
cross-seed rank claim.
