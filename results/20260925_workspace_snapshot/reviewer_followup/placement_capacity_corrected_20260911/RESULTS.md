# Corrected-SFT placement × hidden capacity

Phase: `complete`. Validated full matrix cells: **50/50**.

Final-epoch, independently reloaded test CE; lower is better. Partial seed results are exploratory.
Intervals require all three paired seeds and describe training randomness on this fixed test set.

## qwen3_06b

| hidden rank | placement | seeds | mean test CE | total trainable |
| ---: | --- | --- | ---: | ---: |
| 0 | none | None | 1.875729 | 0 |
| 0 | input | 42,43,44 | 1.477954 | 33,792 |
| 0 | output | 42,43,44 | 1.548638 | 32,768 |
| 1 | none | 42,43,44 | 1.283851 | 630,784 |
| 1 | input | 42,43,44 | 1.278988 | 664,576 |
| 1 | output | 42,43,44 | 1.274678 | 663,552 |
| 8 | none | 42,43,44 | 1.214824 | 5,046,272 |
| 8 | input | 42,43,44 | 1.212111 | 5,080,064 |
| 8 | output | 42,43,44 | 1.210334 | 5,079,040 |

| paired contrast | mean [95% t CI when n=3] |
| --- | --- |
| D(0) = output - input | +0.070684 (n=3) [+0.069569, +0.071800] |
| r0: input - no boundary | -0.397776 (n=3) [-0.399574, -0.395977] |
| r0: output - no boundary | -0.327091 (n=3) [-0.329716, -0.324466] |
| D(1) = output - input | -0.004310 (n=3) [-0.004682, -0.003938] |
| r1: input - no boundary | -0.004862 (n=3) [-0.005232, -0.004493] |
| r1: output - no boundary | -0.009172 (n=3) [-0.009849, -0.008495] |
| D(8) = output - input | -0.001778 (n=3) [-0.002016, -0.001539] |
| r8: input - no boundary | -0.002713 (n=3) [-0.003052, -0.002374] |
| r8: output - no boundary | -0.004490 (n=3) [-0.004646, -0.004334] |
| I(8) = D(8) - D(0) | -0.072462 (n=3) [-0.073816, -0.071108] |
| I(1) = D(1) - D(0) | -0.074994 (n=3) [-0.076361, -0.073628] |

A reversal needs D(0)>0 AND D(8)<0; a negative interaction alone is insufficient.

## qwen25_7b

| hidden rank | placement | seeds | mean test CE | total trainable |
| ---: | --- | --- | ---: | ---: |
| 0 | none | None | 1.657705 | 0 |
| 0 | input | 42,43,44 | 1.266595 | 118,272 |
| 0 | output | 42,43,44 | 1.368168 | 114,688 |
| 1 | none | 42,43,44 | 1.036201 | 2,523,136 |
| 1 | input | 42,43,44 | 1.031244 | 2,641,408 |
| 1 | output | 42,43,44 | 1.028289 | 2,637,824 |
| 8 | none | 42,43,44 | 0.997374 | 20,185,088 |
| 8 | input | 42,43,44 | 0.995213 | 20,303,360 |
| 8 | output | 42,43,44 | 0.994015 | 20,299,776 |

| paired contrast | mean [95% t CI when n=3] |
| --- | --- |
| D(0) = output - input | +0.101573 (n=3) [+0.096257, +0.106889] |
| r0: input - no boundary | -0.391110 (n=3) [-0.393798, -0.388421] |
| r0: output - no boundary | -0.289537 (n=3) [-0.292186, -0.286887] |
| D(1) = output - input | -0.002956 (n=3) [-0.003804, -0.002107] |
| r1: input - no boundary | -0.004957 (n=3) [-0.005907, -0.004006] |
| r1: output - no boundary | -0.007912 (n=3) [-0.009035, -0.006790] |
| D(8) = output - input | -0.001198 (n=3) [-0.001841, -0.000555] |
| r8: input - no boundary | -0.002161 (n=3) [-0.002494, -0.001828] |
| r8: output - no boundary | -0.003359 (n=3) [-0.003809, -0.002909] |
| I(8) = D(8) - D(0) | -0.102771 (n=3) [-0.107840, -0.097702] |
| I(1) = D(1) - D(0) | -0.104529 (n=3) [-0.110478, -0.098580] |

A reversal needs D(0)>0 AND D(8)<0; a negative interaction alone is insufficient.

