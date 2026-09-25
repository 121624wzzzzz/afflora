# Llama-8B WikiSQL learning-rate follow-up

Development selection is frozen before confirmation. Scores are execution accuracy (%).

## Full development grid

| Arm | H LR | E/U ratio | Seed 7200 | Seed 7201 | Mean | Selected |
|---|---:|---:|---:|---:|---:|---|
| hidden | 0.0002 | 1 | 81.1523 | 82.5195 | 81.8359 | False |
| hidden | 0.0001 | 1 | 80.6641 | 80.8594 | 80.7617 | False |
| hidden | 0.0004 | 1 | 82.1289 | 83.2031 | 82.6660 | True |
| hidden | 5e-05 | 1 | 79.1016 | 77.3438 | 78.2227 | False |
| hidden | 0.0003 | 1 | 81.4453 | 83.3984 | 82.4219 | False |
| hidden | 0.0008 | 1 | 82.0312 | 83.1055 | 82.5684 | False |
| hidden_budget | 0.0002 | 1 | 80.8594 | 81.9336 | 81.3965 | False |
| hidden_budget | 0.0001 | 1 | 80.4688 | 81.0547 | 80.7617 | False |
| hidden_budget | 0.0004 | 1 | 81.4453 | 84.0820 | 82.7637 | True |
| hidden_budget | 5e-05 | 1 | 78.6133 | 77.7344 | 78.1738 | False |
| hidden_budget | 0.0003 | 1 | 81.9336 | 83.4961 | 82.7148 | False |
| hidden_budget | 0.0008 | 1 | 82.7148 | 82.5195 | 82.6172 | False |
| hidden_both | 0.0002 | 1 | 81.9336 | 82.6172 | 82.2754 | False |
| hidden_both | 0.0002 | 0.25 | 82.2266 | 82.9102 | 82.5684 | False |
| hidden_both | 0.0001 | 1 | 80.9570 | 81.5430 | 81.2500 | False |
| hidden_both | 0.0001 | 0.25 | 81.7383 | 81.8359 | 81.7871 | False |
| hidden_both | 0.0004 | 1 | 83.4961 | 83.6914 | 83.5938 | True |
| hidden_both | 0.0004 | 0.25 | 82.3242 | 83.7891 | 83.0566 | False |

## New confirmation set

| Role | 7300 | 7301 | 7302 | 7303 | 7304 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| tuned_hidden | 82.3730 | 83.4961 | 82.3242 | 82.7148 | 83.1543 | 82.8125 |
| tuned_budget | 82.0801 | 82.3730 | 82.0312 | 83.0078 | 83.5938 | 82.6172 |
| tuned_heu | 83.4961 | 83.4473 | 82.7637 | 82.8125 | 83.8379 | 83.2715 |
| anchor_budget | 80.8105 | 80.1758 | 80.3711 | 80.6152 | 80.4199 | 80.4785 |
| anchor_heu | 81.0059 | 82.3730 | 82.0312 | 81.0547 | 82.1777 | 81.7285 |

| Paired contrast | Mean delta (pp) | SD | Marginal 95% CI | Family-5 95% CI | Identity |
|---|---:|---:|---|---|---|
| tuned_heu − tuned_hidden | 0.4590 | 0.4691 | [-0.1235, 1.0415] | [-0.5069, 1.4249] | False |
| tuned_heu − tuned_budget | 0.6543 | 0.6428 | [-0.1438, 1.4524] | [-0.6692, 1.9778] | False |
| tuned_heu − anchor_heu | 1.5430 | 0.6771 | [0.7022, 2.3837] | [0.1488, 2.9371] | False |
| tuned_budget − anchor_budget | 2.1387 | 0.7287 | [1.2339, 3.0434] | [0.6383, 3.6390] | False |
| anchor_heu − anchor_budget | 1.2500 | 0.8793 | [0.1582, 2.3418] | [-0.5605, 3.0605] | False |

Identity comparisons use the same saved fit and are not independent replications.

- Two-seed development selection; limited, differently shaped search grids.
- Intervals cover seed variability conditional on the datasets, not dataset sampling uncertainty.
- Fresh project holdout excludes previous tables; not proof of absence from model pretraining.
- Different old/new test populations prohibit interpreting cross-study absolute-score differences as tuning gains.
- Training/clipping diagnostics are descriptive and do not prove a causal explanation.
