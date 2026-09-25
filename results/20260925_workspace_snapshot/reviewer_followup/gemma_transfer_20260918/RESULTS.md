# Gemma-2-9B Base cross-family stacking study

## Development search

| Task | Arm | LR | Seed 7400 | Seed 7401 | Mean | Selected |
|---|---|---:|---:|---:|---:|---|
| wikisql | hidden | 0.0002 | 85.4492 | 84.1797 | 84.8145 | True |
| wikisql | hidden | 0.0001 | 84.0820 | 83.3984 | 83.7402 | False |
| wikisql | hidden | 0.0004 | 84.1797 | 83.8867 | 84.0332 | False |
| wikisql | hidden | 5e-05 | 81.2500 | 81.8359 | 81.5430 | False |
| wikisql | hidden | 0.0003 | 85.1562 | 83.5938 | 84.3750 | False |
| wikisql | hidden | 0.0008 | 84.2773 | 84.8633 | 84.5703 | False |
| wikisql | hidden_budget | 0.0002 | 84.4727 | 83.8867 | 84.1797 | False |
| wikisql | hidden_budget | 0.0001 | 82.6172 | 83.4961 | 83.0566 | False |
| wikisql | hidden_budget | 0.0004 | 85.0586 | 83.3984 | 84.2285 | True |
| wikisql | hidden_budget | 5e-05 | 81.2500 | 81.8359 | 81.5430 | False |
| wikisql | hidden_budget | 0.0003 | 84.1797 | 83.9844 | 84.0820 | False |
| wikisql | hidden_budget | 0.0008 | 83.6914 | 84.2773 | 83.9844 | False |
| wikisql | hidden_both | 0.0002 | 85.0586 | 84.2773 | 84.6680 | True |
| wikisql | hidden_both | 0.0001 | 82.0312 | 83.7891 | 82.9102 | False |
| wikisql | hidden_both | 0.0004 | 84.9609 | 83.3984 | 84.1797 | False |
| wikisql | hidden_both | 5e-05 | 82.3242 | 81.6406 | 81.9824 | False |
| wikisql | hidden_both | 0.0003 | 84.6680 | 83.5938 | 84.1309 | False |
| wikisql | hidden_both | 0.0008 | 83.0078 | 83.2031 | 83.1055 | False |
| trec50 | hidden | 0.0002 | 83.5938 | 84.3750 | 83.9844 | False |
| trec50 | hidden | 0.0001 | 82.4219 | 82.0312 | 82.2266 | False |
| trec50 | hidden | 0.0004 | 86.3281 | 87.8906 | 87.1094 | True |
| trec50 | hidden | 5e-05 | 79.2969 | 77.7344 | 78.5156 | False |
| trec50 | hidden | 0.0003 | 86.7188 | 84.7656 | 85.7422 | False |
| trec50 | hidden | 0.0008 | 85.9375 | 85.9375 | 85.9375 | False |
| trec50 | hidden_budget | 0.0002 | 82.8125 | 83.5938 | 83.2031 | False |
| trec50 | hidden_budget | 0.0001 | 82.4219 | 80.8594 | 81.6406 | False |
| trec50 | hidden_budget | 0.0004 | 88.6719 | 85.5469 | 87.1094 | True |
| trec50 | hidden_budget | 5e-05 | 80.0781 | 77.3438 | 78.7109 | False |
| trec50 | hidden_budget | 0.0003 | 85.1562 | 85.9375 | 85.5469 | False |
| trec50 | hidden_budget | 0.0008 | 84.7656 | 84.3750 | 84.5703 | False |
| trec50 | hidden_both | 0.0002 | 87.8906 | 83.9844 | 85.9375 | False |
| trec50 | hidden_both | 0.0001 | 83.2031 | 83.9844 | 83.5938 | False |
| trec50 | hidden_both | 0.0004 | 87.8906 | 87.1094 | 87.5000 | True |
| trec50 | hidden_both | 5e-05 | 81.2500 | 76.9531 | 79.1016 | False |
| trec50 | hidden_both | 0.0003 | 84.7656 | 87.5000 | 86.1328 | False |
| trec50 | hidden_both | 0.0008 | 2.7344 | 83.5938 | 43.1641 | False |

## Confirmation

| Task | Arm | 7500 | 7501 | 7502 | 7503 | 7504 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| wikisql | base | — | — | — | — | — | 19.7266 |
| wikisql | hidden | 82.2754 | 83.9355 | 83.6426 | 83.8379 | 83.0566 | 83.3496 |
| wikisql | hidden_budget | 83.9355 | 84.1309 | 84.0332 | 83.1055 | 85.1562 | 84.0723 |
| wikisql | hidden_both | 81.7871 | 84.2773 | 83.7891 | 84.7656 | 84.4238 | 83.8086 |
| trec50 | base | — | — | — | — | — | 27.8000 |
| trec50 | hidden | 87.8000 | 90.8000 | 89.4000 | 89.6000 | 89.2000 | 89.3600 |
| trec50 | hidden_budget | 88.2000 | 89.4000 | 89.4000 | 88.6000 | 90.2000 | 89.1600 |
| trec50 | hidden_both | 88.4000 | 88.2000 | 88.4000 | 84.8000 | 89.0000 | 87.7600 |

| Task | Contrast | Mean pp | Marginal seed 95% CI | Family-4 seed 95% CI | Positive seeds |
|---|---|---:|---|---|---:|
| wikisql | hidden_both − hidden | +0.4590 | [-0.4306,+1.3486] | [-0.9235,+1.8414] | 4/5 |
| wikisql | hidden_both − hidden_budget | -0.2637 | [-1.9799,+1.4525] | [-2.9307,+2.4034] | 2/5 |
| trec50 | hidden_both − hidden | -1.6000 | [-4.2631,+1.0631] | [-5.7385,+2.5385] | 1/5 |
| trec50 | hidden_both − hidden_budget | -1.4000 | [-3.2164,+0.4164] | [-4.2227,+1.4227] | 1/5 |

## Limitations

- Finite common LR grid; two development seeds, five confirmation seeds; no globally optimal hyperparameter claim.
- Seed t intervals condition on fixed data and selected configurations and use a small-sample distributional assumption.
- Supplementary cluster bootstrap fixes the five fitted models; not joint seed, data and model-selection uncertainty.
- Shared public benchmark splits previously evaluated on other families; not globally unseen project data or proof against pretraining exposure.
- E has bias and U does not; tied frozen weights with independent effective E/U updates; no pure placement or preserved-tying claim.
- Source checkpoint is FP32; all arms use the same BF16-rounded frozen base, cast back to FP32 for evaluation.
