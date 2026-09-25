# Qwen3.5-4B Base cross-family stacking study

## Development search

| Task | Arm | LR | Seed 7600 | Seed 7601 | Mean | Selected |
|---|---|---:|---:|---:|---:|---|
| wikisql | hidden | 0.0002 | 83.3008 | 83.4961 | 83.3984 | False |
| wikisql | hidden | 0.0001 | 83.1055 | 83.3008 | 83.2031 | False |
| wikisql | hidden | 0.0004 | 84.1797 | 85.3516 | 84.7656 | True |
| wikisql | hidden | 5e-05 | 80.2734 | 80.6641 | 80.4688 | False |
| wikisql | hidden | 0.0003 | 83.3984 | 84.3750 | 83.8867 | False |
| wikisql | hidden | 0.0008 | 83.7891 | 83.4961 | 83.6426 | False |
| wikisql | hidden_budget | 0.0002 | 82.7148 | 83.7891 | 83.2520 | False |
| wikisql | hidden_budget | 0.0001 | 82.9102 | 83.2031 | 83.0566 | False |
| wikisql | hidden_budget | 0.0004 | 83.0078 | 84.3750 | 83.6914 | True |
| wikisql | hidden_budget | 5e-05 | 79.9805 | 80.4688 | 80.2246 | False |
| wikisql | hidden_budget | 0.0003 | 83.8867 | 83.4961 | 83.6914 | False |
| wikisql | hidden_budget | 0.0008 | 82.5195 | 84.2773 | 83.3984 | False |
| wikisql | hidden_both | 0.0002 | 83.9844 | 83.4961 | 83.7402 | False |
| wikisql | hidden_both | 0.0001 | 83.0078 | 83.6914 | 83.3496 | False |
| wikisql | hidden_both | 0.0004 | 82.5195 | 83.2031 | 82.8613 | False |
| wikisql | hidden_both | 5e-05 | 81.2500 | 81.7383 | 81.4941 | False |
| wikisql | hidden_both | 0.0003 | 84.1797 | 83.1055 | 83.6426 | False |
| wikisql | hidden_both | 0.0008 | 83.4961 | 84.3750 | 83.9355 | True |
| trec50 | hidden | 0.0002 | 84.3750 | 85.9375 | 85.1562 | False |
| trec50 | hidden | 0.0001 | 81.6406 | 82.8125 | 82.2266 | False |
| trec50 | hidden | 0.0004 | 87.1094 | 85.9375 | 86.5234 | True |
| trec50 | hidden | 5e-05 | 78.5156 | 78.5156 | 78.5156 | False |
| trec50 | hidden | 0.0003 | 87.5000 | 84.3750 | 85.9375 | False |
| trec50 | hidden | 0.0008 | 84.3750 | 87.5000 | 85.9375 | False |
| trec50 | hidden_budget | 0.0002 | 83.5938 | 83.9844 | 83.7891 | False |
| trec50 | hidden_budget | 0.0001 | 80.8594 | 82.0312 | 81.4453 | False |
| trec50 | hidden_budget | 0.0004 | 87.1094 | 86.7188 | 86.9141 | True |
| trec50 | hidden_budget | 5e-05 | 78.1250 | 78.9062 | 78.5156 | False |
| trec50 | hidden_budget | 0.0003 | 85.9375 | 83.9844 | 84.9609 | False |
| trec50 | hidden_budget | 0.0008 | 87.5000 | 85.1562 | 86.3281 | False |
| trec50 | hidden_both | 0.0002 | 87.1094 | 85.1562 | 86.1328 | False |
| trec50 | hidden_both | 0.0001 | 80.4688 | 83.2031 | 81.8359 | False |
| trec50 | hidden_both | 0.0004 | 87.8906 | 83.9844 | 85.9375 | False |
| trec50 | hidden_both | 5e-05 | 79.2969 | 78.9062 | 79.1016 | False |
| trec50 | hidden_both | 0.0003 | 87.8906 | 85.9375 | 86.9141 | True |
| trec50 | hidden_both | 0.0008 | 42.5781 | 84.7656 | 63.6719 | False |

## Confirmation

| Task | Arm | 7700 | 7701 | 7702 | 7703 | 7704 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| wikisql | base | — | — | — | — | — | 43.3105 |
| wikisql | hidden | 84.9609 | 84.3750 | 83.6426 | 83.6426 | 84.5703 | 84.2383 |
| wikisql | hidden_budget | 85.4004 | 84.5215 | 84.0820 | 82.8125 | 85.1562 | 84.3945 |
| wikisql | hidden_both | 83.4961 | 82.8613 | 83.7891 | 83.7891 | 83.1055 | 83.4082 |
| trec50 | base | — | — | — | — | — | 62.6000 |
| trec50 | hidden | 89.0000 | 88.0000 | 88.2000 | 90.6000 | 90.0000 | 89.1600 |
| trec50 | hidden_budget | 88.8000 | 88.2000 | 89.0000 | 89.6000 | 90.2000 | 89.1600 |
| trec50 | hidden_both | 88.8000 | 88.2000 | 87.6000 | 89.8000 | 89.0000 | 88.6800 |

| Task | Contrast | Mean pp | Marginal seed 95% CI | Family-4 seed 95% CI | Positive seeds |
|---|---|---:|---|---|---:|
| wikisql | hidden_both − hidden | -0.8301 | [-1.9373,+0.2771] | [-2.5507,+0.8905] | 2/5 |
| wikisql | hidden_both − hidden_budget | -0.9863 | [-2.6009,+0.6282] | [-3.4954,+1.5228] | 1/5 |
| trec50 | hidden_both − hidden | -0.4800 | [-1.0781,+0.1181] | [-1.4094,+0.4494] | 1/5 |
| trec50 | hidden_both − hidden_budget | -0.4800 | [-1.4191,+0.4591] | [-1.9393,+0.9793] | 1/5 |

## Limitations

- Finite common LR grid; two development seeds, five confirmation seeds; no globally optimal hyperparameter claim.
- Seed t intervals condition on fixed data and selected configurations and use a small-sample distributional assumption.
- Supplementary cluster bootstrap fixes the five fitted models; not joint seed, data and model-selection uncertainty.
- Shared public benchmark splits previously evaluated on other families; not globally unseen project data or proof against pretraining exposure.
- E has bias and U does not; tied frozen weights with independent effective E/U updates; no pure placement or preserved-tying claim.
- Source mixed BF16/FP32 checkpoint is preserved exactly; all arms use the same frozen base, cast to FP32 for evaluation.
