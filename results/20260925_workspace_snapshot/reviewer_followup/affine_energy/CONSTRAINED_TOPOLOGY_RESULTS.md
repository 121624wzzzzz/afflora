# Constrained A-LoRA topology sweep

Updated: 2026-07-13T03:52:43+08:00

All treatment rows use joint-from-scratch hidden LoRA r4 + A-LoRA r16, seed 42, tau=0.00625, lambda=100.

| model | topology | input rho | output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama31_8b | hidden baseline | - | - | 26.6867% | - | 74.8294% | - |
| llama31_8b | output | - | 0.00623416 | 26.6266% | -0.0601 pp | 74.7536% | -0.0758 pp |
| llama31_8b | input | 0.00625149 | - | 26.7668% | +0.0801 pp | 75.2085% | +0.3791 pp |
| llama31_8b | decoupled | 0.00625333 | 0.00620534 | 26.9069% | +0.2202 pp | 75.4359% | +0.6065 pp |
| llama32_3b | hidden baseline | - | - | 13.8739% | - | 57.0887% | - |
| llama32_3b | input | 0.00622455 | - | 14.0541% | +0.1802 pp | 57.0887% | +0.0000 pp |
| llama32_3b | output | - | 0.00629217 | 14.2342% | +0.3604 pp | 56.5580% | -0.5307 pp |
| llama32_3b | decoupled | 0.00624543 | 0.00628990 | 14.2943% | +0.4204 pp | 56.5580% | -0.5307 pp |
| llama32_3b | shared | 0.00624311 | shared | 14.2743% | +0.4004 pp | 56.2547% | -0.8340 pp |
