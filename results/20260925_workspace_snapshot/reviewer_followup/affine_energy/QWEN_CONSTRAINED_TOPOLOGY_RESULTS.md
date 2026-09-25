# Qwen constrained A-LoRA topology sweep

Updated: 2026-07-13T05:20:48+08:00

All treatment rows use joint-from-scratch hidden LoRA r4 + A-LoRA r16, seed 42, tau=0.00625, lambda=100.

| model | topology | input rho | output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_06b | hidden baseline | - | - | 31.8719% | - | 63.7604% | - |
| qwen3_06b | shared unconstrained | 0.06029779 | shared | 31.7117% | -0.1602 pp | 63.7604% | +0.0000 pp |
| qwen3_06b | input constrained | 0.00624880 | - | 31.7317% | -0.1401 pp | 63.7604% | +0.0000 pp |
| qwen3_06b | output constrained | - | 0.00629462 | 31.1912% | -0.6807 pp | 63.9879% | +0.2274 pp |
| qwen3_06b | shared constrained | 0.00624495 | shared | 31.7918% | -0.0801 pp | 63.0781% | -0.6823 pp |
| qwen3_06b | decoupled constrained | 0.00623689 | 0.00629588 | 31.2513% | -0.6206 pp | 63.2297% | -0.5307 pp |
| qwen25_15b | hidden baseline | - | - | 35.5956% | - | 69.1433% | - |
| qwen25_15b | shared unconstrained | 0.05500340 | shared | 36.0961% | +0.5005 pp | 69.5224% | +0.3791 pp |
| qwen25_15b | input constrained | 0.00625285 | - | 35.6156% | +0.0200 pp | 70.5080% | +1.3647 pp |
| qwen25_15b | output constrained | - | 0.00624112 | 35.7958% | +0.2002 pp | 69.6740% | +0.5307 pp |
| qwen25_15b | shared constrained | 0.00625353 | shared | 35.4354% | -0.1602 pp | 69.3707% | +0.2274 pp |
| qwen25_15b | decoupled constrained | 0.00625324 | 0.00620101 | 35.9159% | +0.3203 pp | 68.7642% | -0.3791 pp |
