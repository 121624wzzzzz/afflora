# Qwen paired multi-seed and scale extension

Updated: 2026-07-13T13:38:44+08:00

Training: MetaMathQA-40K, one epoch, effective batch 16, hidden LoRA r4. A-LoRA uses rank 16 and scale 1. Constrained rows use tau=0.00625 and lambda=100.

## Paired Qwen2.5-1.5B input-constrained confirmation

| seed | hidden MATH | constrained MATH | paired delta | hidden GSM8K | constrained GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 35.5956% | 35.6156% | +0.0200 pp | 69.1433% | 70.5080% | +1.3647 pp | 0.00625285 |
| 43 | 36.0561% | 36.0360% | -0.0200 pp | 70.3563% | 70.9629% | +0.6065 pp | 0.00615076 |
| 44 | 36.1161% | 36.1762% | +0.0601 pp | 70.5838% | 70.0531% | -0.5307 pp | 0.00620268 |
| mean +/- std | 35.9226 +/- 0.2848 | 35.9426 +/- 0.2917 | 0.0200 +/- 0.0400 pp | 70.0278 +/- 0.7744 | 70.5080 +/- 0.4549 | 0.4802 +/- 0.9540 pp | - |

## Qwen 3B/4B scale extension (seed 42)

| model | configuration | rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen25_3b | hidden r4 | - | 41.9820% | - | 78.0895% | - |
| qwen25_3b | shared unconstrained | 0.05190600 | 42.1421% | +0.1602 pp | 77.7862% | -0.3033 pp |
| qwen25_3b | input constrained | 0.00613273 | 42.0020% | +0.0200 pp | 77.4071% | -0.6823 pp |
| qwen25_3b | shared constrained | 0.00617909 | 42.0220% | +0.0400 pp | 78.0895% | +0.0000 pp |
| qwen3_4b | hidden r4 | - | 48.0280% | - | 84.3821% | - |
| qwen3_4b | shared unconstrained | 0.10358613 | 47.9079% | -0.1201 pp | 84.3821% | +0.0000 pp |
| qwen3_4b | input constrained | 0.00624457 | 47.5275% | -0.5005 pp | 84.3063% | -0.0758 pp |
| qwen3_4b | shared constrained | 0.00624892 | 47.9279% | -0.1001 pp | 84.0788% | -0.3033 pp |
