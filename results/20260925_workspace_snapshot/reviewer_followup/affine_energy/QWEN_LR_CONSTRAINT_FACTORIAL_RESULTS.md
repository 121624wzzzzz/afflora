# Qwen A-LoRA LR x energy-constraint factorial

Updated: 2026-07-14T16:39:32+08:00

Selection seed: 42. Training uses MetaMathQA-40K for one epoch, effective batch 16, hidden LoRA r4, and A-LoRA r16 scale 1.
Every new row has full 5,000-example MATH and 1,319-example GSM8K evaluation.
Constrained input uses tau=0.0125; constrained output uses tau=0.00625; both use lambda=100. Output-only beta is disabled.

## qwen25_3b

| topology | constraint | A LR scale | rho | MATH | delta | GSM8K | delta | mean delta | worst delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| - | hidden r4 | - | - | 41.9820% | - | 78.0895% | - | - | - |
| input | none | 0.25 | 0.02400924 | 42.0420% | +0.0601 pp | 78.0895% | +0.0000 pp | +0.0300 pp | +0.0000 pp |
| input | none | 0.5 | 0.03542249 | 42.3223% | +0.3403 pp | 78.0136% | -0.0758 pp | +0.1323 pp | -0.0758 pp |
| input | none | 1 | 0.05409898 | 42.0821% | +0.1001 pp | 78.2411% | +0.1516 pp | +0.1259 pp | +0.1001 pp |
| input | hinge | 0.25 | 0.01211552 | 42.3023% | +0.3203 pp | 78.2411% | +0.1516 pp | +0.2360 pp | +0.1516 pp |
| input | hinge | 0.5 | 0.01207081 | 42.4024% | +0.4204 pp | 77.9378% | -0.1516 pp | +0.1344 pp | -0.1516 pp |
| input | hinge | 1 | 0.01221454 | 41.7818% | -0.2002 pp | 79.1509% | +1.0614 pp | +0.4306 pp | -0.2002 pp |

| output | none | 0.25 | 0.00271556 | 41.9620% | -0.0200 pp | 78.6960% | +0.6065 pp | +0.2933 pp | -0.0200 pp |
| output | none | 0.5 | 0.00480286 | 42.1021% | +0.1201 pp | 78.1653% | +0.0758 pp | +0.0980 pp | +0.0758 pp |
| output | none | 1 | 0.00873199 | 42.2222% | +0.2402 pp | 78.0895% | +0.0000 pp | +0.1201 pp | +0.0000 pp |
| output | hinge | 0.25 | 0.00270498 | 41.7618% | -0.2202 pp | 77.5588% | -0.5307 pp | -0.3755 pp | -0.5307 pp |
| output | hinge | 0.5 | 0.00479276 | 41.8619% | -0.1201 pp | 78.0895% | +0.0000 pp | -0.0601 pp | -0.1201 pp |
| output | hinge | 1 | 0.00622224 | 42.4224% | +0.4404 pp | 78.6202% | +0.5307 pp | +0.4856 pp | +0.4404 pp |

## qwen3_4b

| topology | constraint | A LR scale | rho | MATH | delta | GSM8K | delta | mean delta | worst delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| - | hidden r4 | - | - | 48.0280% | - | 84.3821% | - | - | - |
| input | none | 0.25 | 0.03411210 | 47.8278% | -0.2002 pp | 84.3821% | +0.0000 pp | -0.1001 pp | -0.2002 pp |
| input | none | 0.5 | 0.06191460 | 48.3283% | +0.3003 pp | 84.9128% | +0.5307 pp | +0.4155 pp | +0.3003 pp |
| input | none | 1 | 0.10266560 | 48.5085% | +0.4805 pp | 84.9128% | +0.5307 pp | +0.5056 pp | +0.4805 pp |
| input | hinge | 0.25 | 0.01241122 | 48.4284% | +0.4004 pp | 84.6854% | +0.3033 pp | +0.3518 pp | +0.3033 pp |
| input | hinge | 0.5 | 0.01249799 | 47.9279% | -0.1001 pp | 84.4579% | +0.0758 pp | -0.0121 pp | -0.1001 pp |
| input | hinge | 1 | 0.01249769 | 48.3083% | +0.2803 pp | 84.4579% | +0.0758 pp | +0.1780 pp | +0.0758 pp |

| output | none | 0.25 | 0.00233794 | 47.9479% | -0.0801 pp | 84.7612% | +0.3791 pp | +0.1495 pp | -0.0801 pp |
| output | none | 0.5 | 0.00431214 | 48.1081% | +0.0801 pp | 85.0644% | +0.6823 pp | +0.3812 pp | +0.0801 pp |
| output | none | 1 | 0.00816364 | 48.1882% | +0.1602 pp | 84.4579% | +0.0758 pp | +0.1180 pp | +0.0758 pp |
| output | hinge | 0.25 | 0.00233442 | 48.1281% | +0.1001 pp | 84.6854% | +0.3033 pp | +0.2017 pp | +0.1001 pp |
| output | hinge | 0.5 | 0.00429880 | 48.0881% | +0.0601 pp | 84.6854% | +0.3033 pp | +0.1817 pp | +0.0601 pp |
| output | hinge | 1 | 0.00623145 | 48.3483% | +0.3203 pp | 84.4579% | +0.0758 pp | +0.1981 pp | +0.0758 pp |

