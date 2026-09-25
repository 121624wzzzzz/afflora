# Qwen 3B/4B output-only affine-energy tau sweep

Updated: 2026-07-13T21:43:41+08:00

All treatment rows use jointly trained hidden LoRA r4 + output-only A-LoRA r16 scale 1, seed 42, lambda=100, and no output beta. The multiplicative output adapter is mergeable into lm_head.weight.

## qwen25_3b

| topology | tau | measured output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden r4 | - | - | 41.9820% | - | 78.0895% | - |
| output | 0.05 | 0.00882606 | 42.1021% | +0.1201 pp | 78.1653% | +0.0758 pp |
| output | 0.025 | 0.00872387 | 41.9219% | -0.0601 pp | 78.7718% | +0.6823 pp |
| output | 0.0125 | 0.00873643 | 41.7818% | -0.2002 pp | 78.1653% | +0.0758 pp |
| output | 0.00625 | 0.00622224 | 42.4224% | +0.4404 pp | 78.6202% | +0.5307 pp |

## qwen3_4b

| topology | tau | measured output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden r4 | - | - | 48.0280% | - | 84.3821% | - |
| output | 0.05 | 0.00814662 | 48.4885% | +0.4605 pp | 84.2305% | -0.1516 pp |
| output | 0.025 | 0.00815237 | 48.1081% | +0.0801 pp | 83.9272% | -0.4549 pp |
| output | 0.0125 | 0.00816126 | 48.1281% | +0.1001 pp | 84.8370% | +0.4549 pp |
| output | 0.00625 | 0.00623145 | 48.3483% | +0.3203 pp | 84.4579% | +0.0758 pp |
