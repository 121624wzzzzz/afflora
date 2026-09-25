# Frozen-hidden staged A-LoRA result

Updated: 2026-07-12T23:12:30+08:00

The hidden LoRA r4 checkpoint is loaded and frozen exactly; only output A-LoRA r16 trains.

| config | final rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden r4 baseline | - | 26.6867% | - | 74.8294% | - |
| joint-from-scratch hidden+A | 0.01006077 | 26.0661% | -0.6206 pp | 74.4503% | -0.3791 pp |
| staged A unconstrained | 0.01118211 | 26.6867% | +0.0000 pp | 73.9955% | -0.8340 pp |
| staged A constrained | 0.00602633 | 26.8869% | +0.2002 pp | 74.1471% | -0.6823 pp |

Frozen-hidden integrity:

- staged A unconstrained: hidden adapter tensor-exact = True
- staged A constrained: hidden adapter tensor-exact = True
