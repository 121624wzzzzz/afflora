# Pure Llama-3.1 output A-LoRA energy experiment

Updated: 2026-07-12T20:32:31+08:00

Both rows are standalone output A-LoRA r16, seed 42, with no hidden LoRA.

| config | tau | lambda | final rho | MATH | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen Llama-3.1-8B Base | - | - | - | 11.4715% | 30.3260% |
| pure A-LoRA unconstrained | 0.00000000 | 0 | 0.10195406 | 14.6947% | 17.5891% |
| pure A-LoRA constrained | 0.00625000 | 100 | 0.01138808 | 13.3934% | 24.6399% |

Constrained minus unconstrained:

- MATH: -1.3013 pp
- GSM8K: +7.0508 pp

A-LoRA minus frozen Base:

- Unconstrained: MATH +3.2232 pp; GSM8K -12.7369 pp
- Constrained: MATH +1.9219 pp; GSM8K -5.6861 pp
