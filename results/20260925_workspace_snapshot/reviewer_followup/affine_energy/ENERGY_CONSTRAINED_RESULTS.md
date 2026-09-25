# Energy-constrained Llama-3.1 A-LoRA result

Updated: 2026-07-12T15:02:09+08:00

- Constraint: tau=0.00729014, lambda=100.0.
- Exact final centered-output rho: 0.00724376.

| config | MATH | Δ vs hidden | GSM8K | Δ vs hidden |
| --- | ---: | ---: | ---: | ---: |
| hidden r4 | 26.6867% | - | 74.8294% | - |
| unconstrained A r16 | 26.0661% | -0.6206 pp | 74.4503% | -0.3791 pp |
| energy-constrained A r16 | 26.1461% | -0.5405 pp | 74.9052% | +0.0758 pp |
