# Constraint selected for the pure A-LoRA follow-up

Updated: 2026-07-12T19:08:25+08:00

Selection rule: maximize the equally weighted mean of full MATH-clean and GSM8K accuracy. 
Exact ties prefer the smaller lambda. All candidates are seed 42.

| selected | tau | lambda | MATH | GSM8K | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| yes | 0.00625000 | 100 | 26.6266% | 74.7536% | 50.6901% |
|  | 0.00729014 | 1000 | 26.5666% | 74.7536% | 50.6601% |
|  | 0.00950000 | 100 | 26.3664% | 74.9052% | 50.6358% |
|  | 0.00729014 | 300 | 26.4464% | 74.7536% | 50.6000% |
|  | 0.00729014 | 10 | 26.3263% | 74.8294% | 50.5779% |
|  | 0.00729014 | 30 | 26.3864% | 74.6778% | 50.5321% |
|  | 0.00729014 | 100 | 26.1461% | 74.9052% | 50.5257% |
|  | 0.00519388 | 100 | 27.0470% | 73.6164% | 50.3317% |
|  | 0.00850000 | 100 | 25.8659% | 74.7536% | 50.3097% |

Chosen tau: 0.00625000
Chosen lambda: 100
