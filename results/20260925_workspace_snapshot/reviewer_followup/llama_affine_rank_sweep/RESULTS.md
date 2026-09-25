# Llama A-LoRA rank screening

Updated: 2026-07-11T23:53:52+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 8 |

## Held-out MetaMathQA answer-token loss

| model | placement | hidden rank | A rank | hidden CE | treatment CE | ΔCE | PPL | checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| llama31_8b | lmhead | 4 | 4 | 0.19353316 | 0.19361610 | 0.00008294 | 1.21363028 | new |
| llama31_8b | lmhead | 4 | 8 | 0.19353316 | 0.19352965 | -0.00000350 | 1.21352537 | new |
| llama31_8b | lmhead | 4 | 16 | 0.19353316 | 0.19330226 | -0.00023089 | 1.21324946 | reused |
| llama31_8b | lmhead | 4 | 32 | 0.19353316 | 0.19359910 | 0.00006594 | 1.21360965 | new |
| llama32_3b | mergeable | 4 | 4 | 0.24083799 | 0.24072984 | -0.00010815 | 1.27217730 | new |
| llama32_3b | mergeable | 4 | 8 | 0.24083799 | 0.24047534 | -0.00036265 | 1.27185357 | new |
| llama32_3b | mergeable | 4 | 16 | 0.24083799 | 0.24067378 | -0.00016422 | 1.27210598 | reused |
| llama32_3b | mergeable | 4 | 32 | 0.24083799 | 0.24045827 | -0.00037973 | 1.27183185 | new |
