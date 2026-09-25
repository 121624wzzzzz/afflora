# Llama hidden-LoRA rank screening

Updated: 2026-07-11T22:15:32+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 10 |

## Held-out MetaMathQA answer-token loss

| model | hidden rank | seed | dev CE | dev PPL | checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| llama31_8b | 1 | 42 | 0.20173296 | 1.22352123 | new |
| llama31_8b | 2 | 42 | 0.19763404 | 1.21851638 | new |
| llama31_8b | 4 | 42 | 0.19353316 | 1.21352962 | reused |
| llama31_8b | 8 | 42 | 0.18943303 | 1.20856418 | new |
| llama31_8b | 16 | 42 | 0.18613108 | 1.20458014 | new |
| llama32_3b | 1 | 42 | 0.25579197 | 1.29148403 | new |
| llama32_3b | 2 | 42 | 0.24880585 | 1.28249301 | new |
| llama32_3b | 4 | 42 | 0.24083799 | 1.27231490 | reused |
| llama32_3b | 8 | 42 | 0.23310446 | 1.26251335 | new |
| llama32_3b | 16 | 42 | 0.22544340 | 1.25287813 | new |
