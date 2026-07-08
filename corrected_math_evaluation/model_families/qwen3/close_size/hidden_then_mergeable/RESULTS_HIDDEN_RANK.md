# Qwen3 close-size hidden LoRA rank sweep

Updated: 2026-07-07T19:22:52+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 10 |

## Per-run results

| model | hidden rank | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: | ---: |
| qwen3_06b | 1 | 42 | 31.2913% | 63.6846% |
| qwen3_06b | 2 | 42 | 31.5115% | 63.0781% |
| qwen3_06b | 4 | 42 | 31.5315% | 63.3813% |
| qwen3_06b | 8 | 42 | 31.8719% | 63.6088% |
| qwen3_06b | 16 | 42 | 31.2513% | 63.6846% |
| qwen3_17b | 1 | 42 | 41.9620% | 75.3601% |
| qwen3_17b | 2 | 42 | 42.1221% | 74.9810% |
| qwen3_17b | 4 | 42 | 42.1421% | 75.1327% |
| qwen3_17b | 8 | 42 | 41.4414% | 76.1941% |
| qwen3_17b | 16 | 42 | 41.5015% | 76.1941% |
