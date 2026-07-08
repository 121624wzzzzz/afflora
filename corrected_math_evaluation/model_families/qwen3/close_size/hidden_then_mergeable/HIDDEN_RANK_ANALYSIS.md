# Qwen3 close-size hidden LoRA rank analysis

Updated: 2026-07-07T19:48:47+08:00

This is the Qwen3 main-structure stage before running mergeable AffLoRA.
MATH reports the clean-4,995 score.

## qwen3_06b

| hidden rank | MATH | GSM8K | mean(MATH,GSM8K) |
| ---: | ---: | ---: | ---: |
| 1 | 31.2913% | 63.6846% | 47.4880% |
| 2 | 31.5115% | 63.0781% | 47.2948% |
| 4 | 31.5315% | 63.3813% | 47.4564% |
| 8 | 31.8719% | 63.6088% | 47.7403% |
| 16 | 31.2513% | 63.6846% | 47.4679% |

### Ranking by MATH

| rank by MATH | hidden rank | MATH | GSM8K |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 31.8719% | 63.6088% |
| 2 | 4 | 31.5315% | 63.3813% |
| 3 | 2 | 31.5115% | 63.0781% |
| 4 | 1 | 31.2913% | 63.6846% |
| 5 | 16 | 31.2513% | 63.6846% |

### Suggested mergeable follow-up hidden ranks

- hidden hr8
- hidden hr1

Rationale: best MATH hr8, best GSM8K hr1, best average hr8.

## qwen3_17b

| hidden rank | MATH | GSM8K | mean(MATH,GSM8K) |
| ---: | ---: | ---: | ---: |
| 1 | 41.9620% | 75.3601% | 58.6610% |
| 2 | 42.1221% | 74.9810% | 58.5516% |
| 4 | 42.1421% | 75.1327% | 58.6374% |
| 8 | 41.4414% | 76.1941% | 58.8178% |
| 16 | 41.5015% | 76.1941% | 58.8478% |

### Ranking by MATH

| rank by MATH | hidden rank | MATH | GSM8K |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 42.1421% | 75.1327% |
| 2 | 2 | 42.1221% | 74.9810% |
| 3 | 1 | 41.9620% | 75.3601% |
| 4 | 16 | 41.5015% | 76.1941% |
| 5 | 8 | 41.4414% | 76.1941% |

### Suggested mergeable follow-up hidden ranks

- hidden hr4
- hidden hr8
- hidden hr16

Rationale: best MATH hr4, best GSM8K hr8, best average hr16.

