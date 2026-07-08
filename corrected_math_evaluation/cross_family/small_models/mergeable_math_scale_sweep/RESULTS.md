# Small-model mergeable AffLoRA math sweep results

Updated: 2026-07-06T00:26:41+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 45 |

## Per-run results

| config | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: |
| qwen3_06b_hidden_hr4 | 42 | 31.8719% | 63.7604% |
| qwen3_06b_hidden_hr4 | 43 | 32.0721% | 64.2153% |
| qwen3_06b_hidden_hr4 | 44 | 31.3514% | 63.9879% |
| qwen3_06b_mergeable_ar16_s0p25_hr4 | 42 | 31.4915% | 63.2297% |
| qwen3_06b_mergeable_ar16_s0p25_hr4 | 43 | 31.8719% | 65.1251% |
| qwen3_06b_mergeable_ar16_s0p25_hr4 | 44 | 31.0310% | 63.9879% |
| qwen3_06b_mergeable_ar16_s1_hr4 | 42 | 31.7117% | 63.7604% |
| qwen3_06b_mergeable_ar16_s1_hr4 | 43 | 31.9319% | 63.7604% |
| qwen3_06b_mergeable_ar16_s1_hr4 | 44 | 31.8118% | 63.6846% |
| qwen3_06b_mergeable_ar16_s4_hr4 | 42 | 31.5115% | 63.4572% |
| qwen3_06b_mergeable_ar16_s4_hr4 | 43 | 31.3914% | 64.6702% |
| qwen3_06b_mergeable_ar16_s4_hr4 | 44 | 30.8709% | 63.0781% |
| qwen3_06b_mergeable_ar16_s8_hr4 | 42 | 30.9109% | 63.1539% |
| qwen3_06b_mergeable_ar16_s8_hr4 | 43 | 31.9319% | 64.2153% |
| qwen3_06b_mergeable_ar16_s8_hr4 | 44 | 30.9309% | 63.5330% |
| qwen25_05b_hidden_hr4 | 42 | 21.7618% | 46.4746% |
| qwen25_05b_hidden_hr4 | 43 | 20.8809% | 47.6876% |
| qwen25_05b_hidden_hr4 | 44 | 20.7007% | 46.8537% |
| qwen25_05b_mergeable_ar16_s0p25_hr4 | 42 | 21.5215% | 46.7779% |
| qwen25_05b_mergeable_ar16_s0p25_hr4 | 43 | 20.5806% | 47.3844% |
| qwen25_05b_mergeable_ar16_s0p25_hr4 | 44 | 20.6807% | 46.7779% |
| qwen25_05b_mergeable_ar16_s1_hr4 | 42 | 21.5015% | 47.3086% |
| qwen25_05b_mergeable_ar16_s1_hr4 | 43 | 20.5606% | 47.0811% |
| qwen25_05b_mergeable_ar16_s1_hr4 | 44 | 21.0010% | 45.8681% |
| qwen25_05b_mergeable_ar16_s4_hr4 | 42 | 21.5616% | 47.4602% |
| qwen25_05b_mergeable_ar16_s4_hr4 | 43 | 21.2212% | 47.6876% |
| qwen25_05b_mergeable_ar16_s4_hr4 | 44 | 20.8008% | 46.0955% |
| qwen25_05b_mergeable_ar16_s8_hr4 | 42 | 21.2613% | 46.9295% |
| qwen25_05b_mergeable_ar16_s8_hr4 | 43 | 21.0611% | 48.4458% |
| qwen25_05b_mergeable_ar16_s8_hr4 | 44 | 21.4014% | 46.7020% |
| qwen25_15b_hidden_hr4 | 42 | 35.9159% | 70.7354% |
| qwen25_15b_hidden_hr4 | 43 | 36.2763% | 71.3419% |
| qwen25_15b_hidden_hr4 | 44 | 36.4364% | 70.2047% |
| qwen25_15b_mergeable_ar16_s0p25_hr4 | 42 | 35.7157% | 69.5224% |
| qwen25_15b_mergeable_ar16_s0p25_hr4 | 43 | 35.8358% | 70.2805% |
| qwen25_15b_mergeable_ar16_s0p25_hr4 | 44 | 36.1962% | 69.6740% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 42 | 36.3764% | 69.3707% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 43 | 36.4765% | 70.3563% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 44 | 36.2763% | 70.2047% |
| qwen25_15b_mergeable_ar16_s4_hr4 | 42 | 35.6757% | 69.3707% |
| qwen25_15b_mergeable_ar16_s4_hr4 | 43 | 36.5966% | 70.1289% |
| qwen25_15b_mergeable_ar16_s4_hr4 | 44 | 36.1161% | 69.5982% |
| qwen25_15b_mergeable_ar16_s8_hr4 | 42 | 36.3764% | 69.6740% |
| qwen25_15b_mergeable_ar16_s8_hr4 | 43 | 36.5566% | 71.1903% |
| qwen25_15b_mergeable_ar16_s8_hr4 | 44 | 36.0561% | 70.3563% |

## Three-seed aggregates

| config | completed seeds | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| qwen3_06b_hidden_hr4 | 3/3 | 31.7651% ± 0.3720 | 63.9879% ± 0.2274 |
| qwen3_06b_mergeable_ar16_s0p25_hr4 | 3/3 | 31.4648% ± 0.4211 | 64.1142% ± 0.9540 |
| qwen3_06b_mergeable_ar16_s1_hr4 | 3/3 | 31.8185% ± 0.1103 | 63.7352% ± 0.0438 |
| qwen3_06b_mergeable_ar16_s4_hr4 | 3/3 | 31.2579% ± 0.3405 | 63.7352% ± 0.8317 |
| qwen3_06b_mergeable_ar16_s8_hr4 | 3/3 | 31.2579% ± 0.5838 | 63.6341% ± 0.5379 |
| qwen25_05b_hidden_hr4 | 3/3 | 21.1144% ± 0.5678 | 47.0053% ± 0.6206 |
| qwen25_05b_mergeable_ar16_s0p25_hr4 | 3/3 | 20.9276% ± 0.5168 | 46.9800% ± 0.3502 |
| qwen25_05b_mergeable_ar16_s1_hr4 | 3/3 | 21.0210% ± 0.4708 | 46.7526% ± 0.7744 |
| qwen25_05b_mergeable_ar16_s4_hr4 | 3/3 | 21.1945% ± 0.3811 | 47.0811% ± 0.8611 |
| qwen25_05b_mergeable_ar16_s8_hr4 | 3/3 | 21.2412% ± 0.1711 | 47.3591% ± 0.9479 |
| qwen25_15b_hidden_hr4 | 3/3 | 36.2095% ± 0.2666 | 70.7607% ± 0.5690 |
| qwen25_15b_mergeable_ar16_s0p25_hr4 | 3/3 | 35.9159% ± 0.2500 | 69.8256% ± 0.4012 |
| qwen25_15b_mergeable_ar16_s1_hr4 | 3/3 | 36.3764% ± 0.1001 | 69.9773% ± 0.5307 |
| qwen25_15b_mergeable_ar16_s4_hr4 | 3/3 | 36.1295% ± 0.4606 | 69.6993% ± 0.3891 |
| qwen25_15b_mergeable_ar16_s8_hr4 | 3/3 | 36.3297% ± 0.2535 | 70.4069% ± 0.7594 |
