# Qwen second-tau multi-seed dose results

Updated: 2026-07-15T16:24:52+08:00

Training: MetaMathQA-40K, one epoch, hidden LoRA r4 + A-LoRA r16, A-LoRA LR scale 1, lambda=100.
Shared is tied/mergeable. Output-only has no beta and is mergeable.
All comparisons are paired by model, seed, topology, rank, and optimizer settings.

## qwen25_3b / shared

Current tau=0.025; alternate tau=0.0125.

| seed | current MATH | alternate MATH | delta | current GSM8K | alternate GSM8K | delta | alternate rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.2823% | 42.3624% | +0.0801 pp | 78.1653% | 78.6960% | +0.5307 pp | 0.01222545 |
| 43 | 41.2613% | 41.1411% | -0.1201 pp | 78.9992% | 78.2411% | -0.7582 pp | 0.01237176 |
| 44 | 41.1411% | 41.2613% | +0.1201 pp | 77.8620% | 78.0136% | +0.1516 pp | 0.01223316 |
| 45 | 41.7417% | 41.7017% | -0.0400 pp | 78.6202% | 78.4685% | -0.1516 pp | 0.01229608 |
| 46 | 41.8418% | 41.8418% | +0.0000 pp | 77.5588% | 78.3927% | +0.8340 pp | 0.01211383 |

- alternate-current task mean: +0.0647 +/- 0.3416 pp; 95% CI [-0.3595, +0.4888]; positive seeds 3/5
- current-unconstrained task mean: +0.1701 +/- 0.3086 pp; 95% CI [-0.2131, +0.5533]
- alternate-unconstrained task mean: +0.2348 +/- 0.3566 pp; 95% CI [-0.2080, +0.6775]
- alternate-hidden task mean: +0.2421 +/- 0.4188 pp; 95% CI [-0.2778, +0.7621]

| dose | two-task mean accuracy | std across seeds |
| --- | ---: | ---: |
| hidden | 59.7699% | 0.2676 |
| unconstrained | 59.7773% | 0.1589 |
| tau=0.025 | 59.9474% | 0.3256 |
| tau=0.0125 | 60.0120% | 0.3630 |

## qwen25_3b / output

Current tau=0.00625; alternate tau=0.003125.

| seed | current MATH | alternate MATH | delta | current GSM8K | alternate GSM8K | delta | alternate rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.4224% | 41.9820% | -0.4404 pp | 78.6202% | 78.0136% | -0.6065 pp | 0.00312309 |
| 43 | 41.3614% | 41.6817% | +0.3203 pp | 78.5444% | 78.1653% | -0.3791 pp | 0.00314399 |
| 44 | 41.2613% | 41.2012% | -0.0601 pp | 78.3927% | 78.2411% | -0.1516 pp | 0.00314465 |
| 45 | 41.5015% | 41.8018% | +0.3003 pp | 77.8620% | 77.8620% | +0.0000 pp | 0.00312323 |
| 46 | 41.4014% | 41.8418% | +0.4404 pp | 78.5444% | 78.9992% | +0.4549 pp | 0.00313206 |

- alternate-current task mean: -0.0122 +/- 0.3565 pp; 95% CI [-0.4548, +0.4304]; positive seeds 2/5
- current-unconstrained task mean: +0.1279 +/- 0.1605 pp; 95% CI [-0.0715, +0.3272]
- alternate-unconstrained task mean: +0.1157 +/- 0.2605 pp; 95% CI [-0.2078, +0.4392]
- alternate-hidden task mean: +0.2091 +/- 0.3430 pp; 95% CI [-0.2168, +0.6350]

| dose | two-task mean accuracy | std across seeds |
| --- | ---: | ---: |
| hidden | 59.7699% | 0.2676 |
| unconstrained | 59.8633% | 0.1851 |
| tau=0.00625 | 59.9912% | 0.3184 |
| tau=0.003125 | 59.9790% | 0.2676 |

## qwen3_4b / shared

Current tau=0.025; alternate tau=0.0125.

| seed | current MATH | alternate MATH | delta | current GSM8K | alternate GSM8K | delta | alternate rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 48.1682% | 47.6877% | -0.4805 pp | 84.4579% | 84.4579% | +0.0000 pp | 0.01249816 |
| 43 | 48.2683% | 48.8488% | +0.5806 pp | 84.0030% | 83.6240% | -0.3791 pp | 0.01237538 |
| 44 | 48.7287% | 48.7888% | +0.0601 pp | 84.3821% | 84.3063% | -0.0758 pp | 0.01243580 |
| 45 | 47.9079% | 48.3684% | +0.4605 pp | 84.1547% | 84.6854% | +0.5307 pp | 0.01248982 |
| 46 | 47.9479% | 48.5285% | +0.5806 pp | 84.4579% | 84.3821% | -0.0758 pp | 0.01245436 |

- alternate-current task mean: +0.1201 +/- 0.2762 pp; 95% CI [-0.2228, +0.4630]; positive seeds 3/5
- current-unconstrained task mean: +0.0482 +/- 0.1560 pp; 95% CI [-0.1455, +0.2419]
- alternate-unconstrained task mean: +0.1683 +/- 0.1699 pp; 95% CI [-0.0426, +0.3793]
- alternate-hidden task mean: -0.0896 +/- 0.1661 pp; 95% CI [-0.2958, +0.1166]

| dose | two-task mean accuracy | std across seeds |
| --- | ---: | ---: |
| hidden | 66.4574% | 0.1769 |
| unconstrained | 66.1995% | 0.0673 |
| tau=0.025 | 66.2477% | 0.2002 |
| tau=0.0125 | 66.3678% | 0.2059 |

## qwen3_4b / output

Current tau=0.00625; alternate tau=0.003125.

| seed | current MATH | alternate MATH | delta | current GSM8K | alternate GSM8K | delta | alternate rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 48.3483% | 48.2082% | -0.1401 pp | 84.4579% | 84.2305% | -0.2274 pp | 0.00312701 |
| 43 | 48.4885% | 48.9489% | +0.4605 pp | 84.0030% | 84.1547% | +0.1516 pp | 0.00312487 |
| 44 | 48.7087% | 47.9880% | -0.7207 pp | 84.5337% | 84.7612% | +0.2274 pp | 0.00312536 |
| 45 | 48.2883% | 48.3283% | +0.0400 pp | 84.0030% | 84.2305% | +0.2274 pp | 0.00312799 |
| 46 | 49.1091% | 48.7487% | -0.3604 pp | 84.3821% | 84.5337% | +0.1516 pp | 0.00312509 |

- alternate-current task mean: -0.0190 +/- 0.2320 pp; 95% CI [-0.3070, +0.2690]; positive seeds 2/5
- current-unconstrained task mean: +0.0824 +/- 0.1922 pp; 95% CI [-0.1562, +0.3210]
- alternate-unconstrained task mean: +0.0634 +/- 0.2147 pp; 95% CI [-0.2032, +0.3300]
- alternate-hidden task mean: -0.0441 +/- 0.1501 pp; 95% CI [-0.2305, +0.1423]

| dose | two-task mean accuracy | std across seeds |
| --- | ---: | ---: |
| hidden | 66.4574% | 0.1769 |
| unconstrained | 66.3499% | 0.1191 |
| tau=0.00625 | 66.4323% | 0.2508 |
| tau=0.003125 | 66.4133% | 0.1790 |

