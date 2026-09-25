# Qwen matched constraint-control results

Updated: 2026-07-15T08:11:23+08:00

All contrasts use the same model, seed, topology, rank, hidden LoRA, and A-LoRA LR. 
The only intervention is the existing energy constraint versus the newly completed unconstrained control.

## qwen25_3b / input

Constraint: tau=0.0125, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.

| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.0821% | 41.7818% | -0.3003 pp | 78.2411% | 79.1509% | +0.9098 pp | 0.05409898 |
| 43 | 41.4414% | 41.5215% | +0.0801 pp | 77.9378% | 77.6346% | -0.3033 pp | 0.05158953 |
| 44 | 41.4615% | 41.8018% | +0.3403 pp | 79.0751% | 78.2411% | -0.8340 pp | 0.05324654 |
| 45 | 42.3423% | 41.8018% | -0.5405 pp | 76.6490% | 77.6346% | +0.9856 pp | 0.05571458 |
| 46 | 41.3614% | 41.6216% | +0.2603 pp | 78.1653% | 77.5588% | -0.6065 pp | 0.05330897 |

- constrained-control MATH: -0.0320 +/- 0.3766 pp; 95% CI [-0.4996, +0.4355]
- constrained-control GSM8K: +0.0303 +/- 0.8588 pp; 95% CI [-1.0360, +1.0966]
- constrained-control task mean: -0.0009 +/- 0.2479 pp; 95% CI [-0.3086, +0.3069]; positive seeds 2/5
- constrained-hidden task mean: +0.1050 pp
- control-hidden task mean: +0.1058 pp

## qwen25_3b / shared

Constraint: tau=0.025, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.

| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.1421% | 42.2823% | +0.1401 pp | 77.7862% | 78.1653% | +0.3791 pp | 0.05190600 |
| 43 | 41.5816% | 41.2613% | -0.3203 pp | 78.2411% | 78.9992% | +0.7582 pp | 0.05034286 |
| 44 | 41.4014% | 41.1411% | -0.2603 pp | 78.0136% | 77.8620% | -0.1516 pp | 0.05217709 |
| 45 | 42.1221% | 41.7417% | -0.3804 pp | 77.0281% | 78.6202% | +1.5921 pp | 0.05350662 |
| 46 | 41.8218% | 41.8418% | +0.0200 pp | 77.6346% | 77.5588% | -0.0758 pp | 0.05240327 |

- constrained-control MATH: -0.1602 +/- 0.2274 pp; 95% CI [-0.4425, +0.1222]
- constrained-control GSM8K: +0.5004 +/- 0.7124 pp; 95% CI [-0.3842, +1.3850]
- constrained-control task mean: +0.1701 +/- 0.3086 pp; 95% CI [-0.2131, +0.5533]; positive seeds 3/5
- constrained-hidden task mean: +0.1775 pp
- control-hidden task mean: +0.0074 pp

## qwen25_3b / output

Constraint: tau=0.00625, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.

| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.2222% | 42.4224% | +0.2002 pp | 78.0895% | 78.6202% | +0.5307 pp | 0.00873199 |
| 43 | 41.3013% | 41.3614% | +0.0601 pp | 78.1653% | 78.5444% | +0.3791 pp | 0.00888823 |
| 44 | 41.0811% | 41.2613% | +0.1802 pp | 78.5444% | 78.3927% | -0.1516 pp | 0.00887085 |
| 45 | 41.6016% | 41.5015% | -0.1001 pp | 77.7862% | 77.8620% | +0.0758 pp | 0.00855374 |
| 46 | 41.2212% | 41.4014% | +0.1802 pp | 78.6202% | 78.5444% | -0.0758 pp | 0.00860004 |

- constrained-control MATH: +0.1041 +/- 0.1269 pp; 95% CI [-0.0535, +0.2617]
- constrained-control GSM8K: +0.1516 +/- 0.2936 pp; 95% CI [-0.2130, +0.5162]
- constrained-control task mean: +0.1279 +/- 0.1605 pp; 95% CI [-0.0715, +0.3272]; positive seeds 4/5
- constrained-hidden task mean: +0.2213 pp
- control-hidden task mean: +0.0934 pp

## qwen3_4b / input

Constraint: tau=0.0125, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.

| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 48.5085% | 48.3083% | -0.2002 pp | 84.9128% | 84.4579% | -0.4549 pp | 0.10266560 |
| 43 | 48.4284% | 48.3884% | -0.0400 pp | 84.2305% | 83.8514% | -0.3791 pp | 0.10198561 |
| 44 | 48.4685% | 48.7688% | +0.3003 pp | 84.5337% | 85.2161% | +0.6823 pp | 0.10564104 |
| 45 | 48.1081% | 48.0280% | -0.0801 pp | 84.6096% | 84.3063% | -0.3033 pp | 0.10391561 |
| 46 | 48.6086% | 48.6486% | +0.0400 pp | 83.8514% | 84.3821% | +0.5307 pp | 0.10462185 |

- constrained-control MATH: +0.0040 +/- 0.1869 pp; 95% CI [-0.2281, +0.2361]
- constrained-control GSM8K: +0.0152 +/- 0.5451 pp; 95% CI [-0.6617, +0.6920]
- constrained-control task mean: +0.0096 +/- 0.3572 pp; 95% CI [-0.4339, +0.4531]; positive seeds 2/5
- constrained-hidden task mean: -0.0218 pp
- control-hidden task mean: -0.0313 pp

## qwen3_4b / shared

Constraint: tau=0.025, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.

| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 47.9079% | 48.1682% | +0.2603 pp | 84.3821% | 84.4579% | +0.0758 pp | 0.10358613 |
| 43 | 48.4685% | 48.2683% | -0.2002 pp | 83.8514% | 84.0030% | +0.1516 pp | 0.10127182 |
| 44 | 48.5285% | 48.7287% | +0.2002 pp | 84.0788% | 84.3821% | +0.3033 pp | 0.10397720 |
| 45 | 48.3884% | 47.9079% | -0.4805 pp | 83.9272% | 84.1547% | +0.2274 pp | 0.10565258 |
| 46 | 47.9279% | 47.9479% | +0.0200 pp | 84.5337% | 84.4579% | -0.0758 pp | 0.10502584 |

- constrained-control MATH: -0.0400 +/- 0.3046 pp; 95% CI [-0.4183, +0.3382]
- constrained-control GSM8K: +0.1365 +/- 0.1458 pp; 95% CI [-0.0446, +0.3175]
- constrained-control task mean: +0.0482 +/- 0.1560 pp; 95% CI [-0.1455, +0.2419]; positive seeds 2/5
- constrained-hidden task mean: -0.2097 pp
- control-hidden task mean: -0.2579 pp

## qwen3_4b / output

Constraint: tau=0.00625, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.

| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 48.1882% | 48.3483% | +0.1602 pp | 84.4579% | 84.4579% | +0.0000 pp | 0.00816364 |
| 43 | 48.6286% | 48.4885% | -0.1401 pp | 83.9272% | 84.0030% | +0.0758 pp | 0.00840663 |
| 44 | 48.9489% | 48.7087% | -0.2402 pp | 84.1547% | 84.5337% | +0.3791 pp | 0.00870585 |
| 45 | 48.2683% | 48.2883% | +0.0200 pp | 84.2305% | 84.0030% | -0.2274 pp | 0.00847527 |
| 46 | 48.3884% | 49.1091% | +0.7207 pp | 84.3063% | 84.3821% | +0.0758 pp | 0.00836556 |

- constrained-control MATH: +0.1041 +/- 0.3770 pp; 95% CI [-0.3641, +0.5723]
- constrained-control GSM8K: +0.0607 +/- 0.2171 pp; 95% CI [-0.2089, +0.3302]
- constrained-control task mean: +0.0824 +/- 0.1922 pp; 95% CI [-0.1562, +0.3210]; positive seeds 3/5
- constrained-hidden task mean: -0.0251 pp
- control-hidden task mean: -0.1075 pp

