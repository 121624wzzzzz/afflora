# Llama transfer results

Updated: 2026-09-21T15:31:21+08:00

Only independently audited results. Fixed three-seed blocks; missing values remain missing.

| Task | Model | Base | H | Budget H | H+E+U | ΔH | Δbudget |
|---|---|---:|---:|---:|---:|---:|---:|
| cluener | llama32_1b_base | 0.167 | 55.931 | 56.017 | 58.838 | +2.908 | +2.822 |
| cluener | llama32_3b_base | 5.624 | 63.505 | 63.632 | 65.747 | +2.243 | +2.115 |
| cluener | llama31_8b_base | 20.381 | 70.395 | 70.582 | 71.524 | +1.129 | +0.942 |
| wikisql | llama32_1b_base | 0.000 | 57.161 | 57.422 | 59.798 | +2.637 | +2.376 |
| wikisql | llama32_3b_base | 0.879 | 76.790 | 76.823 | 77.572 | +0.781 | +0.749 |
| wikisql | llama31_8b_base | 15.820 | 80.859 | 80.924 | 80.632 | -0.228 | -0.293 |

## Fixed planned contrasts

- cluener / llama32_1b_base / HEU − hidden: raw [2.383952208045791, 2.6876974611846407, 3.6509832927868686]; mean +2.9075; SD 0.6615; unadjusted CI [1.26426999951877, 4.550818641826098]; Bonferroni-12 CI [-2.9906478123788767, 8.805736453723744].
- cluener / llama32_1b_base / HEU − hidden_budget: raw [2.3916539911987584, 2.4911650855207483, 3.582494146161828]; mean +2.8218; SD 0.6607; unadjusted CI [1.1805467482329728, 4.4629954003545835]; Bonferroni-12 CI [-3.0690630268246766, 8.712605175412232].
- cluener / llama32_3b_base / HEU − hidden: raw [1.4860937703551826, 1.9375969392024643, 3.3043984115143417]; mean +2.2427; SD 0.9468; unadjusted CI [-0.109209058001416, 4.594601805382742]; Bonferroni-12 CI [-6.1989800623824145, 10.68437280976374].
- cluener / llama32_3b_base / HEU − hidden_budget: raw [1.2776455554692845, 1.868794497725304, 3.1998167671422806]; mean +2.1154; SD 0.9845; unadjusted CI [-0.33029412242902767, 4.561132002653608]; Bonferroni-12 CI [-6.6629605181967335, 10.893798398421314].
- cluener / llama31_8b_base / HEU − hidden: raw [1.1890703027758747, 0.42398207086506545, 1.774235548191271]; mean +1.1291; SD 0.6771; unadjusted CI [-0.5529675888058625, 2.8111595366940034]; Bonferroni-12 CI [-4.908322091555039, 7.1665140394431806].
- cluener / llama31_8b_base / HEU − hidden_budget: raw [1.2207739790813292, 0.1181600644986105, 1.4883068937962776]; mean +0.9424; SD 0.7262; unadjusted CI [-0.8616911575312998, 2.7465184491154444]; Bonferroni-12 CI [-5.533046124486266, 7.417873416070411].
- wikisql / llama32_1b_base / HEU − hidden: raw [3.02734375, 1.26953125, 3.61328125]; mean +2.6367; SD 1.2197; unadjusted CI [-0.393249030748934, 5.666686530748934]; Bonferroni-12 CI [-8.238721763952473, 13.512159263952473].
- wikisql / llama32_1b_base / HEU − hidden_budget: raw [2.83203125, 1.171875, 3.125]; mean +2.3763; SD 1.0533; unadjusted CI [-0.240240441314298, 4.992844607980965]; Bonferroni-12 CI [-7.0152342755116095, 11.767838442178277].
- wikisql / llama32_3b_base / HEU − hidden: raw [0.48828125, 2.34375, -0.48828125]; mean +0.7812; SD 1.4386; unadjusted CI [-2.7923470272600452, 4.354847027260045]; Bonferroni-12 CI [-12.045434870291814, 13.607934870291814].
- wikisql / llama32_3b_base / HEU − hidden_budget: raw [0.09765625, 2.44140625, -0.29296875]; mean +0.7487; SD 1.4789; unadjusted CI [-2.925049578958678, 4.422445412292011]; Bonferroni-12 CI [-12.437456274741848, 13.93485210807518].
- wikisql / llama31_8b_base / HEU − hidden: raw [0.0, -1.7578125, 1.07421875]; mean -0.2279; SD 1.4297; unadjusted CI [-3.7794361494529833, 3.3237069827863164]; Bonferroni-12 CI [-12.97549363199343, 12.519764465326762].
- wikisql / llama31_8b_base / HEU − hidden_budget: raw [-0.48828125, -0.5859375, 0.1953125]; mean -0.2930; SD 0.4257; unadjusted CI [-1.3504009030492063, 0.7644634030492063]; Bonferroni-12 CI [-4.088402005460162, 3.502464505460163].

All three sizes strictly parameter-matched. 3B old +1024 budget fits are retained in the historical study, not substituted here. Existing H/HEU and 8B budget fits are reused after verification; this is not independent replication. Positive signs/means alone do not establish corrected significance. All settings and model identities in PROTOCOL.md.
