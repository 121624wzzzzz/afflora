# Llama transfer results

Updated: 2026-09-18T12:30:13+08:00

Only independently audited results. Fixed three-seed blocks; missing values remain missing.

| Task | Model | Base | H | Budget H | H+E+U | ΔH | Δbudget |
|---|---|---:|---:|---:|---:|---:|---:|
| cluener | llama32_3b_base | 5.624 | 63.505 | 63.544 | 65.747 | +2.243 | +2.203 |
| cluener | llama31_8b_base | 20.381 | 70.395 | 70.582 | 71.524 | +1.129 | +0.942 |
| wikisql | llama32_3b_base | 0.879 | 76.790 | 76.921 | 77.572 | +0.781 | +0.651 |
| wikisql | llama31_8b_base | 15.820 | 80.859 | 80.924 | 80.632 | -0.228 | -0.293 |

## Fixed planned contrasts

- cluener / llama32_3b_base / HEU − hidden: raw [1.4860937703551826, 1.9375969392024643, 3.3043984115143417]; mean +2.2427; SD 0.9468; unadjusted CI [-0.109209058001416, 4.594601805382742]; Bonferroni-8 CI [-4.639076585835683, 9.124469333217009].
- cluener / llama32_3b_base / HEU − hidden_budget: raw [1.3954477994411576, 1.9015122444832286, 3.312522092625038]; mean +2.2032; SD 0.9935; unadjusted CI [-0.2648239283709479, 4.671145352737231]; Bonferroni-8 CI [-5.018264832836437, 9.42458625720272].
- cluener / llama31_8b_base / HEU − hidden: raw [1.1890703027758747, 0.42398207086506545, 1.774235548191271]; mean +1.1291; SD 0.6771; unadjusted CI [-0.5529675888058625, 2.8111595366940034]; Bonferroni-8 CI [-3.792691860870777, 6.050883808758918].
- cluener / llama31_8b_base / HEU − hidden_budget: raw [1.2207739790813292, 0.1181600644986105, 1.4883068937962776]; mean +0.9424; SD 0.7262; unadjusted CI [-0.8616911575312998, 2.7465184491154444]; Bonferroni-8 CI [-4.336471926885945, 6.22129921847009].
- wikisql / llama32_3b_base / HEU − hidden: raw [0.48828125, 2.34375, -0.48828125]; mean +0.7812; SD 1.4386; unadjusted CI [-2.7923470272600452, 4.354847027260045]; Bonferroni-8 CI [-9.675243300731358, 11.237743300731358].
- wikisql / llama32_3b_base / HEU − hidden_budget: raw [0.1953125, 1.85546875, -0.09765625]; mean +0.6510; SD 1.0533; unadjusted CI [-1.965500857980965, 3.267584191314298]; Bonferroni-8 CI [-7.005070416299238, 8.30715374963257].
- wikisql / llama31_8b_base / HEU − hidden: raw [0.0, -1.7578125, 1.07421875]; mean -0.2279; SD 1.4297; unadjusted CI [-3.7794361494529833, 3.3237069827863164]; Bonferroni-8 CI [-10.619910470127635, 10.164181303460968].
- wikisql / llama31_8b_base / HEU − hidden_budget: raw [-0.48828125, -0.5859375, 0.1953125]; mean -0.2930; SD 0.4257; unadjusted CI [-1.3504009030492063, 0.7644634030492063]; Bonferroni-8 CI [-3.38705914127577, 2.80112164127577].

3B budget H has 1,024 extra trainable parameters (0.00829%); 8B is exactly matched. Positive signs/means alone do not establish corrected significance. All settings and model identities in PROTOCOL.md.
