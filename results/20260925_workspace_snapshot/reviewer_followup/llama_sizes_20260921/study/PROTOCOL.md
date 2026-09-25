# Llama 1B / 3B / 8B strict-budget size extension

Exploratory extension after seeing prior Llama results. Official pretrained text-only Llama-3.2-1B/3B and Llama-3.1-8B. Versions, tying and pretraining differ: not a pure size causal experiment. Same sealed CLUENER and WikiSQL data and prompts as prior Llama study; native BOS/EOS. No changes to task training/scoring: 2048 samples, 64 steps, LR2e-4, H r8 alpha16 dropout.05, E/U r16 alpha128 dropout0, E biased/U unbiased. Three paired seeds6100..6102 and7100..7102. Greedy FP32 final-checkpoint evaluation, BF16 training, original base frozen. Keep all outcomes.

Exact active-budget rule: minimize number of added MLP down ranks needed to spend 65*d exactly, then maximize Q increments, evenly spaced within each module type; shared r8 initialization and scaling2 preserved. All increments are r8->r9. 1B and 3B require some MLP ranks; 8B q/k allocation matches old exactly. This changes the 3B comparator placement as well as removing 1024 surplus parameters; do not label it a pure parameter-count effect.

Full matrix: 3 models x2 tasks x3 arms x3 seeds =54 fits plus6 Base evaluations. Reuse30 verified historical fits and4 Base evaluations: all 8B main arms, 3B H/HEU and Base. New24 fits: 1B18 and strict-budget3B6; plus2 new1B Base evaluations. Technical smokes precede all new formal fits: 1B allthree arms x2 tasks, 3B strict-budget x2 tasks =8. No prior adapter initializes a new fit.

Verify reused model, source, data and results against original hashes, compare tokenizations, re-run independent output audits. Reused trees are read-only symlink references; no writes to originals. Final audit checks full matrix, actual shared initial tensors and sample order, frozen base, active extras, parameter counts, save/reload and official scoring.

Report paired raw differences and three-seed intervals; fixed family12 comparisons. Old three-seed/five-seed tuned Llama reports remain separate. Not new independent evidence for reused fits; no claim that all seeds, tasks or sizes must improve. No tuning to test results.

Resources: at most4 workers, one per GPU, memory admission48GiB for8B and36GiB for smaller sizes. Stop new admissions below60GiB disk; keep failures and do not automatically retry/overwrite. No other workloads terminated.
