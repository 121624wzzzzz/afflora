# Fourteen-hour model and architecture experiment window

Authorized start: 2026-09-17 21:07:10 Asia/Shanghai. Deadline: 2026-09-18 11:07:10. WINDOW.json is the machine-readable authority. Reserve the last hour for audit, reporting and archive verification; stop new admission at least 90 minutes before the deadline, and earlier for jobs whose fixed worst-case runtime allowance would extend into the audit hour.

The task is broad, budget-limited exploration. No job, seed count, model or task is selected using observed scores. Priority is fixed by scientific question and resource cost. Every launched run and every omitted or incomplete block is disclosed. Do not replace failures, ties or negative results silently. No outcome-adaptive extension of seeds.

## Frozen intended coverage

Core tasks: exact audited CLUENER and WikiSQL datasets/protocols. Models: Qwen2.5 pretrained Base 0.5B, 1.5B, 3B, 7B; Qwen3 pretrained Base 0.6B, 1.7B, 4B, 8B. New task phase: TREC fine-grained question classification and SQuAD2 extractive reading comprehension with unanswerability, constructed and frozen before their first model output. Full prepared matrices are upper bounds on intended coverage, not promises that every row fits the time budget.

Core fitted arms: H (ordinary internal LoRA), parameter-budget H, H+E, H+U, H+E+U, E-only, U-only, E+U without H. Base is retained. E is the input embedding-side hidden-dimensional affine adapter; U is the output unembedding-side adapter. Independent E/U branches do not assert single-matrix tied merging. The original embedding/head and transformer weights remain frozen in all fitted arms.

Main core breadth uses the first three pre-specified paired task seeds (CLUENER 6100–6102; WikiSQL 7100–7102). Both untied models have a separately pre-specified five-seed extension (6103–6104 / 7103–7104). Report common three-seed results for all complete arms and five-seed untied results only when all required arms have all five. Never compare an arm's five-seed average against another arm's three-seed average. Reused runs are accepted only after identities, training, parameter scope, input IDs, sample order, checkpoint hashes and raw predictions have been checked. No fitted adapter initializes a new run.

New tasks use H / parameter-budget H / H+E+U plus Base, three paired seeds; prioritized models Qwen2.5-1.5B, Qwen2.5-7B and Qwen3-8B, then the remaining five sizes. Their data and prompts will be independently frozen in new_tasks/ before any new-task output. Detailed sample counts, scoring and seed values belong to that phase's protocol.

## Fixed admission priorities

0: numeric/memory smoke tests for each model/task. 10: first three seeds of untied Qwen2.5-7B / Qwen3-8B complete core architecture blocks. 15: Qwen2.5-0.5B core breadth. 20: prioritized new-task model blocks. 22: the two additional untied core seeds. 25: Qwen3-1.7B / 4B core breadth. 30: missing tied Qwen2.5-1.5B / 3B and Qwen3-0.6B architecture arms. 40: remaining new-task sizes. Within a tier, seed/task/model/arm order is fixed in the phase job list. A phase may be registered when its pre-output preparation is complete, without modifying other frozen phase code.

Use GPUs 1–7 only when at least 48 GiB is free at admission, one of our workers per GPU. GPU 0 currently has another large job. Other jobs are never stopped or altered. Shared compute affects time; no wall-clock speed superiority claim. Failures pause further admission in the affected phase until inspected; existing workers finish. Hard deadline termination is explicitly labeled incomplete, never scored as a successful fit.

## Interpretation

This matrix answers coverage and architecture questions under one common recipe. It does not replace equal-budget per-method hyperparameter tuning. Standalone and single-side arms have different capacities; their relative scores do not isolate placement from parameter count, bias or branch scaling. Core tasks were selected after earlier positive results, and reused test sets are not untouched confirmation sets. Model sizes also change depth, width, heads and tied/untied status.

All individual seeds, Base scores, negative results and completion counts are retained. Report mean, seed SD, paired differences and direction counts for complete blocks. Exploratory unadjusted 95% seed-t intervals are explicitly labeled; any corrected intervals use the full frozen phase contrast family, including planned comparisons that time limits may leave unobserved. Do not present exploratory signs as new confirmatory discoveries. Development and test results remain separate.
