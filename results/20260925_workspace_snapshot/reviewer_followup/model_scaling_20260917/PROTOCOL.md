# Qwen2.5 model-size extension (3B and 7B)

Fixed before any new-model GPU output, 2026-09-17. This is targeted size extension of two previously promising tasks, CLUENER and WikiSQL, selected after seeing earlier1.5B results. It is not a fresh random task sample, not an untouched-dataset confirmation, and does not replace previous negative/uncertain ANLI, Banking77 or E2E results.

## Identity and reuse

Qwen/Qwen2.5-3B revision3aab1f1954e9cc14eb9509a215f9e5ca08227a9b and Qwen/Qwen2.5-7B revisiond149729398750b98c0af14eb82c78cfe92750796. Every local weight shard must match the official pinned HF LFS SHA256. Official pinned configuration/tokenizer files verified against git blobs, copied to independent verified_models directories. No fitted adapter reused for new models. All24 model/config/tokenizer/license/card files verified.3B embeddings are tied;7B untied. This changes more than parameter count, so any size trend is descriptive and not a causal scaling law.

Exact old data/prompts copied only after sealed-manifest checks. CLUENER from posttraining_tasks_grounded_20260916: pilot_train2048, pilot_dev200, test1343(public labeled dev), renamed train/dev/test locally. WikiSQL from downstream_transfer_20260917: train2048/dev256/test1024, original databases and official reference runtime. No resampling, truncation, new examples or label mapping. Re-encode with each official new tokenizer. Target is canonical JSON + native EOS151643, no ChatML template. Normed input overlaps checked; prompt and target IDs independently match the original1.5B records.

The32 existing1.5B Base/three-arm×five-seed test summaries are historical anchors only. Rehash source manifests, specs, checkpoints and responses; re-decode and re-score every37872 anchor output before any new-model run. Original1.5B model files also rehashed. No new-model claim is computed by pooling historical anchor seeds with larger-model seeds.

## Training and controls

New models:3B and7B; arms Base, hidden(r8/alpha16), hidden_budget, hidden_both. Hidden LoRA covers q/k/v/o/up/down/gate, dropout.05, no bias. Boundary A-LoRA rank16/alpha128, input bias only, dropout0; original model frozen. No changes to placement or hyperparameters based on results.

Trainable counts:3B ordinary14966784, stack15099904, budget15099904 (exact);7B ordinary20185088, stack20418048, budget20418560.7B's standard rank increments cannot exactly match the stack because all projection rank costs are multiples of1024, whereas the boundary budget is232960. The control gets512 additional parameters (0.00251% of total stack adapter parameters). It is explicitly a slightly higher-budget LoRA control, not exact equality. Keep architecture and bias definition unchanged. In both sizes expand28 q and8 k ranks, evenly spaced, scale alpha/r=2. Shared r8 initialization checked within seed.

Five paired seeds per task: CLUENER6100..6104, WikiSQL7100..7104, matching previous training example order. LR2e-4,2048 examples,1epoch64 optimizer steps, effective batch32, microbatch2 for CLUENER /4 for WikiSQL (unchanged from each old task). AdamW betas(.9,.999),eps1e-8,WD0, clip1; warmupceil(.03*64)=2, cosine afterward. Loss is true answer-token mean across microbatches including nativeEOS, excluding prompt/padding. Frozen base BF16, trainables and optimizer FP32, BF16 autocast training. Final checkpoint only; no task/size-specific LR search or checkpoint selection.

Four two-step smoke runs (per model: CLUENER stack, WikiSQL budget), seeds6099/7099. They check original-weight freeze, zero-residual identity, independent shifted loss, reload, declared parameter counts and largest training microbatch memory. Memory probe forwards/backwards preserve RNG via fork_rng and perform no update. Smoke evaluation uses first dev examples at the full test batch size (32 for CLUENER,16 for WikiSQL), not output-dependent sample choice. No quality gate. Then60 formal train runs +4Base. Any implementation failure is preserved and investigated before formal continuation; no result-driven retry or exclusion.

## Evaluation

FP32 eval, TF32 disabled, SDPA, greedy generation; same prompt and cap as historical anchors. CLUENER max512 new tokens, dev batch8/test batch32; WikiSQL max256, dev/test batch16. No repair/reranking/retry. Save raw token IDs and EOS/cap flags. NativeEOS151643. Original1.5B CLUENER test batch32 was verified against batch8 in the sealed original study.

CLUENER primary original-span micro-F1; prediction(type,text,occurrence) maps back to exact source offsets, invalid items count as false positives. Supplement text F1, exact sentence correctness, schema and strict JSON. WikiSQL primary official query execution equality, supplementary logical form, validity, strictJSON and distinct-placeholder execution diagnostic. Retain official repeated-column placeholder convention and report its alternative separately. Official reference execution independently checks predicted valid queries; gold-independent parsers unchanged from sealed studies.

## Statistics and limits

Eight new primary test contrasts: stack minus hidden and minus budget for2tasks×2new sizes. Report all five paired differences, mean, sampleSD and two-sided95% paired Student-t CI with Bonferroni8 (df4). A new condition is called confirmatory-positive only if both corrected lower bounds>0. Preserve every positive/zero/negative case. Development metrics descriptive only, no choices based on them. Historical1.5B results are labelled reused anchors and are outside the new eight-comparison family. All cross-size patterns are descriptive; no post-hoc claim of significance for size-by-method interactions. Intervals are conditional on fixed public test examples, common hyperparameters and selected tasks; not global correction across all earlier explorations or proof of optimal tuned performance.

GPUs1..6 only when memory<512MiB; leave0/7 and other active jobs alone. Single GPU per job, no quantization. Before formal conclusions rehash inputs/code/models; audit all132 evaluations/90432 responses(including96 smoke responses), all13838 new token records,68 parameter scopes, all20 paired initialization/order groups, optimizer whitelist, no base gradients, original tensor hashes and checkpoint reload. Preserve sealed previous studies. Final archive is sealed with complete file hashes and separate verification.
