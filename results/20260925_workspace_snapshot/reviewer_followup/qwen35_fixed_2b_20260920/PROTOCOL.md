# Qwen3.5-2B-Base: corrected four-size numerical-policy study

Four sizes 0.8B/2B/4B/9B are rerun with one common numerical policy after a
controlled zero-update intervention exactly reproduced an old first-loss
discrepancy by changing an admissible FLA L2-normalization kernel configuration.
The old job did not log its chosen configuration; the intervention establishes
a sufficient mechanism, not the historical kernel choice or the cause of task
accuracy differences. No partial old outcome is spliced into this study.
The prior 4B result and all incomplete old-policy fits are retained.

fixed_runtime.py pins 12 official FLA autotuners to architecture/data/score-
independent admissible configurations in each process. No installed packages
are modified. CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before startup; deterministic
PyTorch algorithms and cuDNN are enabled, TF32 and reduced-precision GEMM
reductions disabled. This policy changes multiple numerical settings, so any
effect difference from old results cannot be attributed solely to L2 norm.
NUMERICAL_POLICY_SPEC.json records the controlled experiment and eight-step
cross-GPU pilots. All formal choices remain fixed before corrected task fits.

Official pretrained-only Qwen/Qwen3.5-2B-Base is downloaded from ModelScope.
MODEL_IDENTITY_AUDIT.json pins the official HF revision and independently verified
git/LFS identities. Use native text decoder, tokenizer and actual tied/untied
status; no vision adaptation. Inspect every loaded base tensor against shards.
All original FP32 recurrent decay/norm weights are retained; other matrices BF16.
The isolated Torch 2.6/Transformers 5.6.2 runtime is unchanged. Native FLA and
causal convolution training, native PyTorch FP32 evaluation, unfused gated norm,
eager full attention, TF32 off, non-reentrant RNG-preserving checkpointing.

Verified copies of sealed 4B data/scorers and archived multiscale source are documented in SOURCE_REUSE.
WikiSQL train/dev/confirm=2048/1024/2048, TREC50=2048/256/500. Retokenize all inputs,
verify groups, labels and reference scoring. Same prompts and no truncation.
These public sets are previously evaluated project benchmarks, not globally
unseen data and not evidence against pretraining exposure.
WikiSQL primary=official execution accuracy, greedy generation cap 256/native EOS.
TREC50 primary=joint two-token code likelihood among 50 candidates, no EOS,
no length normalization, five complete native prefixes, no scoring cache.
Native 50-candidate checks on shortest/longest examples of every evaluation,
tolerance 2e-4, identical argmax. Save and audit all responses/scores.

H: rank 8, alpha 16, dropout .05 on all dense decoder projections. HEU: H plus
rank-16 alpha-128 dropout-0 independent E/U maps, E biased/U unbiased. Budget H
adds exactly 65*d active trainable parameters by the rule in architecture.py:
at most one added rank per query/key/output projection, exact sum, maximize query
count then key count, minimize output count; tie by semantic group then ascending
per-rank cost; evenly spaced layers within each semantic/cost group. Preserve
shared rank-8 initialization and alpha/r=2. Verify every extra rank updates.
This is a configured-method comparison, not pure placement causality.

Every fit: 2048 examples once, 64 optimizer steps, effective batch 32/microbatch 2,
AdamW betas(.9,.999), eps1e-8, weight decay0, 2 warmup steps then cosine,
joint global clip1. True token-mean loss including EOS, masked prompts. Final
checkpoint only. Save initialization/order, gradients and update norms. Assert
frozen base, optimizer whitelist, finite FP32 adapters/moments and exact reload.

Same LR candidate and tie order in every arm/task/size:
2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4. Boundary LR equals H LR.
Development seeds 7600/7601: 72 fits per size. Freeze choices only after all dev
audits pass. Then confirmation seeds 7700..7704: 30 fits, plus 2 frozen Base
evaluations. Ten technical smokes cover all arms and low/high boundary LRs.
114 jobs per size, 456 corrected-policy jobs total. Equal tuning counts do not imply equal FLOPs.

Sixteen primary contrasts across the FOUR corrected sizes: HEU-H and HEU-budget H on
each of two tasks. Report all seed differences, mean, sample SD, marginal 95%
and family-16 Bonferroni t intervals with df4. 20,000 paired cluster bootstraps
(WikiSQL table/TREC question; random seeds20300920/21), conditional on the fixed
five fitted models, with marginal and family-16 percentiles. This does not
combine seed/data/tuning uncertainty. The earlier 4B study remains a separate historical result, not an independent replication.
Small samples, fixed data/configuration and t assumptions limit inference.
No score-based retries, selective reporting or universal positivity claims.

Before freezing: fresh formula, actual checkpoint, hybrid-kernel and memory probes. Generate first-batch references for all eight smoke/search/confirmation seeds on training data only, no updates. A separate process verifies all three arms for both tasks at seed7600 against these references. Every admitted non-Base job must exactly match its 16 reference micro-losses before any optimizer update and pass exact local gradient repetition. Numerical failures halt admission and drain active workers for diagnosis. This does not prove full-trajectory cross-device determinism.
One global scheduler admits at most one study worker per eligible GPU. Conservative free-memory thresholds are 32/40/56/68 GiB for 0.8B/2B/4B/9B, respectively. All numerical gates apply on every admitted GPU. Never terminate other workloads. Failed artifacts retained;
technical changes require a recorded fresh freeze before any affected fit.
After completion: independent output and paired-initialization audits, report,
seal all artifacts and independently verify hashes and recomputed contrasts.

Additional numerical gate: every non-Base worker repeats its full first batch
forward/backward twice without updates and with restored RNG. Loss values, every microbatch contribution and all gradients must match exactly; actual step-one loss must equal the independent reference before its optimizer update. Save all measurements; failures stop admission. These extra zero-update
passes can warm kernel caches but do not add optimizer steps or consume RNG.
This differs from the historical 4B startup; no cross-study bitwise claim.
