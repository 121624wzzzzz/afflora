# Qwen3.5-2B-Base: prespecified multiscale extension

This is one of three new sizes (0.8B, 2B, 9B) requested after the sealed 4B
results were known. All three sizes are retained, irrespective of outcomes.
The 4B historical study is not a new independent replication and is not pooled
into the new inferential family. No trained weights/predictions are reused.

Official pretrained-only Qwen/Qwen3.5-2B-Base is downloaded from ModelScope.
MODEL_IDENTITY_AUDIT.json pins the official HF revision and independently verified
git/LFS identities. Use native text decoder, tokenizer and actual tied/untied
status; no vision adaptation. Inspect every loaded base tensor against shards.
All original FP32 recurrent decay/norm weights are retained; other matrices BF16.
The isolated Torch 2.6/Transformers 5.6.2 runtime is unchanged. Native FLA and
causal convolution training, native PyTorch FP32 evaluation, unfused gated norm,
eager full attention, TF32 off, non-reentrant RNG-preserving checkpointing.

Verified copies of the sealed 4B source/data/scorers are documented in SOURCE_REUSE.
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
114 jobs per size, 342 new jobs total. Equal tuning counts do not imply equal FLOPs.

Twelve primary contrasts across the THREE new sizes: HEU-H and HEU-budget H on
each of two tasks. Report all seed differences, mean, sample SD, marginal 95%
and family-12 Bonferroni t intervals with df4. 20,000 paired cluster bootstraps
(WikiSQL table/TREC question; random seeds20300920/21), conditional on the fixed
five fitted models, with marginal and family-12 percentiles. This does not
combine seed/data/tuning uncertainty. Four-B results remain historical family4.
Small samples, fixed data/configuration and t assumptions limit inference.
No score-based retries, selective reporting or universal positivity claims.

Before freezing: formula, actual checkpoint, hybrid kernel, memory and repeated
initial forward/backward technical probes. Known 4B BF16 first-loss variation
must be reported; passing a local repeat does not prove bitwise reproducibility
across kernels/devices. Numerical failures halt admission for diagnosis.
One global scheduler admits at most one study worker per eligible GPU with at
least 68GiB free. Never terminate other workloads. Failed artifacts retained;
technical changes require a recorded fresh freeze before any affected fit.
After completion: independent output and paired-initialization audits, report,
seal all artifacts and independently verify hashes and recomputed contrasts.

Additional numerical gate: every non-Base worker repeats its full first batch
forward/backward twice without updates and with restored RNG. Loss difference
<2e-4, gradient relative RMS<1e-4, then actual step-one loss must agree within
2e-4. Save all measurements; failures stop admission. These extra zero-update
passes can warm kernel caches but do not add optimizer steps or consume RNG.
This differs from the historical 4B startup; no cross-study bitwise claim.
