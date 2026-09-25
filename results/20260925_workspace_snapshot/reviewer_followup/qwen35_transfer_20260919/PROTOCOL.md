# Qwen3.5-4B Base: hybrid-architecture stacking replication

Question: does adding boundary affine LoRA E+U improve hidden LoRA H under exact
parameter budgets and equal development search? This extension was requested
after earlier Qwen, Llama and Gemma results were known. Preserve all outcomes.
No Gemma training or trained-adapter reuse. This protocol is frozen before formal
fits; synthetic timing and native-implementation checks may precede the freeze.

## Identity and native execution

Official pretrained Qwen/Qwen3.5-4B-Base, HF revision
1001bb4d826a52d1f399e183466143f4da7b741b. Download runtime files through the official
Qwen ModelScope repository and check all bytes against HF git/LFS identities.
The repository contains a vision encoder; run its native Qwen3_5ForCausalLM text
component, verify every loaded text tensor against the downloaded checkpoint and
compare to the native multimodal wrapper's text-only path. No vision adaptation.
Native tokenizer: no BOS; EOS and PAD 248044, pads masked. No chat template.
Preserve the 24 GatedDeltaNet and 8 full gated-attention layers, RoPE, gates, norms,
convolution and original tied embedding/head. No embedding scaling or logit cap.
Use the existing isolated qwen35_t26 runtime (Torch2.6/CUDA12.4, FLA0.4.2,
causal-conv1d1.5.0.post8), without modifying the prior shared environment. Native
FLA/causal-conv training; native PyTorch recurrence and convolution for FP32
evaluation; native unfused gated RMSNorm in both. Eager full attention and
non-reentrant gradient checkpointing preserving RNG. Check masked causal target
loss, finite gradients, cached/full consistency and native generation. FP32
adapters and Adam state; official mixed frozen base (BF16 matrices and48 original FP32 decay/gated-norm
tensors preserved exactly), cast to FP32 for evaluation
in all arms including Base. Disable TF32. No kernel-policy or shared
environment upgrades after freezing. Fused training is checked against the native
PyTorch recurrence on a synthetic GPU workload before admission. Verify synthetic nonzero E/U equivalence
against independently folded effective vocabulary matrices; effective E/U need
not remain tied even though original frozen weights stay physically tied.

## Verified data reuse and metrics

Reuse the sealed WikiSQL train/development/confirmation inputs (2048/1024/2048,
1938/854/1628 tables) and TREC50 (2048/256/500 questions), all source bytes checked
against the sealed Gemma study's manifest, which records the preceding Llama/Qwen
origins. Retain prompts and labels; re-encode without truncation. Check disjoint
table/text groups, every gold label and SQL execution, mutation scoring and token
concatenation. These shared public benchmarks were evaluated in prior families:
they are not globally unseen project holdouts or evidence of no pretraining exposure.

WikiSQL primary: official execution accuracy; greedy generation, cap256, native
EOS. Logical-form equality, query/strict-JSON validity, stop/cap rates and alternate
parameter binding are secondary. Independently execute every valid prediction.
TREC50 primary: argmax total two-token code log probability over 00..49, without
EOS or length normalization. Use five complete native prefix forwards, no cached
scoring. Check all 50 candidate scores against independent native causal-LM loss
on the shortest/longest evaluated examples, absolute tolerance2e-4 and identical
argmax. Save all example scores. Macro F1 over all50 and coarse accuracy secondary.
Do not pool the two task metrics.

## Three arms and strict parameter control

H targets every dense linear projection in the text decoder: q/k/v/o and
up/down/gate, plus linear-attention in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and
out_proj. Rank8, alpha16, dropout.05. Exclude lm_head, vision, convolution, norms
and recurrent scalar parameters. E maps native embeddings; U maps final normalized
hidden states before the LM head. Each rank16, alpha128, dropout0; E has bias,
U has none. Extra boundary parameters =65*2560=166400.

Budget H adds exactly166400 real parameters. Enumerate at most one added rank in
query-like (q_proj or in_proj_qkv), key(k_proj), output(o_proj or out_proj) modules.
Minimize nonnegative excess, maximize query-like ranks, then key ranks; evenly
space selected modules by layer order within each group. The resulting allocation
is13 query-like and4 output ranks. Preserve common rank8 initial tensors and
alpha/r=2 exactly; assert all extra ranks update. Architecture-derived module
shapes and totals are frozen in BUDGET_ALLOCATION.json and BUDGET_PLAN.json.
This comparison tests the configured methods; differing boundary alpha/dropout,
bias and initialization prevent attributing all differences solely to placement.

Each fit trains2048 examples once:64 optimizer updates, effective batch32,
microbatch2; same paired seed order/common hidden initialization. AdamW(.9,.999),
eps1e-8, weight_decay0,2-step warmup then cosine decay, joint global clip1.
True target-token mean across microbatches including EOS; prompts masked. Final
checkpoint only. CPU diagnostics record gradient/update norms. Verify frozen
weight digests, exact optimizer whitelist, no base gradients, FP32 moments,
finite values and saved adapter destructive-zero/reload equality.

## Equal search and confirmation

All tasks/arms search the same LR grid and tie order:
2e-4,1e-4,4e-4,5e-5,3e-4,8e-4. Boundary LR equals hidden LR.
Paired development seeds7600/7601:72 fits. Greatest two-seed mean primary score
per task/arm wins; exact ties use candidate order. Freeze SELECTION.json only
after all72 fits and audits pass. Then five fresh seeds7700..7704:30 confirmation
fits. Frozen Base per task(seed7598) enters at the same gate. Ten short technical
smokes(seed7599;64examples/2updates) cover all arms and low/high bilateral LRs.
Finite queue114 jobs. Equal search fits do not imply equal FLOPs or runtime.

Four primary paired contrasts: bilateral minus H, bilateral minus budget H, each
task separately. All five seed differences, sample SD, marginal95% and Bonferroni
family4 paired t intervals(df4). Small-sample distributional assumptions and
conditioning on fixed data/selected configuration apply. Supplement:20,000 paired
cluster bootstrap draws, WikiSQL tables/TREC questions, seeds20300920/20300921,
correctness averaged over fixed five models; ratio of summed differences to
sampled example count. Report marginal and family4 percentile intervals(.625%,
99.375%). This is conditional data uncertainty, not joint seed/tuning uncertainty.
Norms/clipping, validity partitions and recovered/regressed cases are descriptive,
not causal explanations. Retain negative and uncertain findings.

## Admission and closeout

One worker per eligible GPU with at least68GiB free; leave other processes intact.
Before formal training, all technical smokes and shared-initialization checks
must pass. Failure stops admission and drains active jobs. Preserve failed
artifacts; technical fixes require a documented new gate, never retries by score.
After completion, independently audit saved outputs, code/data/model/source
hashes, actual shared tensors and orders, then report and seal all files and
perform a separate full verification. No wall-clock speed comparisons.
