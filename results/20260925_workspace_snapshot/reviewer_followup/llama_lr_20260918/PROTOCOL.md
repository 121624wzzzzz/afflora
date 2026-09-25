# Llama-3.1-8B WikiSQL learning-rate follow-up

This is a new follow-up motivated by the already observed Llama WikiSQL results.
The earlier negative result remains valid and will not be replaced. This protocol
and executable code are frozen before any new model response is generated.

## Question and fixed settings

Can development-set tuning improve the Llama-8B result, and does H+E+U outperform
equally tuned ordinary H and parameter-matched H? Frozen official Base model;
same 2,048 training examples, one epoch, 64 updates, batch 32, seed-paired order
and common initialization. Internal rank 8 / alpha 16 / dropout .05; E/U rank 16
/ alpha 128 / dropout 0. E has a bias; U does not. AdamW betas (.9,.999), eps 1e-8,
weight decay 0; 2-step warmup and cosine. Joint global norm clipping stays at 1.
No capacity, scaling, clipping, epoch, prompt, decoder or scorer search in this
round. Frozen weights BF16 for training, adapters/optimizer FP32, evaluation FP32.
H has 20,971,520 trainable parameters; budget-H and H+E+U both have 21,237,760.

## Equal search budget, explicitly different search spaces

Each arm receives exactly 6 candidates x 2 new paired seeds (7200, 7201), 12 fits.
H and budget-H each search a single LR in this tie-breaking order:
2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4.
H+E+U searches (hidden LR, boundary/hidden LR ratio) in this order:
(2e-4,1), (2e-4,.25), (1e-4,1), (1e-4,.25), (4e-4,1), (4e-4,.25).
E and U share the boundary LR. Splitting the optimizer does not split clipping.
This equalizes fits and processed training examples, not grid dimensionality;
it does not establish that either search space is exhaustive or globally optimal.
Choose each arm's largest mean development execution accuracy over its two seeds;
exact ties use the above candidate order. Never select seeds or epochs by score.

## Data separation and confirmation gate

Construct 1,024 development examples from the remaining official dev split and
2,048 confirmation examples from the remaining official test split. Exclude all
table IDs, example IDs and normalized prompt duplicates present in prior
`reviewer_followup/**/data/wikisql_*.jsonl` files, including old preflight data.
Use SHA256 ordering with salt `llama-lr-20260918-v1`, independent of model outputs.
Retain original source_split for database execution. Table sets in train/dev/
confirmation are disjoint. Audit these exclusions and source hashes. These are
new to the enumerated project evaluations, not claimed absent from pretraining.
The changed table population means direct old-test/new-test score comparisons
are not estimates of an improvement.

Search workers may evaluate only the new development set. After all 36 fits and
their independent audits pass, write immutable SELECTION.json containing all
development scores, tie resolution, selected configurations and input hashes.
Only then admit confirmation workers: five fresh paired seeds 7300..7304 for
each of three selected configurations plus original LR=2e-4 budget-H and H+E+U
anchors. Identical selected/anchor configurations are fit once and reused by
both labels with explicit provenance. All labels use the same confirmation data.
No additional tuning, retries based on score, or seed expansion is allowed.

## Analysis and uncertainty

Primary metric: official WikiSQL execution accuracy, percentage points. Report
all seed scores and these five prespecified paired contrasts:
1. selected H+E+U minus selected H;
2. selected H+E+U minus selected budget-H;
3. selected H+E+U minus original H+E+U;
4. selected budget-H minus original budget-H.
5. original H+E+U minus original budget-H (replication on the new holdout).
Report mean, paired sample SD and df=4 t intervals, both marginal 95% and
Bonferroni family-5 simultaneous 95%. If two labels map to the same fit, report
identity / zero difference explicitly; do not interpret as independent evidence.
These intervals describe seed variability conditional on the fixed datasets and
selection procedure; they do not capture table sampling or pretraining exposure.
Logical-form accuracy, validity, clipping frequency, H/E/U gradient norms and
actual parameter-update norms are secondary descriptive diagnostics. Gradient
association alone does not identify the causal origin of a generalization gap.
No pooled claims with historical test sets or prior search families.

## Integrity and finite work

All copied scientific source/data verified against the sealed source manifest.
Two original-setting compatibility fits (budget-H and H+E+U, seed 7100, no eval)
compare initialization, training order, losses and final saved tensors against
the sealed original artifacts. Any discrepancy must be resolved or documented
before search admission; require exact adapter equality for compatibility.
Six smoke fits exercise all arms, low boundary LR and extreme hidden LRs.
Each fit checks frozen model digests, gradients, optimizer whitelist, FP32 state,
native tokens, masking, save/reload and exact counts. Each generated response is
independently redecoded, rescored and checked with the official SQL evaluator.
Log pre-clipping H/E/U norms, joint clip factor and actual update norms per step.
No old trained fit or old evaluation is substituted for a new search/confirmation
fit. Finite queue: 6 smokes + 2 compatibility + 36 search + at most 25 confirmation.
Any failed worker/audit stops further admission while existing workers drain.

## Preflight-only implementation revision (v2)

The preserved v1 preflight failed exact final-tensor reproduction for budget-H,
while H+E+U reproduced exactly. Two unmodified-original repetitions on GPUs 1/7
and a CPU-instrumented repetition all reproduced the original budget-H initial
and final tensors and every step's loss/norm exactly. Extra GPU diagnostic
allocations affected the numerical training trajectory; the specific CUDA kernel
mechanism was not identified. V2 moves only diagnostic copies/norm calculations
to CPU. See PREFLIGHT_REVISION.json and the preserved preflight archive.
No search fit or confirmation response preceded this revision. The grid, seeds,
data, optimizer update, clipping and statistical rules remain unchanged. Repeat
all smoke and exact compatibility gates before admitting formal work.
