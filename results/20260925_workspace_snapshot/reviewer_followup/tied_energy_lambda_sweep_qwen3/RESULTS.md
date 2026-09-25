# Corrected-geometry energy λ sweep (Qwen3-0.6B, seed 42)

This sweep holds the tied-transpose affine adapter, hidden LoRA, and energy
threshold fixed, and varies only the energy penalty weight. The energy
calculation is the corrected input-codebook geometry (`W + s W D^T U^T`), not
the obsolete output-head orientation.

| energy λ | final input rho | independent dev CE | 8-GPU held-out CE | IFEval strict | IFEval loose |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (existing no-energy tied control) | n/a | 1.192491 | 1.211280 | 0.207024 | 0.231054 |
| 1 | 0.029288 | 1.192135 | 1.210732 | 0.218115 | 0.238447 |
| 10 | 0.009158 | 1.192321 | 1.211015 | 0.227357 | 0.242144 |
| 30 | 0.006886 | 1.192167 | 1.210882 | 0.214418 | 0.231054 |
| 100 (existing corrected-geometry run) | 0.006327 | 1.192063 | 1.210562 | 0.205176 | 0.229205 |

All newly produced held-out and IFEval values use the matched 1,000-example
held-out split and official matched-512 IFEval, respectively. Each model's
IFEval generation used all eight GPUs, one shard per GPU. A concurrent
three-model held-out launch initially OOMed in one shard; the missing shard was
detected and rerun alone, so every final held-out number has all 1,000 examples.

## λ=10 three-seed confirmation

Seeds 43 and 44 were trained independently with the same λ=10 setting, then
reloaded for the same full 1,000-example held-out CE and eight-GPU matched-512
IFEval protocol.

| λ=10 seed | independent dev CE | held-out CE | IFEval strict | IFEval loose |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 1.192321 | 1.211015 | 0.227357 | 0.242144 |
| 43 | 1.192490 | 1.210938 | 0.203327 | 0.219963 |
| 44 | 1.191282 | 1.210276 | 0.221811 | 0.253235 |
| mean | 1.192031 | 1.210743 | 0.217498 | 0.238447 |

Against the paired strict-tied no-energy control, the mean held-out CE delta is
-0.000102 (95% paired-t CI [-0.000717, +0.000513]); strict IFEval delta is
+0.003081 (CI [-0.034034, +0.040195]); loose IFEval delta is +0.003080 (CI
[-0.027493, +0.033654]). None is statistically resolved with three seeds.

## Expanded paired λ=0 versus λ=10 test (seven seeds)

To avoid selecting on the seed-42 sweep result, seeds 45--48 were trained for
both λ=0 and λ=10. Together with seeds 42--44 this gives seven strictly paired
runs, all with independent reload, full held-out CE, and matched-512 IFEval.

| Metric | λ=0 mean | λ=10 mean | paired λ=10 - λ=0 (95% CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210837 | 1.210817 | -0.000020 [-0.000245, +0.000205] |
| IFEval strict | 0.212305 | 0.212041 | -0.000264 [-0.009493, +0.008965] |
| IFEval loose | 0.234751 | 0.233166 | -0.001585 [-0.010545, +0.007376] |

Thus the apparent λ=10 improvement does not replicate: its average effect is
effectively zero at this precision.

## Strong constraint λ=100 (five paired seeds)

Two additional λ=100 runs (seeds 45/46) were added to the existing 42--44
runs. The comparison below uses the same-seed no-energy tied controls.

| Metric | λ=0 mean | λ=100 mean | paired λ=100 - λ=0 (95% CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.211048 | 1.210547 | -0.000501 [-0.000768, -0.000234] |
| IFEval strict | 0.214048 | 0.197043 | -0.017005 [-0.029825, -0.004186] |
| IFEval loose | 0.236969 | 0.218115 | -0.018854 [-0.034539, -0.003170] |

The strong energy constraint therefore produces a small but repeatable CE gain
and a statistically resolved instruction-following loss.

## Locating the λ=100 behavior cost

`reviewer_followup/evaluate_ifeval.py` now supports the same diagnostic
`--affine-ablation` modes as the CE evaluator. The ablation edits loaded model
parameters only; it never modifies a checkpoint.

For seed 42, normal λ=100 is 0.205176 strict / 0.229205 loose. Clearing only
the low-rank update gives 0.207024 / 0.223660; clearing only affine bias gives
0.216266 / 0.234750; clearing both gives 0.214418 / 0.231054. The bias is the
dominant direct source of the loss, while the update has a smaller mixed role.

The bias-only ablation was repeated for all five λ=100 seeds:

| Metric | normal λ=100 | zero affine bias | paired ablation delta (95% CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210547 | 1.210527 | -0.000020 [-0.000134, +0.000095] |
| IFEval strict | 0.197043 | 0.208872 | +0.011830 [+0.002423, +0.021237] |
| IFEval loose | 0.218115 | 0.227726 | +0.009612 [-0.000956, +0.020179] |

Removing affine bias recovers most of the strict IFEval loss without a
measurable held-out CE penalty. This identifies the learned input embedding
translation, rather than the low-rank affine update alone, as the main direct
mechanism behind the strong-constraint behavior cost.

## Direct no-bias λ=100 training (three seeds)

For a causal test, strict tied adapters were extended to allow matching
input/output *no-bias* settings (the transpose merge remains exact in this
case). λ=100 was then trained from scratch without affine bias for seeds
42--44, rather than merely zeroing a trained bias after loading.

| Metric | free-bias λ=100 | no-bias λ=100 | paired no-bias - free-bias (95% CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210361 | 1.211476 | +0.001115 [+0.000941, +0.001288] |
| IFEval strict | 0.197166 | 0.210105 | +0.012939 [-0.019203, +0.045081] |
| IFEval loose | 0.215650 | 0.227357 | +0.011707 [-0.023363, +0.046777] |

No-bias training removes most of the behavior deficit but also removes the CE
benefit: versus the no-energy tied baseline its CE is worse by +0.000631 (95%
CI [+0.000078, +0.001183]). Therefore simply disabling bias is not the final
solution; bias helped the training trajectory even though its final value is
the dominant direct behavior cost.

## Interpretation

The old catastrophic result was caused by the obsolete energy geometry /
checkpoint inconsistency. With corrected geometry, λ=10 is neutral within
seven-seed uncertainty, while λ=100 yields a reproducible CE--behavior
trade-off. The direct behavior cost at λ=100 is largely carried by affine bias.

The immediate design implication is to retain the correctly oriented energy
constraint but constrain affine bias separately, rather than either sweeping a
single shared coefficient or deleting bias entirely. The next experiment should
compare free bias with an explicitly norm-limited or lower-learning-rate bias
under λ=100.

## Bias learning-rate sweep

The trainer now supports `--affine-bias-learning-rate-scale`, a multiplier
relative to the affine learning rate. A seed-42 screen at 0.03, 0.1, and 0.3
selected 0.3: it retained held-out CE 1.210876 and gave strict/loose IFEval
0.227357/0.242144. The selected 0.3 setting was then expanded to five paired
seeds.

| Metric | no-energy tied | λ=100, bias LR ×0.3 | paired delta (95% CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.211048 | 1.210789 | -0.000259 [-0.000378, -0.000140] |
| IFEval strict | 0.214048 | 0.211460 | -0.002588 [-0.023785, +0.018610] |
| IFEval loose | 0.236969 | 0.229205 | -0.007764 [-0.028824, +0.013296] |

Reducing bias LR from 1 to 0.3 preserves the CE benefit and removes the
previously significant five-seed IFEval harm, but does not establish a reliable
instruction-following gain: seed 45 remains negative. The remaining candidate
is a separate bias norm constraint, which controls the final bias magnitude
more directly than an optimizer-only multiplier.

## Joint global-energy and bias trust regions

The trainer now also has a separate bias trust region on
`sqrt(V) ||beta|| / ||W||_F`.  A seed-42 screen fixed the global constraint at
`lambda=100, tau=0.00625` and tested bias radii 0.0008, 0.0012, and 0.0016.
The 0.0016 point was selected: held-out CE 1.210044, strict IFEval 0.203327,
and loose IFEval 0.225508.  The other two screen points were respectively
(1.210170, 0.212569, 0.225508) and (1.210333, 0.216266, 0.234750).

Nine corrected-SFT seeds (42--50) were then paired against matched strict-tied
no-energy controls.  Every held-out evaluation used eight shards/GPUs and every
IFEval result used the official 541-prompt matched-512 protocol.

| Metric | no-energy mean | global + bias-cap mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210808 | 1.209972 | -0.000836 [-0.000992, -0.000683] |
| IFEval strict | 0.210926 | 0.208051 | -0.002875 [-0.010064, +0.004518] |
| IFEval loose | 0.233313 | 0.226535 | -0.006778 [-0.014787, +0.001027] |

Thus the CE gain is 9/9 directional and statistically stable.  The earlier
strong-energy behavior loss is no longer statistically established, but neither
strict nor loose IFEval has a stable gain after seed 50; this is a CE improvement
with behavior approximately neutral within current uncertainty, not an
instruction-following improvement claim.

An attempted direct legacy-trainer multiseed run was excluded: it bypassed
`train_corrected_sft.py`, lacked `corrected_data_pipeline` metadata, and had an
incomparable training loss around 1.076.  It is retained on disk only for audit;
all numbers in this section come from corrected-SFT checkpoints.

Adding bias-LR x0.3 to the selected cap was evaluated through four completed
seeds (42--45).  It trades slightly worse held-out CE for higher IFE scores in
some seeds; it is therefore not a clean Pareto improvement.  The remaining
scheduled evaluations were intentionally stopped in favor of the more direct
deployment-time `zero_bias` diagnostic below.  (An earlier draft incorrectly
treated IFE scores as loss-like; higher IFE is better.)

The previously trained remaining four checkpoints (46--49) were subsequently
evaluated under the same eight-GPU protocol, making this an eight-seed result.
Relative to matched no-energy controls, the combination improves held-out CE by
-0.000378 [-0.000505, -0.000234] but lowers loose IFEval by
-0.006470 [-0.011784, -0.000924] (strict: -0.003004
[-0.009473, +0.003004]).  Relative to cap-only, it gives nonsignificant IFE
increases of +0.002773 on both strict and loose while worsening CE by
+0.000454 [+0.000350, +0.000567].  It is consequently a CE/behavior trade-off,
not a Pareto-improving intervention.

### Deployment-time zero-bias check

The seed-50 post-hoc result motivated a complete nine-seed check: retain the
bias while training, then set only the affine bias to zero while loading the
same checkpoint for both held-out and IFEval.  This is not a useful fix.

| Metric | normal cap | zero-bias reload | paired delta (zero - normal; 95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.209972 | 1.209991 | +0.000019 [-0.000041, +0.000072] |
| IFEval strict | 0.208051 | 0.202711 | -0.005340 [-0.009037, -0.001232] |
| IFEval loose | 0.226535 | 0.221606 | -0.004929 [-0.008831, -0.001027] |

Thus zeroing the final bias leaves CE effectively unchanged but *reliably
reduces* IFEval (strict wins 1/9 and loose wins 1/9).  The original seed-50
observation was not representative.  The learned final bias is behavior-helpful
on average; it is not the mechanism responsible for the residual IFE deficit.

### Intermediate global energy: lambda=30 with the selected bias cap

To test whether the behavior-neutral `lambda=10` regime and the CE-effective
`lambda=100` regime have a usable threshold between them, a pre-specified
`lambda=30, tau=0.00625, bias-cap=0.0016` replication was trained and evaluated
on independently matched seeds 43--50.  Each held-out and IFEval run again used
eight GPU shards.

| Metric | no-energy mean | lambda=30 + cap mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210749 | 1.210060 | -0.000689 [-0.000823, -0.000566] |
| IFEval strict | 0.211414 | 0.207948 | -0.003466 [-0.008549, +0.001617] |
| IFEval loose | 0.233595 | 0.224122 | -0.009473 [-0.015018, -0.004159] |

This intermediate point retains a stable CE improvement but has a statistically
resolved loose-IFE loss.  Relative to `lambda=100 + cap`, it also worsens CE by
+0.000097 [+0.000019, +0.000185] without a reliable IFE gain.  It therefore
rules out the simple "intermediate lambda is the Pareto point" hypothesis.

### Low global energy: lambda=10 with the selected bias cap

An independently matched eight-seed (43--50) `lambda=10 + cap` replication
tested whether the cap could make the previously behavior-neutral low-energy
regime useful.  It cannot: CE improves in all eight pairs, but behavior degrades
reliably.

| Metric | no-energy mean | lambda=10 + cap mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210749 | 1.210073 | -0.000676 [-0.000818, -0.000554] |
| IFEval strict | 0.211414 | 0.202634 | -0.008780 [-0.012708, -0.005776] |
| IFEval loose | 0.233595 | 0.222736 | -0.010860 [-0.014094, -0.007856] |

It is worse than `lambda=100 + cap` in CE (+0.000110
[+0.000027, +0.000201]) and has no reliable IFE advantage.  Hence bias-capping
does not repair the CE--IFE trade-off at either low or intermediate energy;
further work should change the constraint form rather than sweep this family.

### Linear-only global energy (bias excluded)

The original global energy norm includes the affine translation in addition to
the low-rank linear update.  Because the preceding runs also added a separate
bias cap, this exposed a double-penalty ambiguity.  The trainer now supports
`--affine-energy-exclude-bias`; its codebook geometry is covered by a direct
regression test against the actual linear update.  An independent seeds 43--50
run used `lambda=100, tau=0.00625` with this flag and no bias cap.

| Metric | no-energy mean | linear-only energy mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210749 | 1.211026 | +0.000276 [+0.000199, +0.000369] |
| IFEval strict | 0.211414 | 0.208872 | -0.002542 [-0.012015, +0.006701] |
| IFEval loose | 0.233595 | 0.227588 | -0.006007 [-0.014325, +0.002311] |

The linear-only constraint removes the robust CE benefit completely (0/8 CE
wins) without giving a reliable IFE gain.  Thus the CE improvement of the full
constraint depends on restricting the translation-containing update, while the
translation restriction is also implicated in the behavior trade-off.  This
rules out the current family of linear-vs-bias energy decompositions as a
reproducible CE--IFE Pareto solution.

### Full global energy with slower learned-bias updates

The remaining plausible explanation was that full energy should still constrain
the translation-containing codebook update, but the affine bias itself may need
a slower optimizer clock.  We therefore completed the previously inconclusive
free-bias experiment with four new corrected-SFT seeds (47--50), giving matched
seeds 42--50: `lambda=100`, `tau=0.00625`, no separate bias cap, and
`affine-bias-learning-rate-scale=0.3`.  All four new held-out and IFEval runs
used the same eight-GPU sharding protocol.

| Metric | no-energy mean | energy + bias-LR x0.3 mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210808 | 1.210627 | -0.000181 [-0.000266, -0.000081] |
| IFEval strict | 0.210926 | 0.209489 | -0.001438 [-0.009653, +0.005956] |
| IFEval loose | 0.233313 | 0.228589 | -0.004724 [-0.013350, +0.002259] |

The CE improvement remains reproducible (8/9 directional pairs), but neither
IFE metric improves reliably (only 2/9 directional wins each and both mean
deltas are negative).  Slowing the bias changes the CE magnitude, but does not
create a defensible CE--IFE Pareto point.  Together with the zero-bias and
linear-only checks, this makes the negative result quite specific: the observed
held-out gain is real, but this unconditional energy regularizer alone is not a
behavior-preserving post-training objective.

### Bias-LR x0.1 falsification screen

The original seed-42 `bias-LR x0.1` screen appeared promising, so it was
pre-registered for four independent corrected-SFT replications (seeds 43--46).
It failed decisively; extending it to the remaining seeds would not be an
efficient use of the experimental budget.  Across the five matched seeds
(42--46), with full `lambda=100, tau=0.00625` energy and no separate bias cap:

| Metric | no-energy mean | energy + bias-LR x0.1 mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.211048 | 1.210835 | -0.000213 [-0.000316, -0.000111] |
| IFEval strict | 0.214048 | 0.205915 | -0.008133 [-0.012569, -0.001848] |
| IFEval loose | 0.236969 | 0.226248 | -0.010721 [-0.018484, -0.002957] |

All five CE pairs improve, but strict IFE improves in only 1/5 pairs and loose
IFE in 0/5.  The apparent seed-42 IFE result was therefore a false positive;
making the bias update still slower is actively harmful and is not expanded.

### Translation-weighted global energy screen

To separate the linear update from affine translation without the all-or-nothing
choice of the linear-only ablation, the trainer now supports a translation
weight inside the global energy norm.  Weight 1 is the exact merged codebook
update; weight 0 is linear-only; the new `weight=0.5` point exactly penalizes
`linear_update + 0.5 * translation` (covered by a direct geometry regression
test).  A four-seed corrected-SFT screen on seeds 43--46 used
`lambda=100, tau=0.00625` and otherwise identical settings.

| Metric | no-energy mean | translation weight 0.5 mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210990 | 1.210965 | -0.000025 [-0.000125, +0.000128] |
| IFEval strict | 0.215804 | 0.213031 | -0.002773 [-0.008318, +0.005083] |
| IFEval loose | 0.238447 | 0.231516 | -0.006932 [-0.010166, -0.002311] |

This interpolation loses the reliable CE benefit (only 3/4 directional CE
pairs) while retaining a resolved loose-IFE loss (0/4 directional wins).  It
is therefore not expanded, and rules out the direct translation-weight family
between the already-tested linear-only and full global norms.

### No-energy ↔ full-energy adapter interpolation

As a separate deployment-side diagnostic, each matched no-energy and full-energy
checkpoint pair was linearly interpolated at the adapter-parameter midpoint
(`w_energy=0.5`), including the affine and hidden-LoRA safetensor states.  Key
and shape equality are checked before construction; the resulting minimal run
directories were evaluated through the same 8-way held-out and 8-way IFEval
pipeline.  This asks whether a Pareto point lies on the direct path between the
two trained solutions, rather than in the energy coefficient itself.

| Metric | no-energy mean | adapter midpoint mean | paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.210990 | 1.210493 | -0.000497 [-0.000626, -0.000367] |
| IFEval strict | 0.215804 | 0.208410 | -0.007394 [-0.011091, -0.003697] |
| IFEval loose | 0.238447 | 0.231516 | -0.006932 [-0.012015, -0.002773] |

The midpoint is a clean CE improvement in all 4/4 pairs, but strict and loose
IFE both decline in all 4/4 pairs.  Thus the direct no-energy/full-energy
parameter path contains no useful Pareto point at its midpoint; it is not
expanded before trying a qualitatively different preservation mechanism.

## Qwen2.5-1.5B strict-tied crosscheck

A fresh corrected-SFT, strict-tied rank-16/alpha-128 three-seed comparison was
run on Qwen2.5-1.5B; it is separate from the older non-tied rank-48 results.
The same global constraint and bias cap improve held-out CE in all three pairs:
delta -0.001254, -0.000783, and -0.000637 for seeds 42--44.  Official IFEval
is mixed: strict deltas are -0.011091, -0.005545, +0.020333 and loose deltas
are -0.001848, -0.001848, +0.014787.  This confirms that the CE benefit is not
unique to Qwen3-0.6B, while behavior transfer remains seed/model dependent.

## Energy scheduling and behavior-preserving follow-ups

Delaying `lambda=100` until step 712 was stopped after two matched seeds: both
held-out CE values worsened (+0.001989 and +0.001701) and loose IFEval fell by
-0.012939 in both pairs. Applying energy only in the first half, then releasing
it, retained CE gains in the first three completed pairs (-0.000724, -0.000691,
-0.000756) but lowered loose IFEval in all three; strict IFE also fell in two.
Neither schedule is a Pareto point.

A short continuation started both arms from the same completed no-energy
checkpoint. The existing no-energy solution has initial global-energy rho about
0.0819, versus the from-scratch trust-region radius 0.00625. Reapplying the
original penalty force-compresses this solution: the first matched pair raised
held-out CE by +0.002696. This branch was stopped before IFEval expansion.

An assistant-token teacher-KL preservation loss was then added, using each
same-seed no-energy checkpoint as a frozen teacher while training fresh
`lambda=100` models. A four-seed held-out screen was strongly positive for
KL=0.1 (CE deltas -0.001121, -0.001459, -0.001187, -0.001557). However its
first official IFEval replication (seed 43) was negative: strict/loose changed
from 0.208872/0.231054 to 0.199630/0.219963. Matching teacher logits on
corrected-SFT assistant tokens therefore does not preserve out-of-domain
constraint-following generation; this IFE branch was stopped rather than
selectively extended.

### 50% non-evaluation UltraChat behavior anchor

To test whether the IFE cost was caused by insufficient generic instruction
coverage, a new matched four-seed screen kept the total corrected-SFT epoch
budget fixed at 22,780 rows, but replaced half of its rows with an independently
sampled UltraChat anchor corpus.  The anchor was first filtered using the exact
Qwen3 native template and `max_seq_len=1024`, dropping 1,943/99,986 rows that
would otherwise have zero supervised assistant tokens after truncation.  Each
seed therefore used 11,390 corrected-SFT rows + 11,390 filtered UltraChat rows,
with shared sampling and ordering between lambda=0 and lambda=100.

| seed | anchor l0 held-out CE | anchor l100 held-out CE | paired l100 - l0 |
| ---: | ---: | ---: | ---: |
| 43 | 1.258204 | 1.258755 | +0.000551 |
| 44 | 1.258057 | 1.258281 | +0.000224 |
| 45 | 1.259012 | 1.259427 | +0.000415 |
| 46 | 1.257702 | 1.257776 | +0.000073 |

The energy arm loses held-out CE in all 4/4 pairs (mean +0.000316).  This is
the opposite of the original corrected-SFT result, so the configuration cannot
be a CE--IFE Pareto candidate; IFEval was deliberately not run.  It does show
that generic multi-turn instruction data materially changes the response to
the energy objective, rather than simply repairing its behavior cost.

### 25% UltraChat anchor dose follow-up

The same preprocessed corpus and matched protocol were repeated with a 25%
anchor dose (17,085 corrected-SFT + 5,695 UltraChat rows, still 22,780 total
rows and one epoch).  This checks whether the 50% mixture merely overwhelmed
the original CE objective.

| seed | anchor l0 held-out CE | anchor l100 held-out CE | paired l100 - l0 |
| ---: | ---: | ---: | ---: |
| 43 | 1.231572 | 1.231765 | +0.000193 |
| 44 | 1.231645 | 1.231571 | -0.000074 |
| 45 | 1.232999 | 1.233213 | +0.000214 |
| 46 | 1.230838 | 1.230964 | +0.000125 |

The mean paired delta is +0.000114, with only 1/4 directional CE wins.  Thus
reducing generic anchor exposure weakens but does not reverse the CE loss; it
is not an IFEval candidate.  The combined 25%/50% dose response rules out an
ordinary, untargeted multi-turn dialogue mixture as a behavior-preserving
solution.  The next data-preservation attempt should use non-evaluation
examples with explicit format/length/count/list constraints, rather than a
generic dialogue anchor.

### 10% explicit-constraint UltraChat anchor: eight-seed replication

The follow-up used the same native-template, `max_seq_len=1024` filtered
UltraChat corpus, but selected only examples whose first user turn contains an
explicit lexical constraint (`exactly`, `at least`, `at most`, `must`, `do not`,
`without`, or `only`). This left 18,550 candidate anchors. Each run kept the
one-epoch training budget exactly fixed at 22,780 rows by using 20,502
corrected-SFT rows plus 2,278 anchors (10%), with the same anchor sample and
row order in its matched lambda=0/lambda=100 pair. A normalized exact-match and
long-string-containment check against all 541 Google IFEval prompts found zero
overlaps, so the result is not explained by prompt leakage.

The first four seeds (43--46) were directionally encouraging on held-out CE
(4/4 wins), so a pre-specified independent replication added seeds 47--50
before inspecting the latter IFE results. All eight held-out evaluations use
the full 1,000 examples in eight GPU shards; all IFE results use official
matched-512 generation, eight GPU shards, and the official strict/loose
scorers.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 43 | -0.000163 | +0.011091 | +0.012939 |
| 44 | -0.000526 | -0.009242 | -0.005545 |
| 45 | -0.000055 | -0.024030 | -0.025878 |
| 46 | -0.000532 | +0.009242 | +0.011091 |
| 47 | +0.000373 | +0.018484 | +0.014787 |
| 48 | -0.000140 | +0.005545 | +0.012939 |
| 49 | +0.000031 | -0.005545 | -0.012939 |
| 50 | -0.000506 | -0.022181 | -0.020333 |

| Metric | lambda=0 mean | lambda=100 mean | paired lambda=100 - lambda=0 (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.218121 | 1.217931 | -0.000190 [-0.000457, +0.000078] |
| IFEval strict | 0.162200 | 0.160120 | -0.002079 [-0.015229, +0.011070] |
| IFEval loose | 0.172828 | 0.171211 | -0.001617 [-0.015524, +0.012289] |

The apparent first-four-seed CE benefit did not replicate (only 2/4 CE wins in
the new seeds; 6/8 overall), and strict/loose IFE each improve in exactly 4/8
pairs. Explicitly constraint-shaped, non-evaluation behavior data therefore
does alter the response relative to generic UltraChat, but simple replacement
mixing does not yield a reproducible CE--IFE Pareto improvement. Future
preservation tests should retain the original corrected-SFT sampling measure
rather than replacing 10% of it with anchor examples.

### Auxiliary explicit-constraint anchor (lambda=0.1): twelve-seed test

Replacement mixing changes the corrected-SFT sampling measure.  To test a
preservation mechanism without dropping any primary examples, the trainer was
extended with an auxiliary supervised-anchor forward pass.  Every normal
corrected-SFT microbatch remains in the task loss; a separate batch of two
examples from the same 2,278-row explicit-constraint UltraChat sample is cycled
deterministically and contributes `0.1 * anchor_CE`.  Thus the two matched arms
have identical corrected-SFT rows, anchor order, initialization, and optimizer
settings; they differ only by global energy lambda (0 versus 100).  The small
auxiliary batch avoids retaining two full batch-8 activation graphs at once.

A one-step energy-enabled smoke run logged a finite auxiliary loss (1.77984),
zero initial energy excess, and a finite gradient norm before the full runs.
The full experiment uses independent seeds 43--54.  The data leakage checks in
the preceding explicit-anchor section still apply: there is no normalized exact
prompt overlap or long containment overlap with any of the 541 IFEval prompts.

| Metric | lambda=0 mean | lambda=100 mean | paired lambda=100 - lambda=0 (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.211876 | 1.211590 | -0.000286 [-0.000420, -0.000153] |
| IFEval strict | 0.156808 | 0.160197 | +0.003389 [-0.002942, +0.009720] |
| IFEval loose | 0.172828 | 0.175139 | +0.002311 [-0.005920, +0.010541] |

Held-out CE improves in all 12/12 pairs.  The behavior point estimates are
positive, but strict and loose each improve in only 6/12 pairs and both paired
intervals cross zero.  This is a materially better outcome than the original
strong-energy IFE deficit and the replacement-anchor test, but it is not yet a
reproducible CE--IFE Pareto claim.  The next pre-specified branch increases the
auxiliary-anchor coefficient while retaining the same full primary SFT measure.

### Stronger auxiliary anchor (lambda=0.2) falsification screen

The first auxiliary result had an unresolved positive IFE point estimate, so a
new independent four-seed screen (55--58) doubled only the auxiliary CE
coefficient from 0.1 to 0.2.  The primary SFT rows, 2,278 deterministic anchor
rows, energy setting, and eight-GPU evaluation protocol were otherwise
unchanged.

| Metric | lambda=0 mean | lambda=100 mean | paired lambda=100 - lambda=0 (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.215113 | 1.214804 | -0.000309 [-0.000414, -0.000204] |
| IFEval strict | 0.169131 | 0.162200 | -0.006932 [-0.019497, +0.005633] |
| IFEval loose | 0.189464 | 0.177449 | -0.012015 [-0.028740, +0.004710] |

CE still improves in all four pairs, but strict IFE improves in only 1/4 and
loose IFE in 0/4.  Increasing plain anchor supervision therefore moves the
behavior outcome in the wrong direction and is ruled out; the next mechanism
must preserve the no-energy model's *anchor distributions*, not merely fit the
anchor targets more strongly.

### Anchor-distribution KL on the auxiliary constraint data: four-seed screen

The failed `lambda=0.2` dose response suggested that merely increasing target
supervision is not behavior preserving.  The next mechanism therefore keeps
the original `lambda=0.1` auxiliary CE anchor, but adds a frozen same-seed,
same-anchor `lambda=0` teacher and applies `0.05 * KL(student || teacher)` on
the *auxiliary explicit-constraint batch* (not on corrected-SFT).  This makes
the preservation signal explicitly local to constraint-following examples.
The four pairs use seeds 43--46; their hidden-LoRA initialization hashes match
their pre-existing `lambda=0` controls exactly.  All CE evaluations use the
full 1,000-example held-out set in eight shards, and IFEval uses all 541
prompts, matched-512 decoding, eight generation shards, and the official
strict/loose scorers.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 43 | -0.000136 | +0.012939 | +0.005545 |
| 44 | -0.000361 | +0.011091 | +0.005545 |
| 45 | -0.000504 | +0.014787 | +0.007394 |
| 46 | -0.000810 | -0.007394 | -0.009242 |

| Metric | lambda=0 mean | energy + anchor-KL mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.212036 | 1.211584 | -0.000453 [-0.000902, -0.000004] |
| IFEval strict | 0.151109 | 0.158965 | +0.007856 [-0.008498, +0.024210] |
| IFEval loose | 0.167745 | 0.170055 | +0.002311 [-0.010023, +0.014644] |

CE improves in 4/4 pairs and strict/loose IFE improve in the same 3/4 pairs.
This is the first positive CE--IFE direction found in the search, and it
reverses the `lambda=0.2` result.  However, the planned estimate is still only
four seeds and both IFE intervals cross zero, driven by seed 46.  It is a
candidate rather than a reproducible Pareto result; an independent four-seed
replication (47--50), using pre-existing matched `lambda=0` controls, is
required before making a claim.

### Independent anchor-KL replication (seeds 47--50)

The independent seeds use the identical frozen-teacher, data, initialization
matching, and eight-GPU evaluation protocol above.  No run from the first four
seeds was reused; only their already-complete matched `lambda=0` controls are
used for the pairwise comparison.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 47 | -0.000221 | +0.007394 | +0.012939 |
| 48 | -0.000480 | +0.011091 | +0.020333 |
| 49 | -0.000558 | -0.005545 | -0.012939 |
| 50 | -0.000377 | +0.001848 | +0.012939 |

| Metric | lambda=0 mean | energy + anchor-KL mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.214828 | 1.214419 | -0.000409 [-0.000641, -0.000177] |
| IFEval strict | 0.157116 | 0.160813 | +0.003697 [-0.007820, +0.015214] |
| IFEval loose | 0.172828 | 0.181146 | +0.008318 [-0.014904, +0.031540] |

The replication preserves the CE effect in all 4/4 pairs and has a positive
IFE point estimate with 3/4 strict and loose wins.  Combining the two fully
evaluated cohorts gives 8/8 CE wins, strict IFE +0.005776 (6/8 wins; 95% CI
[-0.001350, +0.012903]), and loose IFE +0.005314 (6/8 wins; 95% CI
[-0.004120, +0.014748]).  Thus the independent cohort supports the positive
direction, but neither behavioral interval yet excludes zero.  The result is
the strongest candidate so far, not a completed reproducible Pareto claim;
the remaining pre-existing matched controls (51--54) provide a final,
transparent power extension.

### Final power extension and twelve-seed conclusion

The last four pre-existing controls (51--54) were evaluated with the same
energy-plus-anchor-KL arm.  This is a transparent power extension rather than
a newly pre-specified independent cohort: it was launched after the eight-seed
behavior intervals remained inconclusive.  The CE effect nevertheless remains
unambiguous: the extension has 4/4 CE wins (mean delta -0.000512, CI
[-0.000678, -0.000345]) and all twelve paired runs improve CE.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 51 | -0.000533 | -0.009242 | -0.014787 |
| 52 | -0.000547 | +0.016636 | +0.020333 |
| 53 | -0.000604 | -0.011091 | -0.022181 |
| 54 | -0.000362 | -0.009242 | -0.018484 |

| Metric | lambda=0 mean | energy + anchor-KL mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.211876 | 1.211419 | -0.000458 [-0.000572, -0.000343] |
| IFEval strict | 0.156808 | 0.159581 | +0.002773 [-0.004011, +0.009556] |
| IFEval loose | 0.172828 | 0.173444 | +0.000616 [-0.009112, +0.010344] |

CE improves in 12/12 pairs (one-sided exact sign-test p=0.000244), while each
IFE metric improves in only 7/12 pairs (one-sided sign-test p=0.387).  Thus
anchor-distribution KL successfully repairs the *mean direction* relative to
the plain energy and strong-anchor failures, but it does **not** establish a
reproducible IFE Pareto improvement: both behavioral confidence intervals
still include zero and the final cohort contains three negative IFE pairs.  It
is therefore incorrect to claim that this constraint mechanism improves
instruction following, despite its robust held-out CE benefit.

### Stronger anchor-distribution KL (0.1) falsification screen

The twelve-seed `KL=0.05` diagnosis showed that IFE deltas are negatively
correlated with control IFE, consistent with an energy-induced contraction
toward a common behavior.  A fresh matched four-seed screen (55--58) therefore
increased *only* the auxiliary-anchor teacher KL coefficient from 0.05 to 0.1.
It used fresh lambda=0 controls, the same explicit-constraint anchor, the same
energy lambda=100, matched hidden-LoRA initialization, and full eight-GPU CE
and IFEval evaluation.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 55 | -0.000506 | -0.011091 | -0.011091 |
| 56 | -0.000631 | +0.003697 | +0.003697 |
| 57 | -0.000599 | -0.012939 | -0.014787 |
| 58 | -0.000500 | -0.012939 | -0.007394 |

| Metric | lambda=0 mean | energy + anchor-KL=.1 mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.212119 | 1.211560 | -0.000559 [-0.000664, -0.000454] |
| IFEval strict | 0.163586 | 0.155268 | -0.008318 [-0.021139, +0.004503] |
| IFEval loose | 0.176063 | 0.169582 | -0.007394 [-0.020101, +0.005314] |

The stronger KL retains the CE win in 4/4 pairs but makes both IFE metrics
negative in 3/4.  It therefore does not repair the contraction and is rejected.
Together with the failed stronger auxiliary CE anchor, this rules out the
simple strategy of increasing either anchor-preservation coefficient while
keeping global energy at 100.

### Reduced global energy (lambda=50) screen

Because stronger KL harmed IFE, a fresh matched screen reused the 55--58
`lambda=0` controls and reduced only the global energy coefficient from 100 to
50, returning the anchor-distribution KL coefficient to 0.05.  The same full
eight-GPU held-out and IFEval protocols were used.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 55 | -0.000449 | +0.000000 | +0.005545 |
| 56 | -0.000219 | +0.001848 | -0.001848 |
| 57 | -0.000230 | -0.005545 | -0.009242 |
| 58 | -0.000286 | -0.007394 | -0.005545 |

| Metric | lambda=0 mean | energy=50 + anchor-KL=.05 mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.212119 | 1.211823 | -0.000296 [-0.000466, -0.000127] |
| IFEval strict | 0.163586 | 0.160813 | -0.002773 [-0.009774, +0.004229] |
| IFEval loose | 0.176063 | 0.173290 | -0.002773 [-0.012819, +0.007274] |

Halving energy reduces the observed IFE cost relative to the `KL=.1` run but
does not make it positive: strict and loose each improve in only 1/4 pairs.
It retains a robust but smaller CE gain (4/4).  Thus simply lowering global
energy from 100 to 50 is insufficient for a Pareto result.

### Low global energy (lambda=25) falsification screen

The final dose screen again kept anchor-KL at 0.05 and used the same matched
55--58 controls, reducing global energy from 50 to 25.  It completes the
local dose response under otherwise identical training and full eight-GPU
evaluation.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 55 | -0.000090 | +0.000000 | -0.001848 |
| 56 | -0.000179 | -0.007394 | -0.005545 |
| 57 | -0.000013 | -0.009242 | -0.016636 |
| 58 | +0.000042 | -0.003697 | +0.003697 |

| Metric | lambda=0 mean | energy=25 + anchor-KL=.05 mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.212119 | 1.212059 | -0.000060 [-0.000213, +0.000093] |
| IFEval strict | 0.163586 | 0.158503 | -0.005083 [-0.011605, +0.001439] |
| IFEval loose | 0.176063 | 0.170979 | -0.005083 [-0.018748, +0.008581] |

At lambda=25 even the CE benefit is no longer reliable (only 3/4 wins and its
interval crosses zero), while strict IFE never improves and loose improves in
only 1/4.  The 25/50/100 comparison therefore rejects the simple global-energy
dose explanation: reducing its strength sacrifices CE before it produces a
stable behavior benefit.

### Early-only global energy (first 400/1424 steps) screen

The dose screens suggest that the useful CE effect needs strong energy, but a
continuing energy penalty may contract behavior late in SFT.  This fresh
matched 55--58 screen therefore kept the successful configuration unchanged
(energy=100, explicit-constraint anchor CE=.1, anchor-distribution KL=.05),
but applied energy only through step 399 and disabled it from step 400 onward.
Reference KL remained active for the complete run.  The training logs verify
that the final energy-bearing update was step 390; all four runs completed
1424 steps.  Held-out and IFEval again used eight GPUs per seed.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 55 | -0.001367 | -0.009242 | -0.003697 |
| 56 | -0.001158 | +0.007394 | +0.011091 |
| 57 | -0.001280 | -0.009242 | -0.016636 |
| 58 | -0.001203 | +0.009242 | +0.014787 |

| Metric | lambda=0 mean | early-only energy + anchor-KL=.05 mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.212119 | 1.210867 | -0.001252 [-0.001398, -0.001106] |
| IFEval strict | 0.163586 | 0.163124 | -0.000462 [-0.016639, +0.015715] |
| IFEval loose | 0.176063 | 0.177449 | +0.001386 [-0.021570, +0.024343] |

Early-only energy makes the CE effect much larger than continuously applying
energy (4/4 CE wins), so the representation-side benefit is established.  It
does not, however, establish an instruction-following Pareto improvement:
strict IFE is essentially zero on average and loose is slightly positive, but
both metrics split 2/4 by seed and their paired confidence intervals are wide
and cross zero.  This timing intervention removes the consistently negative
mean seen in several continuous-energy screens, but it does not yet produce a
reproducible behavioral gain.

### Late-only global energy (steps 400--1423) timing replication

The complementary timing arm applied exactly the same energy=100, explicit
constraint-anchor CE=.1 and anchor-distribution KL=.05 configuration only from
step 400 onward.  It was run on the fresh 55--58 cohort and independently
replicated on pre-existing matched controls 51--54, so this timing result has
eight pairs.  All training used the same 1424 steps; held-out and IFEval used
the full eight-GPU protocols.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 51 | +0.000970 | -0.005545 | -0.011091 |
| 52 | +0.001228 | +0.016636 | +0.027726 |
| 53 | +0.001153 | +0.000000 | -0.007394 |
| 54 | +0.001242 | -0.009242 | -0.014787 |
| 55 | +0.001201 | -0.003697 | -0.005545 |
| 56 | +0.001056 | +0.011091 | +0.011091 |
| 57 | +0.001260 | +0.003697 | -0.001848 |
| 58 | +0.000977 | -0.001848 | +0.000000 |

| Metric | lambda=0 mean | late-only energy + anchor-KL=.05 mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.212015 | 1.213151 | +0.001136 [+0.001036, +0.001235] |
| IFEval strict | 0.162893 | 0.164279 | +0.001386 [-0.005897, +0.008670] |
| IFEval loose | 0.176987 | 0.176756 | -0.000231 [-0.011719, +0.011257] |

Late-only energy worsens CE in all 8/8 pairs with a narrow positive interval.
Its IFE effects remain mixed and do not compensate for that loss.  Together
with the early-only result, this localizes the reliable CE effect to **early
representation shaping** rather than late-stage regularization; the late stage
is neither a CE nor an IFE Pareto mechanism.

### Reproducible strict-IFE Pareto result: early energy + dual-domain KL

The early-only single-domain-KL arm had a large CE gain but inconclusive IFE.
Category diagnostics showed gains in content/length/case constraints but losses
in format and keywords.  We therefore introduced one narrowly targeted change:
keep the **total** teacher-KL coefficient at .05, but average its assistant-token
KL equally over the corrected-SFT batch and the independent explicit-constraint
anchor batch.  This preserves both the in-domain response mapping and the
constraint anchor rather than increasing the preservation force.  Energy=100
is active only on steps [0,400); all other data, optimizer, tied-transpose
adapter, anchor CE=.1, and evaluation settings remain unchanged.

The first 51--58 screen was followed by an independent, objective-identical
43--50 replication using pre-existing matched lambda=0 controls.  Training
logs in both cohorts verify `source=main+auxiliary_anchor`, with the final
energy-bearing update at step 390.  Every held-out and IFEval evaluation used
the normal eight-GPU shard protocol.

| cohort | pairs | CE delta (95% paired-t CI) | strict IFE delta (95% paired-t CI) | loose IFE delta (95% paired-t CI) |
| --- | ---: | ---: | ---: | ---: |
| independent replication 43--50 | 8 | -0.001384 [-0.001507, -0.001260] | +0.006238 [+0.000519, +0.011957] | +0.004159 [-0.003030, +0.011348] |
| initial screen 51--58 | 8 | -0.001394 [-0.001490, -0.001298] | +0.003928 [-0.000554, +0.008410] | +0.006470 [-0.003545, +0.016484] |
| combined 43--58 | 16 | -0.001389 [-0.001457, -0.001321] | +0.005083 [+0.001856, +0.008310] | +0.005314 [-0.000091, +0.010720] |

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 43 | -0.001455 | +0.007394 | +0.003697 |
| 44 | -0.001129 | +0.016636 | +0.012939 |
| 45 | -0.001401 | +0.001848 | -0.005545 |
| 46 | -0.001550 | +0.003697 | +0.000000 |
| 47 | -0.001326 | +0.011091 | +0.007394 |
| 48 | -0.001414 | +0.011091 | +0.014787 |
| 49 | -0.001555 | -0.005545 | -0.009242 |
| 50 | -0.001239 | +0.003697 | +0.009242 |
| 51 | -0.001331 | +0.003697 | -0.001848 |
| 52 | -0.001429 | +0.012939 | +0.027726 |
| 53 | -0.001501 | -0.005545 | -0.011091 |
| 54 | -0.001197 | +0.003697 | +0.012939 |
| 55 | -0.001483 | +0.007394 | +0.011091 |
| 56 | -0.001502 | +0.005545 | +0.009242 |
| 57 | -0.001435 | +0.000000 | -0.003697 |
| 58 | -0.001273 | +0.003697 | +0.007394 |

This is the first configuration in this series that establishes the requested
Pareto result on the strict IFEval metric: CE improves in 16/16
pairs (one-sided exact sign test p=1.53e-5), and strict IFE improves in 7/8 of
the independent pairs (p=0.0352) and 13/15 non-tied combined pairs (p=0.00369).
The combined strict confidence interval is entirely positive.  Loose IFE also
has a positive mean, but its interval still crosses zero by 0.000091, so it is
supportive rather than a separate reproducible claim.  The correct conclusion
is therefore **robust CE improvement plus reproducible strict instruction-
following improvement**, not an unqualified claim that every IFE variant is
significantly improved.

### Qwen2.5-1.5B-Base cross-model replication (seeds 59--66)

The final Qwen3 configuration was transferred without changing its mechanism:
tied-transpose affine input/LM-head adapter (rank 16, alpha 128), hidden LoRA
(rank 8, alpha 16), explicit-constraint anchor CE=.1, early-only energy=100
over steps [0,400), and total teacher KL=.05 averaged equally over the main
corrected-SFT and auxiliary anchor batches. The matched control retains the
same tied adapter and auxiliary anchor but sets energy and reference KL to
zero. Since the Qwen tokenizer differs, the auxiliary anchor was filtered
against its native chat template before training (18,490 usable rows). All
eight training pairs, held-out CE shards, and IFEval shards completed normally.

| seed | held-out CE delta | IFE strict delta | IFE loose delta |
| ---: | ---: | ---: | ---: |
| 59 | -0.001613 | -0.001848 | +0.003697 |
| 60 | -0.001216 | +0.016636 | +0.014787 |
| 61 | -0.001086 | +0.042514 | +0.038817 |
| 62 | -0.001313 | +0.003697 | +0.018484 |
| 63 | -0.001332 | -0.001848 | +0.003697 |
| 64 | -0.001210 | +0.005545 | +0.003697 |
| 65 | -0.001326 | +0.012939 | +0.005545 |
| 66 | -0.001446 | -0.011091 | -0.009242 |

| Metric | matched control mean | final configuration mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| held-out CE | 1.127345 | 1.126027 | -0.001318 [-0.001452, -0.001184] |
| IFEval strict | 0.191312 | 0.199630 | +0.008318 [-0.005355, +0.021991] |
| IFEval loose | 0.218346 | 0.228281 | +0.009935 [-0.002033, +0.021904] |

This is a clean **cross-model replication of the CE effect**: the candidate
wins in all 8/8 held-out pairs, and its paired interval excludes zero. IFEval
has a positive mean on both strict and loose scoring (strict 5/8 positive;
loose 7/8 positive), but its eight-pair confidence intervals remain wide and
cross zero. Therefore the Qwen2.5 result supports transfer of the early
representation-shaping/optimization benefit; it is directionally consistent,
but not yet an independent statistically conclusive replication of the Qwen3
strict-IFE gain.
