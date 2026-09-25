# Interpretation of the frozen protocol

Recorded while confirmation seed2004 is running. No computation, selection,
sample exclusion, endpoint, or statistical rule is changed here.

The sentence "Test was not examined during this study's design/tuning" in the
frozen DESIGN.md refers to **model test performance**, not to blind custody of
the raw test data. Preparation necessarily read test questions and labels to
check overlap, duplicate choices and native tokenization. During that preflight,
two test examples with duplicated wrong options were also printed for inspection.
No model test score was computed or consulted until after SELECTION.json was
frozen. Thus this is a test set held out from task training and model selection,
not a claim that its text/labels were never inspected by the researcher.

Before any smoke or formal training,19 train rows overlapping normalized
validation/test questions (12 and7 respectively) and14 ambiguous-gold training
rows were removed. Train has11,646 rows and11,576 unique normalized questions;
within-train duplicates remain. Validation and test each have1,000 unique
normalized questions and no normalized-question overlap with one another.
Primary test excludes only the two preidentified ambiguous-gold questions.
No semantic near-duplicate or pretraining-contamination guarantee is made.

The LR search is the fixed symmetric two-value grid, not an exhaustive search.
All arms selected its upper value,2e-4; no later grid expansion is allowed in
this experiment. Because selection happened to choose the same LR everywhere,
the confirmation also compares arms at the same LR.

"Unrestricted first-token accuracy" checks the full-vocabulary argmax for the
first answer token only. It is not a full autoregressive generation evaluation.
The four cyclic option rotations assess label-position robustness, not all24
possible option orders. Neither analysis establishes general generation gains.

This experiment tests independent bilateral input/output A-LoRA. It does not
test output-only A-LoRA, the shared tied-transpose variant, or superiority over
direct Vocab-LoRA. Parameter matching is exact raw trainable count; it does not
match intrinsic function-space dimension or exhaustively optimize placement.
