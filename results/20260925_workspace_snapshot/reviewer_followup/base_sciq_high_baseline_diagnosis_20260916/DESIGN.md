# Diagnose high unadapted Base accuracy on SciQ

Post hoc descriptive diagnosis, not additional method confirmation. No training,
no adapter, no prompt selection and no change to sealed previous results.

Two official Base checkpoints; canonical1,000 test questions with the previously
fixed two ambiguous rows excluded from summaries (998). Reuse the verified
native-completion inference implementation and FP32 scoring. Verify model files
and exactly replay original predictions before interpreting ablations.

Conditions fixed before diagnostic inference:
1. Original question + original options (exact prior prompt).
2. Empty question field + the same options.
3. Question from another test item + the same options, using a fixed random
   derangement seeded20260916. Never alter gold labels or option positions.

Report four-label candidate accuracy, full-vocabulary first-token accuracy and
label validity. Do not generate, parse explanations or retune anything. Above-
chance question-omitted accuracy indicates exploitable signals without the
question, potentially option plausibility, corpus priors or memorization. It
does not establish which mechanism is responsible. A decrease when removing or
shuffling questions shows sensitivity to question information, not proof of
reasoning. This cannot confirm or rule out pretraining benchmark contamination.

Also summarize longest-option and fixed-letter heuristics, question/option length,
full-vocabulary original scores, remaining error reduction and the larger-budget
hidden LoRA reference from the sealed study. These are descriptive checks on the
same already inspected benchmark, not substitutes for untouched evaluation.
