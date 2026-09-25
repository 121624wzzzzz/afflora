# Held-out confirmation of the fixed small-data configurations

This is a separate phase following the documented development pilot. Written
before any reserved-test prediction and before completion of the five-seed
development matrix. It does not change the frozen pilot protocol or training.

Evaluate BOTH tasks and BOTH Base models, all five paired seeds and all five
fitted arms (hidden, input, output, hidden_both, hidden_budget), plus unadapted
Base. Keep every sign and result. ToolACE uses all710 reserved V2 positive,
literal-grounded, API-disjoint rows. CLUENER uses all1,343 original public-dev
rows reserved from training and internal development. Describe the latter as
public-dev held-out evaluation, not an official hidden-test submission.

The training subset, final step64, common LR2e-4 and all seeds are fixed. No
test-based checkpoint, learning-rate, prompt, parser, length or subset selection.
Evaluate the same trained weights: no retraining on development. Keep all
input/model/source/adapter hashes and confirm original-weight/tensor identities
on load. Inputs and frozen V2 scorer are unchanged. Gold is passed only to the
scorer after generation. Failed/capped outputs stay in denominators.

For throughput use FP32 greedy inference batch32, 512new tokens, TF32off, SDPA,
native EOS151643, original plain prompts. Before ANY test generation, compare
batch32 with saved batch8 development responses for the first32 development
examples of all four unadapted and all four seed6100 hidden configurations.
If any raw token sequence differs, globally use batch8 for this test phase;
otherwise use batch32. Record all checks and the choice. This is a numerical
reproducibility check on development, not selection against test outcomes.
No test-time teacher-forced CE is necessary; this phase confirms generated
content, syntax/schema validity, termination and truncation.

Main family remains2tasks x2models x2contrasts=8: hidden_both−hidden and
hidden_both−hidden_budget. Use five-seed paired-t intervals with Bonferroni
correction and separate nominal95% paired item/cluster-bootstrap intervals
conditional on the five fitted models. Report all mean/seed effects. Input and
output versus Base are secondary descriptive comparisons. Do not replace a
null main contrast with a favorable auxiliary metric.

Also retain the two fixed training demonstrations as a separate unadapted-Base
NER prompt-sensitivity diagnostic on the same1,343 held-out rows. It cannot
serve as a matched-prompt adapter control. Formal content tables compare each
adapter only with the same zero-shot input protocol.

Limitations: small training subset, a single common LR rather than separately
tuned methods, two related model families, public data may have appeared during
pretraining, literal-grounded ToolACE subset excludes abstention and inference.
This is neither BFCL nor an end-to-end execution/agent benchmark, and does not
establish general SFT or new-knowledge acquisition. Held-out scores strengthen
the fixed-configuration evidence without removing those limits.
