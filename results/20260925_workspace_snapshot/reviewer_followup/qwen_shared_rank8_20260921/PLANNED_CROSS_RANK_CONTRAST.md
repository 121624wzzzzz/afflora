# Planned cross-rank contrast

Recorded before formal rank8 results are available.

Untied Qwen2.5-7B and Qwen3-8B, CLUENER/WikiSQL, seeds6100..6102 and7100..7102:

H + independent E8/U8 uses hidden_H + (17+16)d = hidden_H +33d parameters.
H + E16 uses hidden_H +33d parameters.
Therefore compare these directly at exactly equal total trainable count, same frozen data, H initialization, LR2e-4 and training duration. This tests splitting a fixed boundary budget across E/U vs concentrating it on E. Do not claim it isolates placement alone: ranks and optimization geometry also change. Report all four model/task conditions and all three seed deltas, not only positives.

U16 has32d extra and is not an exact-budget comparator to E8/U8; report its difference explicitly. No inert parameters added to create artificial matching.
