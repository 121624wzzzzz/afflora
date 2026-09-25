# Strict tied-transpose AffLoRA without energy: Qwen3-0.6B

This is the decisive follow-up to the tied-plus-energy result.  It changes
exactly one training variable: `affine_energy_lambda` is left at its default
zero and no `--affine-energy-*` argument is supplied.  Everything else is
paired with `tied_energy_qwen3`: corrected SFT, Qwen3-0.6B base, seeds
42/43/44, tied input/output affine map, affine rank 16/alpha 128, input bias
and the required tied output bias, hidden LoRA rank 8/alpha 16, one epoch,
and effective batch 16.

The existing historical seed-42 no-energy checkpoint is re-evaluated under
the current shard evaluator as a reproducibility check.  The new three-seed
checkpoints will then receive full held-out CE and matched-512 IFEval using
eight non-overlapping GPU shards.

This design distinguishes an energy-induced optimization failure from a
failure intrinsic to the tied map.  It does not alter the tied-map bias
semantics: under exact tying the output-side shared bias is necessarily a
token-independent logit shift and therefore cancels in softmax.
