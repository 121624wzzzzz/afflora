# IFEval: corrected generic SFT checkpoints

## Scope

This evaluates the twelve formal corrected-SFT checkpoints (three paired seeds
per model and method) with the official IFEval checker over all 541 prompts.
Generation uses each model's native chat template, greedy decoding, and a
maximum of 512 new tokens.  The score is prompt-level `follow_all_instructions`
(a prompt passes only if all of its constraints pass).  `strict` and `loose`
are the two official IFEval matching modes.

The original planned FollowEval run was not substituted with an invented local
version: its bilingual prompt/checker package could not be obtained publicly.
IFEval is consequently a fully automatic, reproducible proxy rather than a
Chinese FollowEval result.

## Results

`delta = bilateral input+lm_head AffLoRA - hidden LoRA`, paired by seed.

| Model | Mode | Hidden LoRA mean | AffLoRA mean | Paired deltas (42, 43, 44) | Mean delta | 95% paired-t CI |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | strict | 0.217498 | 0.198398 | -0.005545, -0.012939, -0.038817 | -0.019100 | [-0.062500, +0.024299] |
| Qwen3-0.6B | loose | 0.237831 | 0.219347 | -0.007394, -0.009242, -0.038817 | -0.018484 | [-0.062287, +0.025318] |
| Qwen2.5-1.5B | strict | 0.228589 | 0.236599 | +0.005545, +0.012939, +0.005545 | +0.008010 | [-0.002594, +0.018614] |
| Qwen2.5-1.5B | loose | 0.253235 | 0.257548 | -0.005545, +0.011091, +0.007394 | +0.004313 | [-0.017387, +0.026013] |

## Reading the result

For Qwen3-0.6B, the held-out generic-SFT cross-entropy improvement does not
transfer to IFEval: AffLoRA is lower in every paired seed under both official
scoring modes.  Qwen2.5-1.5B has a consistent but small strict-score increase;
with only three seeds the paired confidence interval still includes zero.
Thus these measurements do not support a model-agnostic instruction-following
benefit.

## Artifacts

- `responses/`: one JSONL file per checkpoint (541 completions each).
- `scores/<checkpoint>/eval_results_{strict,loose}.jsonl`: output of the
  official IFEval scorer.
- `../evaluate_ifeval.py`: generation wrapper used for the experiment.
