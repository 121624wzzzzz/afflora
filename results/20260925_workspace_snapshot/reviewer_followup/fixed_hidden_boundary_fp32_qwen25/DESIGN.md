# Fixed-hidden FP32 output-boundary control (Qwen2.5-1.5B)

## Question

The previous joint-training comparison gave direct output Vocab-LoRA a large,
repeatable held-out CE advantage over output A-LoRA, but essentially no IFEval
advantage.  This control asks whether that CE gap remains after removing
hidden/boundary co-adaptation, hidden-LoRA dropout, gradient clipping, mixed
precision in the boundary residual, and treatment-dependent base-logit paths.

This is a mechanism control, not a new benchmark claim.  The corrected SFT
test split and IFEval have already been inspected in earlier experiments, so
neither is virgin held-out evidence.  No endpoint is selected in this run:
the two boundary configurations and all three seeds are fixed before any
dev/test result is read.

## Paired construction

Each completed corrected-SFT hidden-only checkpoint is used as a deterministic
frozen feature extractor:

| pair | frozen source | boundary seed | A-LoRA endpoint | Vocab-LoRA endpoint |
|---|---|---:|---|---|
| 42 | `formal/qwen25_15b_hidden_sd42` | 42 | rank 50, alpha 800 (scale 16) | rank 1, alpha 32 (scale 32) |
| 43 | `formal/qwen25_15b_hidden_sd43` | 43 | rank 50, alpha 800 (scale 16) | rank 1, alpha 32 (scale 32) |
| 44 | `formal/qwen25_15b_hidden_sd44` | 44 | rank 50, alpha 800 (scale 16) | rank 1, alpha 32 (scale 32) |

For a given seed, the A-LoRA and Vocab-LoRA jobs load tensor-identical hidden
LoRA and see the same shuffled training order.  All hidden parameters are
frozen, and all loaded hidden-LoRA dropout modules are replaced by `Identity`.
There is no input adapter, boundary bias, boundary dropout, energy penalty,
reference KL, anchor loss, or gradient clipping.

The common frozen Qwen2.5-1.5B backbone is loaded with `base_dtype=auto`;
the pinned model config resolves this to BF16 and the completion audit requires
the frozen `lm_head` weight to be BF16.  Both treatments
compute the same frozen native-precision base projection with autocast
disabled and return FP32 logits.  The separately computed boundary residual,
its trainable parameters, and its optimizer state are FP32.  CUDA matmul and
cuDNN TF32 are denied and audited.

## Budget and hypothesis boundary

With hidden size `d = 1536` and vocabulary size `V = 151936`, the raw
trainable boundary-factor counts are nearly equal:

- output A-LoRA rank 50: `2*d*50 = 153600`;
- direct output Vocab-LoRA rank 1: `V+d = 153472`.

The raw difference is 128 parameters (0.0834%).  This is **not** an
intrinsic-degrees-of-freedom match.  Away from rank-degenerate endpoints, the
usual factorization gauge gives at most
`2*d*50 - 50^2 = 151100` intrinsic dimensions for the rank-50 hidden-space
map, versus `V+d-1 = 153471` for a direct rank-1 vocabulary map.  More
importantly, A-LoRA vocabulary directions are constrained through the frozen
`lm_head` column space, while direct Vocab-LoRA can learn an arbitrary
vocabulary-side vector.  Results therefore isolate a strict raw-parameter
control; they do not establish equal function-class capacity.

## Frozen training endpoint

All six jobs use:

- corrected SFT train split, 22,780 examples, maximum sequence length 1024;
- one epoch, microbatch 8, gradient accumulation 2;
- AdamW through the existing trainer, learning rate `2e-4`;
- cosine scheduler, warmup ratio `0.03`, `max_grad_norm=0`;
- BF16 frozen backbone, FP32 boundary/master weights, TF32 denied;
- boundary dropout and bias zero; `save_strategy=no`.

There is no training-time endpoint selection or early stopping.  After the
single fixed endpoint, the dedicated fixed-boundary CE evaluator scores all
1,000 corrected dev examples and all 1,000 corrected test examples, preserving
per-example NLL records for paired comparisons.

## Fail-closed/restart contract

`run_fixed_hidden_boundary_fp32_qwen25.sh` performs these checks before
launching GPU work:

1. exactly six distinct, installed physical GPU indices are supplied;
2. train/dev/test and model-config hashes match this protocol;
3. every source hidden checkpoint has the pinned adapter/config/run-args
   hashes and the expected pure hidden-LoRA topology;
4. the current training entry and dedicated evaluator hashes are recorded in
   an immutable launch manifest.

A run is considered complete only when its terminal
`fixed_hidden_boundary_audit.json` exists and all saved tensors, source hashes,
precision flags, trainable/optimizer coverage, and run arguments validate.
Valid training and CE reports are skipped on restart.  Missing terminal
artifacts are recomputed.  A present but invalid terminal training audit fails
closed rather than being silently accepted.

The launcher uses one process per physical GPU:

```bash
GPU_IDS="0 1 2 3 4 5" \
  bash reviewer_followup/fixed_hidden_boundary_fp32_qwen25/run_fixed_hidden_boundary_fp32_qwen25.sh
```

The default IDs are `0 1 2 3 4 5`.  Changing code, data, source checkpoints,
or the pinned protocol after a launch manifest has been created requires a new
experiment directory rather than mixing artifacts.
