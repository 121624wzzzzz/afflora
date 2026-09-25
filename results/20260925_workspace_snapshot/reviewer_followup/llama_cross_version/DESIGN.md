# Llama-3.1/3.2 cross-version design

This experiment replaces the cancelled Gemma contrast and is fixed before
downstream results are observed.

| model | topology | selected boundary placement | affine R² | rank-95 of A-I | A-I energy rank 10 | A-I energy rank 20 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | untied | output/lm_head | 0.997575 (output) | 1,075 (output) | 0.342493 | 0.469000 |
| Llama-3.2-3B | tied | shared input/output | 0.979507 | 1,945 | 0.205936 | 0.296373 |

## Fixed protocol

- training: MetaMathQA 40k, one epoch;
- evaluation: complete MATH 5,000 (clean 4,995) and GSM8K 1,319;
- baseline: hidden LoRA rank 4, alpha 8;
- treatment: same hidden LoRA plus A-LoRA rank 16, alpha 16;
- Llama-3.1 output adapter has no affine translation beta;
- Llama-3.2 uses one mergeable adapter shared by tied input/output paths;
- seeds: 42, 43, 44;
- BF16 frozen base and FP32 trainable parameters;
- no downstream rank or scale selection.

Eight single-GPU workers pull jobs from one shared queue. A GPU that finishes a
run immediately takes the next pending run; there are no same-wave barriers.
Generation evaluation uses 16 shards per dataset in the same work-conserving
queue, so a GPU that finishes a shard immediately takes another shard, including
from a different model or benchmark when appropriate.
