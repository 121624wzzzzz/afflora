# Corrected full MATH evaluation

This directory evaluates the historical MetaMathQA checkpoints on all 5,000
MATH test questions with the following corrections:

- slice generated tokens at the padded batch input width;
- parse the last `\\boxed{...}` expression with balanced braces;
- preserve the global dataset index in every result;
- report both the official full-5000 score and a clean-4995 score that excludes
  five confirmed test questions with training-equivalent MetaMathQA records;
- support deterministic multi-GPU sharding and merge-time integrity checks.

The historical training checkpoints are not modified.

