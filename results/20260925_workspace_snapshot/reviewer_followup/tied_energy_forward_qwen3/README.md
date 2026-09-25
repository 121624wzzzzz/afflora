# Forward-consistent tied energy check: Qwen3-0.6B

This is a seed-42 diagnostic rerun of the tied-transpose energy configuration.
It is paired with the prior run in data, model, initialization, rank, alpha,
hidden LoRA, batch, optimizer, epoch count, tau (0.00625), and lambda (100).

The only semantic change is the energy metric for input/tied maps: it now
measures the update actually applied by `AffineEmbedding`,
`W D^T U^T + 1 beta^T`, rather than the output-head geometry `W U D`.
Checkpoints are saved every 250 steps for independent post-save dev CE checks.
