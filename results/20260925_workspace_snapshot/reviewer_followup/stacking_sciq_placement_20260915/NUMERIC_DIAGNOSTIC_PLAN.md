# Additional numerical check

Added while seed2005 is running, after24 confirmation endpoints were inspected.
This is a post hoc measurement diagnostic, not a confirmatory endpoint or a
change to the frozen protocol. Original primary scores stay unchanged.

For all40 completed checkpoints, screen all4,000 test-rotation queries. Flag
top-two candidate logit gaps <=0.001, twice the preexisting0.0005 batch-shape
check tolerance. Independently reload only checkpoints with flagged queries
and rerun those queries individually in the same FP32 eager path. Report every
changed prediction and whether errors meet the existing tolerance. The screen
does not establish a global error bound for all possible tensor shapes.

The purpose is to check whether observed near ties are numerically fragile
when an effect may amount to only a few questions. No examples are removed,
no alternate score replaces the original, and no model/configuration is selected
from this check.
