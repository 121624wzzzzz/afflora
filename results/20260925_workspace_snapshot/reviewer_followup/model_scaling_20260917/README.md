# Qwen2.5 model-size extension: 3B / 7B

Completed: 60 fitted runs, 4 Base evaluations, 4 smoke runs on CLUENER and WikiSQL. Five paired seeds per task; fixed 2048-example training sets and common hyperparameters. Verified 1.5B results are historical anchors.

Observed mean stack gains over budget LoRA: CLUENER +1.675 / +1.247 F1 and WikiSQL +1.152 / +0.684 percentage points for 3B / 7B. All four means exceed both controls, but only the 3B WikiSQL vs exact-budget contrast has a positive corrected lower bound; no new condition passes both controls. The last 7B WikiSQL seed ties budget and loses one test query to ordinary LoRA.

3B: exact parameter match. 7B: budget LoRA has 512 MORE trainable parameters. Tasks selected after earlier positive results; size trends also mix tied/untied architectures.

Read [full Chinese report](FINAL_INTERPRETATION_ZH.md), [frozen protocol](PROTOCOL.md), [methods and limits](METHOD_AND_LIMITS_ZH.md), [primary numbers](TEST_RESULTS.md), [diagnostics](CONTENT_DIAGNOSTICS.json), and [paired-effect plot](figures/new_model_effects.png).

Audit: 90432 new responses, 37872 historical anchor responses, 13838 new token records, 68 scopes, 20 new paired initialization/order groups. No fitted adapter reused for new model runs.

After sealing this directory is immutable. verify_seal.py is read-only; do not rerun writer scripts in a sealed archive.
