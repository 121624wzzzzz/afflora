# V4: fine-grained intent and natural language generation

Fixed before any model output, 2026-09-17. This is a new two-task family, not a replacement for any previous negative result.

## Sources and data

Official Banking77 (PolyAI-LDN/task-specific-datasets, revision 57ec275d8078af65b7731c2a98be812d844a6d6b): official test 3080 retained including one duplicate text. Normalize NFKC/casefold/whitespace for overlap removal from official train, then deterministic hash partition. Five examples per class for development (385), training 26 per class plus one for 46 hash-selected classes (2048). No test label enters training selection. Categories sorted alphabetically, two-digit codes 00..76. The full category-name ontology is in every prompt; this is definition-aware generative classification, not a standard encoder leaderboard setup.

Cleaned E2E (tuetschek/e2e-cleaning, revision 164acdf986146d76f18893b3de9488725deda4c2): use corrected mr, never orig_mr. Canonical MR is the sorted multiset of slot/value pairs; preserve multiple values for a slot. Group all references, de-duplicate identical references. Official nonoverlapping train/dev/test MRs independently verified. Hash-select 2048 training MRs and 256 dev MRs; retain all 1847 test MRs. One reference per training MR selected by an independent fixed hash; all references retained for evaluation. These are regex-cleaned annotations, not perfect human semantic labels. Multiple-valued or inconsistent MRs retained, and reported separately as a diagnostic if necessary.

No truncation: all tokenized prompt/target lengths checked before training. Native EOS 151643, no chat template. No fitted adapter reuse; source and 14 original model files verified against sealed V3 archive and hashes.

## Conditions

Qwen3-0.6B-Base and Qwen2.5-1.5B official verified original weights. Base plus ordinary hidden LoRA (r8/alpha16 on q/k/v/o/up/down/gate), exact parameter budget hidden LoRA, and hidden LoRA plus input/output A-LoRA (boundary rank16/alpha128; input bias only). Exact budget adds ranks to deterministic q/k modules with scale alpha/r=2. Shared r8 initialization and example order are identical across paired arms. Only declared adapter parameters train; original tensors frozen and hashed.

Five paired seeds 8100..8104; common LR 2e-4; 2048 examples, one epoch, 64 optimizer steps, effective batch32/microbatch4, answer-token mean loss including EOS; AdamW(.9,.999), eps1e-8, weight decay0, clip1, 3% warmup (2steps), cosine. Frozen weights BF16, trainables/optimizer FP32; BF16 autocast training, FP32 evaluation, TF32 off, SDPA. No task-specific hyperparameter search or checkpoint selection. Final checkpoint only. Four seed8099 two-step smoke runs verify both architectures/models plus evaluation. 60 formal trained runs + 4 Base evaluations. GPU pool1..6, require memory<512MiB; exclude0/7 and occupied GPUs.

## Evaluation

Banking77 primary: accuracy from argmax full two-token code joint conditional log probability over77 candidates. No candidate-only normalization or EOS in candidate score. All codes verified length2 for both models. Exact prefix-cache computation checked against eight independent full-prefix forwards for two examples on every evaluation (absolute logprob tolerance2e-4, identical argmax). Eval batch8. Macro-F1 secondary. All77 scores saved.

E2E primary: corpus SacreBLEU2.5.1, tokenizer13a, case-sensitive, exponential smoothing, effective_order=False, all available references per MR. Secondary chrF++ (word_order2), heuristic slot error rate and no-slot-error percentage, generation length/EOS/cap. FP32 greedy decoding, batch16, max256 new tokens. No retries, post-hoc repairs, beam search, reranking or output normalization. Save exact token IDs and text. Public corpus_score independently verifies sufficient-statistic aggregation; never average sentence BLEU.

Official slot regex scorer is retained unmodified. A separately named copy fixes one stale-variable bug in repeated-value accumulation (see SLOT_SCORER_PATCH.md). Save both outputs. Both use gold-assisted regex disambiguation, are incomplete heuristic diagnostics, and do not establish human-rated faithfulness. Corrected scorer tested for missing/wrong/repeated attributes before model execution.

## Statistics and scope

Eight primary test contrasts: A-LoRA stack minus ordinary and minus equal-budget LoRA for each of2tasks×2models. Report five individual paired differences, mean, sample SD, paired Student-t confidence interval with familywise Bonferroni8 correction (two-sided95%, df4). Same descriptive analysis on development separately, not used to choose runs/tasks. Formal positive evidence requires both corrected lower bounds>0; retain all zero/negative/uncertain findings. Report Base, all arms and secondary metrics. These intervals quantify training-seed uncertainty conditional on this frozen test split, prompt and recipe; no population/general-purpose benefit claim, no global multiplicity claim across earlier task explorations. No data bootstrap is a primary inference. No optional stopping.

Before final interpretation: rehash code/data/model sources; audit all parameter scopes, pairwise initialization and orders, frozen gradients/tensors, save/reload, all tokenization and saved prediction/score records; recompute public multi-reference metrics. Archive failures and deviations explicitly. Seal final artifacts with a manifest after reporting.
