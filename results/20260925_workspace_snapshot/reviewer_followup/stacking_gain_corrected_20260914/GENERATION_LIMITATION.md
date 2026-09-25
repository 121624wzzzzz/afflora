# Generation limitation found during monitoring

This is an observation about the local frozen base checkpoints, not a change to the registered matrix.

## Qwen3-0.6B-Base

The tied output embedding rows for IDs 151644 (`<|im_start|>`) and 151645 (`<|im_end|>`) are exactly equal. All 56 inspected rows 151644 through 151699 match the end-token row exactly. See `special_token_diagnostic.json`.

Every arm keeps the base vocabulary output matrix frozen. Hidden LoRA and boundary A-LoRA change the vector supplied to that matrix; they do not give its identical rows distinct parameters. Therefore, if w_i = w_j, both logits remain w_i^T h = w_j^T h for every adapted hidden state. The model cannot learn to prefer the assistant end token over its identical lower-ID start token through any arm in this matrix. The same limitation applies to the budget control. This is separate from the generation-engine stop-list fix.

The supervised end-token probability is at most 1/56 from these inspected identical rows alone, so its token NLL is at least log(56). The CE reports remain reproducible measures under the frozen objective, but the objective includes this structural limitation. A small overall CE reduction does not establish usable chat generation.

## Qwen2.5-7B-Base

The inspected start/end output rows differ by at most 6.103515625e-05; they are not exactly equal. The exact impossibility statement above is not established for this model. Both fixed synthetic short-answer prompts nevertheless reached the 128-token diagnostic limit; this is an observed outcome, not proof of its cause.

## Reporting implications

- Keep every registered arm, seed, CE result and generation outcome. Do not choose or retune conditions based on these observations.
- All valid generation uses base EOS plus native im_end stopping. Earlier generation without this rule is archived and excluded.
- Report generation accuracy alongside length, token-cap hits and this frozen-head limitation. Do not describe current CE gains as established practical generation gains.
- A later experiment intended to support general chat utility needs a compatible pretrained chat head or a common, explicitly controlled adaptation of special-token output rows. That would be a new training protocol; the present checkpoints could not silently serve as matched baselines for it.
