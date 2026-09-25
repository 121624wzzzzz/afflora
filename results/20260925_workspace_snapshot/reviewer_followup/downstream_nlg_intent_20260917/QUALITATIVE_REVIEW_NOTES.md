# Qualitative review limits

During execution, inspected the first two MRs from the fixed hash panel (v4-fixed-qualitative-panel), Qwen2.5-1.5B seed8100, all four arms. This was an unblinded descriptive check, not a human evaluation study and not used to select outputs or modify the protocol.

- The Punter / pub / English / high / near Raja Indian Cuisine: Base invents menu items, decor and staff details. The regex scorer reports added=0, demonstrating that its fixed ontology cannot detect arbitrary unsupported claims. All three fitted arms produced the same concise description covering the supplied attributes.
- The Wrestlers / Japanese / riverside / not family friendly / moderate / near Raja Indian Cuisine: the adapted outputs conveyed the same attributes with different ordering. Such wording changes can affect BLEU without changing semantic correctness.

Consequently, heuristic SER is explicitly not an exhaustive hallucination metric. BLEU/chrF++ gains must not be represented as proven factual-quality gains. The full ten-MR fixed panel will be archived after all runs.
