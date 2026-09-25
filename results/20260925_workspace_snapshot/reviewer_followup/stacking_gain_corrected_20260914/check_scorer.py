"""Exercise all canonical IFEval constraints and verify deterministic scoring."""
import json
import random
from unittest.mock import patch

import evaluate_generation as evaluation
from instruction_following_eval import instructions_registry as registry


def main():
    rows = evaluation.ce.load_rows(evaluation.HERE / "data/ifeval.jsonl")
    fallbacks = []
    for row in rows:
        for name, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
            with patch.object(random, "choice", wraps=random.choice) as choice, \
                 patch.object(random, "randint", wraps=random.randint) as randint, \
                 patch.object(random, "sample", wraps=random.sample) as sample:
                instruction = registry.INSTRUCTION_DICT[name](name)
                instruction.build_description(**kwargs)
                if any(function.called for function in (choice, randint, sample)):
                    fallbacks.append({"key": row["key"], "instruction": name, "kwargs": kwargs})
    assert {row["key"] for row in fallbacks} == {1122, 1129}
    responses = [{"response": "This is a short English sentence. Another sentence tests the local scorer.\n"
                  "A second paragraph contains a numbered example: 1. Hello world.",
                  "generated_tokens": 32, "hit_token_cap": False} for _ in rows]
    first = evaluation.score(rows, responses)
    random.seed(9821)
    for _ in range(73): random.random()
    second = evaluation.score(rows, responses)
    assert first == second
    assert first["valid_prompt_sensitivity"]["examples"] == 539
    for row in first["per_example"]:
        if row["key"] not in (1122, 1129):
            assert not row["strict"] or row["loose"]
    result = {"status": "passed", "examples": len(rows), "instructions": first["instructions"],
              "distinct_instruction_types": len({i for r in rows for i in r["instruction_id_list"]}),
              "deterministic_repeat": True, "upstream_random_fallbacks": fallbacks,
              "note": "Synthetic responses exercise scorer dependencies only; not model results."}
    evaluation.support.write_json(evaluation.HERE / "scorer_check.json", result)
    print(json.dumps(result))


if __name__ == "__main__": main()
