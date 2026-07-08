from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from data_pipeline import ConversationFormatError, messages_from_row, tokenize_conversation


class ConversationValidationTest(unittest.TestCase):
    def test_rejects_trailing_user(self) -> None:
        row = {
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "unanswered"},
            ]
        }
        with self.assertRaisesRegex(ConversationFormatError, "does_not_end"):
            messages_from_row(row)

    def test_rejects_consecutive_assistants(self) -> None:
        row = {
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer one"},
                {"role": "assistant", "content": "answer two"},
            ]
        }
        with self.assertRaisesRegex(ConversationFormatError, "strictly_alternate"):
            messages_from_row(row)


class NativeTemplateMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoTokenizer

        cls.tokenizers = [
            AutoTokenizer.from_pretrained(ROOT / "../models/Qwen2.5-0.5B-Base"),
            AutoTokenizer.from_pretrained(ROOT / "../models/Qwen3-0.6B-Base"),
        ]

    def test_only_assistant_content_is_supervised(self) -> None:
        row = {
            "conversations": [
                {"role": "user", "content": "UNIQUE_USER_ALPHA"},
                {"role": "assistant", "content": "UNIQUE_ASSISTANT_ONE"},
                {"role": "user", "content": "UNIQUE_USER_BETA"},
                {"role": "assistant", "content": "UNIQUE_ASSISTANT_TWO"},
            ]
        }
        for tokenizer in self.tokenizers:
            item = tokenize_conversation(row, tokenizer, 1024)
            supervised_ids = [
                token_id
                for token_id, label in zip(item["input_ids"], item["labels"])
                if label != -100
            ]
            supervised = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            self.assertIn("UNIQUE_ASSISTANT_ONE", supervised)
            self.assertIn("UNIQUE_ASSISTANT_TWO", supervised)
            self.assertNotIn("UNIQUE_USER_ALPHA", supervised)
            self.assertNotIn("UNIQUE_USER_BETA", supervised)
            self.assertEqual(supervised.count("<|im_end|>"), 2)

    def test_real_row_has_supervised_tokens(self) -> None:
        row = json.loads(
            (ROOT / "data/sft_t2t_mini_25k/train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        for tokenizer in self.tokenizers:
            item = tokenize_conversation(row, tokenizer, 1024)
            self.assertTrue(any(label != -100 for label in item["labels"]))
            self.assertLessEqual(len(item["input_ids"]), 1024)


if __name__ == "__main__":
    unittest.main()

