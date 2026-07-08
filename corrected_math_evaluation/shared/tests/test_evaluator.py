from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_gsm8k_full import extract_gsm8k_answer, normalize_gsm8k_answer
from evaluate_math_full import extract_math_answer, normalize_math_answer


class MathAnswerTest(unittest.TestCase):
    def test_nested_box(self) -> None:
        self.assertEqual(extract_math_answer(r"work \boxed{\frac{1}{11}}"), r"\frac{1}{11}")

    def test_uses_last_box(self) -> None:
        self.assertEqual(extract_math_answer(r"\boxed{2} then \boxed{3}"), "3")

    def test_latex_normalization(self) -> None:
        self.assertEqual(normalize_math_answer(r"$ \dfrac{1}{2} $"), r"\frac{1}{2}")

    def test_compact_fraction_normalization(self) -> None:
        self.assertEqual(normalize_math_answer(r"\frac78"), r"\frac{7}{8}")

    def test_units_and_assignment_normalization(self) -> None:
        self.assertEqual(normalize_math_answer(r"a=5\text{ digits}"), "5")


class GSM8KAnswerTest(unittest.TestCase):
    def test_ground_truth_marker(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work\n#### 70,000"), "70,000")

    def test_generated_answer_marker(self) -> None:
        self.assertEqual(extract_gsm8k_answer("The answer is: $18."), "$18")
        self.assertEqual(normalize_gsm8k_answer("$18"), "18")

    def test_gsm8k_normalization(self) -> None:
        self.assertEqual(normalize_gsm8k_answer("$70,000.0"), "7E+4")


if __name__ == "__main__":
    unittest.main()
