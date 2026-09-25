import unittest
from scoring import decode_output, extract_answer


class AnswerTests(unittest.TestCase):
    def test_content_and_format_are_separate(self):
        choices = ['adult', 'phenotype', 'male', 'clone']
        for text in ['D', 'D.', 'D. clone', ' D because it is genetically identical.',
                     'The correct answer is D.', 'Answer: D', '**D**. clone', '(D) clone', 'clone']:
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text, choices)['prediction'], 3)

    def test_does_not_guess_or_match_letters_inside_words(self):
        choices = ['adult', 'phenotype', 'male', 'clone']
        for text in ['', 'Because the answer needs more context', 'A or B', 'A and D',
                     'D / B', 'D. adult', 'The correct answer is A. clone', 'I do not know']:
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text, choices)['prediction'])

    def test_exact_option_and_prefix_collisions(self):
        self.assertIsNone(extract_answer('clone', ['clone', 'clone', 'male', 'adult'])['prediction'])
        self.assertEqual(extract_answer('A. oxygen', ['oxygen', 'oxygen atoms', 'male', 'adult'])['prediction'], 0)
        self.assertIsNone(extract_answer('A. oxygen atoms', ['oxygen', 'oxygen atoms', 'male', 'adult'])['prediction'])
        self.assertEqual(extract_answer('B. oxygen atoms', ['oxygen', 'oxygen atoms', 'male', 'adult'])['prediction'], 1)

    def test_native_end_padding_and_caps(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return ''.join({35: 'D', 13: '.', 99: ' clone'}[i] for i in ids)
        choices = ['adult', 'phenotype', 'male', 'clone']
        parsed = decode_output([35, 13, 99, 151643, 151643], Tokenizer(), [151643], choices, 3, 128)
        self.assertTrue(parsed['correct'])
        self.assertFalse(parsed['strict_correct'])
        self.assertTrue(parsed['completed_correct'])
        self.assertEqual(parsed['generated_ids'], [35, 13, 99, 151643])
        capped = decode_output([35], Tokenizer(), [151643], choices, 3, 1)
        self.assertTrue(capped['correct'])
        self.assertTrue(capped['hit_length_cap'])
        self.assertFalse(capped['completed_correct'])


if __name__ == '__main__':
    unittest.main()
