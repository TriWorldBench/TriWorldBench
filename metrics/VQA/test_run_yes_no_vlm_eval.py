#!/usr/bin/env python3

import unittest

from run_yes_no_vlm_eval import parse_model_responses, score_answer


QUESTIONS = [
    {
        "selections": {"A": "one", "B": "two", "C": "three", "D": "four"},
    }
]


class MultiAnswerEvaluationTest(unittest.TestCase):
    def test_parses_single_answer(self) -> None:
        parsed = parse_model_responses('[{"reasoning":"visible","answer":"A"}]', QUESTIONS)
        self.assertEqual(parsed[0]["answer"], "A")

    def test_parses_and_deduplicates_multiple_answers(self) -> None:
        parsed = parse_model_responses(
            '[{"reasoning":"visible","answer":["d","B","D"]}]', QUESTIONS
        )
        self.assertEqual(parsed[0]["answer"], ["D", "B"])

    def test_rejects_invalid_code_in_multiple_answers(self) -> None:
        parsed = parse_model_responses(
            '[{"reasoning":"visible","answer":["A","Z"]}]', QUESTIONS
        )
        self.assertIsNone(parsed[0]["answer"])

    def test_multi_answer_scoring_requires_exact_set(self) -> None:
        self.assertEqual(score_answer(["D", "B"], ["B", "D"]), 1)
        self.assertEqual(score_answer("B", ["B", "D"]), 0)
        self.assertEqual(score_answer(["A", "B"], "A"), 0)

    def test_malformed_response_is_unanswered(self) -> None:
        parsed = parse_model_responses("not json", QUESTIONS)
        self.assertIsNone(parsed[0]["answer"])


if __name__ == "__main__":
    unittest.main()
