import unittest

from tasks.aime24.utils import (
    build_gpt_oss_reasoning_tags,
    clean_completions,
    configure_runtime_reasoning_tags,
    extract_vote_key,
    process_results,
    score_match,
)


class Aime24UtilsTest(unittest.TestCase):
    def tearDown(self):
        configure_runtime_reasoning_tags(None)

    def test_process_results_scores_boxed_answer(self):
        doc = {"Answer": "42"}
        result = process_results(doc, ["We simplify the equation and get \\boxed{42}."])
        self.assertEqual(result["exact_match"], 1)

    def test_process_results_uses_runtime_reasoning_tags(self):
        configure_runtime_reasoning_tags(build_gpt_oss_reasoning_tags())
        doc = {"Answer": "42"}
        result = process_results(
            doc,
            [
                "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>"
                "<|start|>assistant<|channel|>final<|message|>\\boxed{42}<|end|>"
            ],
        )
        self.assertEqual(result["exact_match"], 1)

    def test_score_match_accepts_assignment_style_answer(self):
        self.assertEqual(score_match("42", "x = 42"), 1.0)

    def test_clean_completions_removes_gpt_oss_reasoning(self):
        completions = [
            "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Therefore, the final answer is "
            "\\boxed{42}.<|end|>"
        ]
        self.assertEqual(
            clean_completions(completions, build_gpt_oss_reasoning_tags()),
            ["Therefore, the final answer is \\boxed{42}.<|end|>"],
        )

    def test_clean_completions_removes_think_tags(self):
        completions = ["<think>reasoning</think>Therefore, the final answer is \\boxed{42}."]
        self.assertEqual(
            clean_completions(completions, [("<think>", "</think>")]),
            ["Therefore, the final answer is \\boxed{42}."],
        )

    def test_unknown_parser_keeps_text(self):
        completions = ["Plain answer: \\boxed{42}"]
        self.assertEqual(clean_completions(completions), completions)

    def test_extract_vote_key_uses_extracted_target(self):
        first = extract_vote_key("I think the answer is \\boxed{42}.", "42")
        second = extract_vote_key("x = 42", "42")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
