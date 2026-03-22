import unittest

from tasks.aime24.utils import (
    clean_completions,
    extract_vote_key,
    process_results,
    resolve_reasoning_profile,
    score_match,
)


class Aime24UtilsTest(unittest.TestCase):
    def test_process_results_scores_boxed_answer(self):
        doc = {"Answer": "42"}
        result = process_results(doc, ["We simplify the equation and get \\boxed{42}."])
        self.assertEqual(result["exact_match"], 1)

    def test_score_match_accepts_assignment_style_answer(self):
        self.assertEqual(score_match("42", "x = 42", profile="identity"), 1.0)

    def test_clean_completions_removes_gpt_oss_reasoning(self):
        completions = [
            "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Therefore, the final answer is "
            "\\boxed{42}.<|end|>"
        ]
        self.assertEqual(
            clean_completions(completions, "gptoss_harmony"),
            ["Therefore, the final answer is \\boxed{42}."],
        )

    def test_clean_completions_removes_deepseek_reasoning(self):
        completions = ["<think>reasoning</think>Therefore, the final answer is \\boxed{42}."]
        self.assertEqual(
            clean_completions(completions, "deepseek_think_end"),
            ["Therefore, the final answer is \\boxed{42}."],
        )

    def test_clean_completions_removes_qwen_reasoning(self):
        completions = ["reasoning</think>Therefore, the final answer is \\boxed{42}."]
        self.assertEqual(
            clean_completions(completions, "qwen_think_end"),
            ["Therefore, the final answer is \\boxed{42}."],
        )

    def test_unknown_profile_keeps_text(self):
        completions = ["Plain answer: \\boxed{42}"]
        self.assertEqual(clean_completions(completions, "identity"), completions)

    def test_extract_vote_key_uses_extracted_target(self):
        first = extract_vote_key("I think the answer is \\boxed{42}.", "42", "identity")
        second = extract_vote_key("x = 42", "42", "identity")
        self.assertEqual(first, second)

    def test_resolve_reasoning_profile_handles_supported_families(self):
        self.assertEqual(resolve_reasoning_profile("openai/gpt-oss-120b"), "gptoss_harmony")
        self.assertEqual(resolve_reasoning_profile("deepseek-ai/DeepSeek-R1"), "deepseek_think_end")
        self.assertEqual(resolve_reasoning_profile("Qwen/Qwen3-32B"), "qwen_think_end")
        self.assertEqual(resolve_reasoning_profile("other/model"), "identity")


if __name__ == "__main__":
    unittest.main()
