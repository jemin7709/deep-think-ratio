import unittest

from tasks.aime24.utils import (
    build_gpt_oss_reasoning_tags,
    configure_runtime_reasoning_tags,
    process_results,
)


class Aime24TaskFiltersTest(unittest.TestCase):
    def tearDown(self):
        configure_runtime_reasoning_tags(None)

    def test_process_results_scores_single_answer(self):
        doc = {"Answer": "42"}
        result = process_results(doc, ["42"])
        self.assertEqual(result["exact_match"], 1)

    def test_process_results_scores_gpt_oss_output(self):
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


if __name__ == "__main__":
    unittest.main()
