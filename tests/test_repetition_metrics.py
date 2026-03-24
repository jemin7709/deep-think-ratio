import unittest
from unittest.mock import patch

from tasks.aime24.utils import clean_completions

from src.experiment.repetition_metrics import mean_seq_rep_n_for_completions


class RepetitionMetricsTest(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self, mapping: dict[str, list[int]]):
            self._mapping = mapping

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return self._mapping.get(text, [0])

    def test_mean_seq_rep_n_word_level(self):
        score = mean_seq_rep_n_for_completions(
            ["a b c d", "x x x x"],
            n=2,
            level="word",
            model_name="dummy",
        )
        self.assertAlmostEqual(score, (0.0 + (1.0 - 1 / 3)) / 2)

    @patch("src.experiment.repetition_metrics.AutoTokenizer.from_pretrained")
    def test_mean_seq_rep_n_token_level(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.FakeTokenizer(
            {
                "dup": [1, 2, 1, 2],
                "unique": [3, 4, 5],
            }
        )
        score = mean_seq_rep_n_for_completions(
            ["dup", "unique"],
            n=2,
            level="token",
            model_name="token-model",
        )
        self.assertAlmostEqual(score, (1.0 - 2 / 3) / 2)

    @patch("src.experiment.repetition_metrics.AutoTokenizer.from_pretrained")
    def test_seq_rep_n_short_sequence_is_zero(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.FakeTokenizer(
            {"short-token-completion": [1]}
        )

        word_score = mean_seq_rep_n_for_completions(
            ["a b"],
            n=4,
            level="word",
            model_name="dummy",
        )
        self.assertEqual(word_score, 0.0)

        score = mean_seq_rep_n_for_completions(
            ["short-token-completion"],
            n=4,
            level="token",
            model_name="short-token-model",
        )
        self.assertEqual(score, 0.0)

    @patch("src.experiment.repetition_metrics.AutoTokenizer.from_pretrained")
    def test_reasoning_tags_are_removed_before_metric(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.FakeTokenizer(
            {"x y x y": [1, 2, 1, 2]}
        )

        completion = "<think>noise noise</think> x y x y"
        score = mean_seq_rep_n_for_completions(
            [completion],
            n=2,
            level="word",
            model_name="reasoning-model",
            reasoning_tags=[("<think>", "</think>")],
        )

        cleaned = clean_completions([completion], [("<think>", "</think>")])[0]
        expected_tokens = cleaned.split()
        ngram_count = len(expected_tokens) - 2 + 1
        expected = 1.0 - len({tuple(expected_tokens[i : i + 2]) for i in range(ngram_count)}) / ngram_count
        self.assertAlmostEqual(score, expected)


if __name__ == "__main__":
    unittest.main()
