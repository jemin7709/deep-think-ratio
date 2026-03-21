import unittest

from tasks.aime24.utils import process_results


class Aime24TaskFiltersTest(unittest.TestCase):
    def test_process_results_scores_single_answer(self):
        doc = {"Answer": "42"}
        result = process_results(doc, ["42"])
        self.assertEqual(result["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
