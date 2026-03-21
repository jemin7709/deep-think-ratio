import unittest

from tasks.aime24.utils import extract_answer, is_equiv, strip_string


class Aime24UtilsTest(unittest.TestCase):
    def test_extracts_boxed_answer(self):
        response = "We simplify the equation and get \\boxed{42}."
        self.assertEqual(extract_answer(response), "42")

    def test_extracts_dollar_wrapped_answer(self):
        response = "The final answer is $84$."
        self.assertEqual(extract_answer(response), "84")

    def test_equates_assignment_style_answer(self):
        self.assertTrue(is_equiv("x = 42", "42"))

    def test_normalizes_fraction_spacing(self):
        self.assertEqual(strip_string(" 1/2 "), "\\frac{1}{2}")


if __name__ == "__main__":
    unittest.main()
