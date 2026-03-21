"""Local YAML shim for the upstream AIME task helpers."""

from lm_eval.tasks.aime import utils as upstream_utils


fix_a_slash_b = upstream_utils.fix_a_slash_b
fix_fracs = upstream_utils.fix_fracs
fix_sqrt = upstream_utils.fix_sqrt
is_equiv = upstream_utils.is_equiv
last_boxed_only_string = upstream_utils.last_boxed_only_string
remove_boxed = upstream_utils.remove_boxed
remove_right_units = upstream_utils.remove_right_units
strip_string = upstream_utils.strip_string


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    answer_key = next(key for key in doc if key.lower() == "answer")
    return {
        "exact_match": int(is_equiv(extract_answer(results[0]), str(doc[answer_key])))
    }


def extract_answer(response: str) -> str:
    answer = response

    dollar_positions = [pos for pos, char in enumerate(response) if char == "$"]
    if len(dollar_positions) > 1:
        answer = response[dollar_positions[0] + 1 : dollar_positions[-1]]

    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass

    return answer
