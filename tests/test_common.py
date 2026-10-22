"""The provider-neutral helpers must reproduce, character for character, the output of the
helpers the research code used before they were isolated (reference strings captured from the
original implementation)."""

from collections import Counter

from expected_goals_lab.common import DataAccessError, count_table, table


def test_table_output_matches_the_original_helper():
    rows = [["a", 3, "50.0%"], ["b|c", 0, None], ["é", 1.5, ""]]
    expected = (
        "| Group | Shots | Share |\n|---|---|---|\n| a | 3 | 50.0% |\n"
        "| b|c | 0 | None |\n| é | 1.5 |  |"
    )
    assert table(["Group", "Shots", "Share"], rows) == expected
    assert table(["h"], []) == "| h |\n|---|"


def test_count_table_output_matches_the_original_helper():
    counts = Counter({"right": 5, "left": 3, "head": 2})
    expected = (
        "| Body part | Shots | Share |\n|---|---|---|\n| right | 5 | 50.0% |\n"
        "| left | 3 | 30.0% |\n| head | 2 | 20.0% |"
    )
    assert count_table("Body part", counts, 10) == expected


def test_data_access_error_is_a_runtime_error_with_the_original_meaning():
    assert issubclass(DataAccessError, RuntimeError)
    assert DataAccessError.__doc__ == "The data could not be fetched or was not valid JSON."
