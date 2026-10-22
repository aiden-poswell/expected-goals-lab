"""Small provider-neutral helpers shared by the Wyscout code and its reports.

These are the only pieces of the earlier, wider audit tooling that the Wyscout pipeline used: a
Markdown table writer and the error raised when source data cannot be fetched or read. They were
copied unchanged in behaviour (see ``tests/test_common.py``, which pins their exact output).
"""

from collections import Counter


class DataAccessError(RuntimeError):
    """The data could not be fetched or was not valid JSON."""


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def count_table(label: str, counter: Counter, total: int) -> str:
    rows = [[k, n, f"{100 * n / total:.1f}%"] for k, n in counter.most_common()]
    return table([label, "Shots", "Share"], rows)
