from datetime import UTC, datetime

import pytest

from app.cli import rag_report
from app.cli.rag_report import Group, format_report

SINCE = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
FLASH = "gemini:gemini-3.8-flash"
GROQ = "groq:openai/gpt-oss-120b"
GROUPS = [
    Group(FLASH, "ok", True, None, 6),
    Group(FLASH, "ok", False, None, 2),
    Group(FLASH, "empty", False, None, 1),
    Group(FLASH, "error", False, "timeout", 1),
    Group(GROQ, "ok", True, None, 1),
    Group(GROQ, "error", False, "timeout", 1),
    Group(GROQ, "error", False, "RateLimitError:429", 1),
    Group(GROQ, "unknown", False, None, 1),
]


def test_report_has_shares_kb_used_errors_and_links() -> None:
    report = format_report(GROUPS, 7, SINCE)

    assert report.splitlines() == [
        "RAG report: last 7 days (since 2026-10-26 12:00 UTC), 14 answers",
        "retrieval: ok 9 (64.3%)  empty 1 (7.1%)  error 3 (21.4%)  unknown 1 (7.1%)",
        "kb_used: 7 of 14 answers (50.0%); of retrieval ok: 77.8%",
        "errors: timeout 2, RateLimitError:429 1",
        "",
        "link                      answers      ok   empty   error  kb_used",
        "gemini:gemini-3.8-flash        10   80.0%   10.0%   10.0%    60.0%",
        "groq:openai/gpt-oss-120b        4   25.0%    0.0%   50.0%    25.0%",
    ]


def test_no_answers_is_one_line() -> None:
    assert format_report([], 1, SINCE) == (
        "RAG report: last 1 days (since 2026-10-26 12:00 UTC), 0 answers"
    )


@pytest.mark.parametrize("days", ["0", "-1"])
def test_cli_refuses_non_positive_days(days: str) -> None:
    with pytest.raises(SystemExit) as exc:
        rag_report.main(["--days", days])
    assert exc.value.code == 2


def test_cli_reports_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> None:
        raise ValueError("DATABASE_URL is required")

    monkeypatch.setattr(rag_report, "get_settings", broken)

    assert rag_report.main(["--days", "3"]) == 2
    assert "config error: DATABASE_URL is required" in capsys.readouterr().err
