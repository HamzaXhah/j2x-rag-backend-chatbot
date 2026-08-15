import pytest

from app.services.financial_analysis.presentation import (
    FinancialPresentationError,
    financial_presentation,
)


def test_create_chart_returns_presentable_markdown():
    result = financial_presentation.create_chart(
        title="Revenue trend",
        labels=["2022A", "2023A", "2024E"],
        values=[2.0, 3.5, 5.25],
        unit="€m",
    )

    assert result["title"] == "Revenue trend"
    assert "```text" in result["markdown"]
    assert "2022A" in result["markdown"]
    assert "5.25 €m" in result["markdown"]


def test_create_chart_rejects_mismatched_series():
    with pytest.raises(FinancialPresentationError):
        financial_presentation.create_chart(
            title="Revenue",
            labels=["2022A", "2023A", "2024E"],
            values=[2.0, 3.5],
        )


def test_create_chart_requires_at_least_three_points():
    with pytest.raises(FinancialPresentationError):
        financial_presentation.create_chart(
            title="Revenue",
            labels=["2022A", "2023A"],
            values=[2.0, 3.5],
        )
