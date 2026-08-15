from types import SimpleNamespace

from app.services.financial_analysis.evidence import FinancialEvidenceService


def test_financial_query_selects_expected_topics():
    service = FinancialEvidenceService()

    labels = service._labels_for_query(
        "Assess revenue growth, EBITDA margin, debt, and liquidity"
    )

    assert "Revenue ($M)" in labels
    assert "EBITDA" in labels
    assert "TOTAL ASSETS" in labels
    assert "CASH AND SHORT-TERM DEPOSITS" in labels


def test_evidence_format_includes_exact_document_locator():
    service = FinancialEvidenceService()
    chunk = SimpleNamespace(
        document_id="doc-1",
        content="TURNOVER,100,125",
        chunk_metadata={
            "filename": "model.xlsx",
            "sheet": "Yearly IS",
            "row_range": "0-19",
        },
    )

    formatted = service._format_chunks([chunk])

    assert "Source: model.xlsx" in formatted
    assert "Sheet: Yearly IS" in formatted
    assert "Rows: 0-19" in formatted
