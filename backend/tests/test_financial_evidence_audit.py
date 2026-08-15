from types import SimpleNamespace

from app.services.financial_analysis.evidence_audit import (
    FinancialEvidenceAuditService,
)


def _document(text, company, source="deck.pdf", **metadata):
    return SimpleNamespace(
        page_content=text,
        metadata={"company": company, "filename": source, **metadata},
    )


def test_entity_mismatch_requires_clarification():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2023?",
        documents=[
            _document(
                "Agent IQ reported 2023 revenue of $10 million.",
                "Agent IQ",
                source="agent-iq-deck.pdf",
            )
        ],
    )

    assert result.decision == "clarify"
    assert "Firecell" in result.follow_up
    assert "Agent IQ" in result.reason
    assert result.verified_context == ""


def test_missing_period_requires_clarification_when_multiple_periods_exist():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What is Firecell's revenue?",
        documents=[
            _document(
                "Firecell actual revenue was €5 million in 2023.",
                "Firecell",
                page_num=4,
            ),
            _document(
                "Firecell actual revenue was €8 million in 2024.",
                "Firecell",
                page_num=5,
            ),
        ],
    )

    assert result.decision == "clarify"
    assert "2023 to 2024" in result.follow_up
    assert result.options == ["2023", "2024", "All available periods"]


def test_clear_evidence_is_verified_for_answering():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's actual revenue in 2023?",
        documents=[
            _document(
                "Firecell reported actual revenue of €5 million in 2023.",
                "Firecell",
                source="financial-model.xlsx",
                sheet="P&L",
                row_range="12-12",
            )
        ],
    )

    assert result.decision == "answer"
    assert result.confidence >= 0.9
    assert "€5 million" in result.verified_context
    assert "Sheet: P&L" in result.verified_context
    assert result.checks["requested_periods"] == ["2023"]


def test_conflicting_figures_require_source_resolution():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2024?",
        documents=[
            _document(
                "Firecell revenue in 2024 was €10 million.",
                "Firecell",
                source="board-pack.pdf",
            ),
            _document(
                "Firecell revenue in 2024 was €12 million.",
                "Firecell",
                source="investor-deck.pdf",
            ),
        ],
    )

    assert result.decision == "clarify"
    assert "conflicting" in result.reason.lower()
    assert result.checks["conflicts"][0]["values"] == [
        10_000_000.0,
        12_000_000.0,
    ]
    assert "Show and compare both" in result.options


def test_absent_evidence_is_missing_not_an_ambiguity():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2028?",
        documents=[
            _document(
                "Firecell provides private 5G networking products.",
                "Firecell",
                source="product-overview.pdf",
            )
        ],
    )

    assert result.decision == "missing"
    assert result.follow_up is None
    assert "not disclose revenue" in result.limitations[0].lower()
    assert result.verified_context == ""


def test_requested_period_not_disclosed_is_missing_not_a_follow_up():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2028?",
        documents=[
            _document(
                "Firecell reported revenue of €8 million in 2024.",
                "Firecell",
                source="financial-model.xlsx",
            )
        ],
    )

    assert result.decision == "missing"
    assert result.follow_up is None
    assert "2028" in result.reason
    assert "2024" in result.limitations[0]


def test_mixed_currencies_require_a_conversion_choice():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "Compare Firecell's revenue in 2024.",
        documents=[
            _document(
                "Firecell revenue in 2024 was €10 million in Europe.",
                "Firecell",
            ),
            _document(
                "Firecell revenue in 2024 was $12 million in the United States.",
                "Firecell",
            ),
        ],
    )

    assert result.decision == "clarify"
    assert result.options == [
        "Keep reported currencies",
        "I will provide an exchange rate",
    ]
    assert "currency" in result.reason.lower()


def test_simple_question_keeps_mixed_currencies_without_over_asking():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2024?",
        documents=[
            _document(
                "Firecell revenue in 2024 was EUR 10 million in Europe.",
                "Firecell",
            ),
            _document(
                "Firecell revenue in 2024 was $12 million in the United States.",
                "Firecell",
            ),
        ],
    )

    assert result.decision == "answer"
    assert "reported currency" in result.limitations[0]


def test_broad_trend_request_does_not_force_one_period():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "Show Firecell's historical revenue trend.",
        documents=[
            _document(
                "Firecell actual revenue was EUR 5 million in 2023.",
                "Firecell",
            ),
            _document(
                "Firecell actual revenue was EUR 8 million in 2024.",
                "Firecell",
            ),
        ],
    )

    assert result.decision == "answer"


def test_decimal_financial_value_is_not_treated_as_a_year():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2024?",
        documents=[
            _document(
                "Firecell revenue in 2024 was EUR 2007.866 million.",
                "Firecell",
            )
        ],
    )

    assert result.decision == "answer"
    assert result.checks["evidence_periods"] == ["2024"]


def test_requested_forecast_is_missing_when_only_actuals_exist():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What is Firecell's projected revenue in 2024?",
        documents=[
            _document(
                "Firecell reported actual revenue of €10 million in 2024.",
                "Firecell",
            )
        ],
    )

    assert result.decision == "missing"
    assert result.follow_up is None
    assert "forecast" in result.limitations[0]


def test_multi_period_series_is_not_misread_as_conflicting_values():
    service = FinancialEvidenceAuditService()

    result = service.audit(
        "What was Firecell's revenue in 2024?",
        documents=[
            _document(
                (
                    "Firecell revenue was EUR 10 million in 2022, "
                    "EUR 12 million in 2023, and EUR 15 million in 2024."
                ),
                "Firecell",
                source="financial-model.xlsx",
            )
        ],
    )

    assert result.decision == "answer"
    assert result.checks["conflicts"] == []


def test_optional_llm_can_raise_semantic_clarification_but_not_replace_evidence():
    def review(payload):
        assert payload["deterministic_audit"]["decision"] == "answer"
        return {
            "decision": "clarify",
            "confidence": 0.85,
            "reason": "The accounting definition is unclear.",
            "follow_up": "Do you mean reported or adjusted revenue?",
            "options": ["Reported revenue", "Adjusted revenue"],
            "limitations": ["The source uses both definitions."],
        }

    service = FinancialEvidenceAuditService(llm_callable=review)
    result = service.audit(
        "What was Firecell's revenue in 2023?",
        documents=[
            _document(
                "Firecell revenue in 2023 was €5 million.",
                "Firecell",
                source="model.xlsx",
            )
        ],
    )

    assert result.decision == "clarify"
    assert result.checks["llm_reviewed"] is True
    assert "€5 million" in result.verified_context
