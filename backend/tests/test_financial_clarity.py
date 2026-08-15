from app.services.financial_analysis.clarity import (
    FinancialQuestionClarityService,
)


def test_vague_entity_requests_one_focused_follow_up():
    result = FinancialQuestionClarityService().assess(
        "What is the revenue for 2023?",
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "clarify"
    assert result.follow_up == "Which company should I analyze?"
    assert result.options == ["Firecell", "Agent IQ"]
    assert result.extracted_slots.metric == "revenue"


def test_ambiguous_metric_requests_definition():
    result = FinancialQuestionClarityService().assess(
        "What is Firecell's return for 2027?"
    )

    assert result.decision == "clarify"
    assert "return metric" in result.reason
    assert result.options == ["IRR", "ROI", "ROE"]


def test_missing_period_proceeds_to_retrieval_for_grounded_year_choices():
    result = FinancialQuestionClarityService().assess(
        "What is Firecell's projected EBITDA?"
    )

    assert result.decision == "answer"
    assert result.follow_up is None
    assert result.extracted_slots.period is None


def test_clear_firecell_question_proceeds_directly():
    result = FinancialQuestionClarityService().assess(
        "What is Firecell's projected 2027 EBITDA?"
    )

    assert result.decision == "answer"
    assert result.follow_up is None
    assert result.resolved_query == "What is Firecell's projected 2027 EBITDA?"
    assert result.extracted_slots.entity == "Firecell"
    assert result.extracted_slots.metric == "ebitda"
    assert result.extracted_slots.period == "2027"
    assert result.extracted_slots.basis == "projected"


def test_follow_up_answer_resolves_original_question():
    history = [
        {"role": "user", "content": "Who are the main customers?"},
        {"role": "assistant", "content": "Which company should I analyze?"},
    ]

    result = FinancialQuestionClarityService().assess(
        "Firecell",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "Who are Firecell's main customers?"
    assert result.extracted_slots.entity == "Firecell"
    assert result.extracted_slots.metric == "customers"


def test_optional_llm_can_flag_only_material_semantic_ambiguity():
    def clarity_llm(_prompt):
        return {
            "decision": "clarify",
            "confidence": 0.9,
            "material_ambiguity": True,
            "reason": "The comparison basis is missing.",
            "follow_up": "Should I compare growth, profitability, or risk?",
            "options": ["Growth", "Profitability", "Risk"],
            "resolved_query": "Which company is better?",
        }

    result = FinancialQuestionClarityService(
        llm_callable=clarity_llm
    ).assess("Which company is better?")

    assert result.decision == "clarify"
    assert result.options == ["Growth", "Profitability", "Risk"]


def test_broad_investor_request_does_not_over_ask():
    result = FinancialQuestionClarityService().assess(
        "Give me a comprehensive overview of Firecell's financial performance."
    )

    assert result.decision == "answer"


def test_educational_metric_definition_needs_no_entity_or_period():
    result = FinancialQuestionClarityService().assess("What is EBITDA?")

    assert result.decision == "answer"
    assert result.extracted_slots.metric == "ebitda"


def test_short_evidence_conflict_choice_resolves_original_question():
    history = [
        {
            "role": "user",
            "content": "What is Firecell's projected debt in 2026?",
        },
        {
            "role": "assistant",
            "content": (
                "The documents contain different values. "
                "Should I show both and compare their source dates?"
            ),
        },
    ]

    result = FinancialQuestionClarityService().assess(
        "Show and compare both",
        conversation_history=history,
        known_entities=["Firecell"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == (
        "What is Firecell's projected debt in 2026 — Show and compare both?"
    )


def test_evidence_entity_choice_replaces_original_entity():
    history = [
        {
            "role": "user",
            "content": "What is Agent IQ's revenue in 2023?",
        },
        {
            "role": "assistant",
            "content": (
                "Should I analyze Agent IQ, or use the retrieved Firecell "
                "documents instead?"
            ),
        },
    ]

    result = FinancialQuestionClarityService().assess(
        "Firecell",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "What is Firecell's revenue in 2023?"


def test_chained_clarification_uses_assistant_resolved_question():
    history = [
        {"role": "user", "content": "What is the revenue for 2023?"},
        {"role": "assistant", "content": "Which company should I analyze?"},
        {"role": "user", "content": "Firecell"},
        {
            "role": "assistant",
            "content": "Which currency should I use for the analysis?",
            "resolved_question": "What is Firecell's revenue for 2023?",
        },
    ]

    result = FinancialQuestionClarityService().assess(
        "EUR",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "What is Firecell's revenue for 2023 — EUR?"
    assert result.extracted_slots.entity == "Firecell"
    assert result.extracted_slots.currency == "EUR"


def test_contextual_follow_up_is_rewritten_from_complete_thread():
    prompts = []

    def clarity_llm(prompt):
        prompts.append(prompt)
        if "You rewrite a context-dependent follow-up" in prompt:
            return {
                "decision": "answer",
                "confidence": 0.96,
                "material_ambiguity": False,
                "reason": "The active company and metric are clear.",
                "follow_up": None,
                "options": [],
                "resolved_query": "What was Firecell's revenue in 2024?",
            }
        return {
            "decision": "answer",
            "confidence": 0.95,
            "material_ambiguity": False,
            "resolved_query": "What was Firecell's revenue in 2024?",
        }

    history = [
        {
            "role": "user",
            "content": "Start with Firecell's product differentiation.",
        },
        {
            "role": "assistant",
            "content": "Firecell differentiates through private 5G features.",
            "resolved_question": "What is Firecell's product differentiation?",
        },
        {
            "role": "user",
            "content": "What was its revenue in 2023?",
        },
        {
            "role": "assistant",
            "content": "The documents report the 2023 figure.",
            "resolved_question": "What was Firecell's revenue in 2023?",
        },
    ]

    result = FinancialQuestionClarityService(
        llm_callable=clarity_llm
    ).assess(
        "What about 2024?",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "What was Firecell's revenue in 2024?"
    assert "Start with Firecell's product differentiation." in prompts[0]
    assert "What was Firecell's revenue in 2023?" in prompts[0]


def test_contextual_follow_up_has_deterministic_fallback():
    history = [
        {
            "role": "assistant",
            "content": "Firecell's 2023 revenue is disclosed.",
            "resolved_question": "What was Firecell's revenue in 2023?",
        }
    ]

    result = FinancialQuestionClarityService().assess(
        "What about 2024?",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "What was Firecell's revenue in 2024?"


def test_follow_up_metric_inherits_company_and_period():
    history = [
        {
            "role": "assistant",
            "content": "Firecell's 2023 revenue is disclosed.",
            "resolved_question": "What was Firecell's revenue in 2023?",
        }
    ]

    result = FinancialQuestionClarityService().assess(
        "And EBITDA?",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "What is Firecell's 2023 ebitda?"


def test_old_clarification_is_not_reused_after_a_later_answer():
    history = [
        {"role": "user", "content": "What is the revenue in 2023?"},
        {"role": "assistant", "content": "Which company should I analyze?"},
        {"role": "user", "content": "Firecell"},
        {
            "role": "assistant",
            "content": "Firecell's 2023 revenue is disclosed.",
            "resolved_question": "What is Firecell's revenue in 2023?",
        },
    ]

    result = FinancialQuestionClarityService().assess(
        "What about 2024?",
        conversation_history=history,
        known_entities=["Firecell", "Agent IQ"],
    )

    assert result.decision == "answer"
    assert result.resolved_query == "What is Firecell's revenue in 2024?"
