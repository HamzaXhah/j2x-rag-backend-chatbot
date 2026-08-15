from app.services.financial_analysis.clarity import (
    ClarityDecision,
    FinancialQuestionSlots,
)
from app.services.financial_analysis.evidence_audit import EvidenceAuditResult
from app.services.retrieval import generator as generator_module
from app.schemas.schemas import QueryRequest


class _Document:
    def __init__(self, text="Revenue 2023: EUR 10 million"):
        self.page_content = text
        self.metadata = {
            "document_id": "doc-1",
            "filename": "Firecell.pdf",
            "page_num": 4,
        }


class _Retriever:
    def __init__(self, result=None):
        self.result = result or {
            "context": "[Source: Firecell.pdf]\nRevenue 2023: EUR 10 million",
            "documents": [_Document()],
        }
        self.calls = 0

    def retrieve_for_rag(self, **kwargs):
        self.calls += 1
        self.query = kwargs["query"]
        return self.result


class _Supplement:
    def supplement(self, **_kwargs):
        return "", []


class _Audit:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def audit(self, **kwargs):
        self.calls += 1
        self.query = kwargs["query"]
        return self.result


class _Clarity:
    def __init__(self, result):
        self.result = result

    def assess(self, **_kwargs):
        return self.result


def _generator(clarity):
    service = generator_module.RAGGenerator.__new__(
        generator_module.RAGGenerator
    )
    service.client = object()
    service.clarity_service = _Clarity(clarity)
    return service


def _answer_clarity(query):
    return ClarityDecision(
        decision="answer",
        confidence=0.95,
        reason="Clear",
        resolved_query=query,
        extracted_slots=FinancialQuestionSlots(
            entity="Firecell",
            metric="revenue",
            period="2023",
        ),
    )


def test_query_contract_preserves_resolved_question_in_history():
    request = QueryRequest(
        query="EUR",
        conversation_history=[
            {
                "role": "assistant",
                "content": "Which currency should I use?",
                "resolved_question": "What is Firecell's revenue in 2023?",
            }
        ],
    )

    assert request.conversation_history == [
        {
            "role": "assistant",
            "content": "Which currency should I use?",
            "resolved_question": "What is Firecell's revenue in 2023?",
        }
    ]


def test_query_contract_preserves_complete_thread_beyond_twenty_turns():
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"Thread turn {index}",
        }
        for index in range(30)
    ]

    request = QueryRequest(
        query="Continue the analysis",
        conversation_history=history,
    )

    assert len(request.conversation_history) == 30
    assert request.conversation_history[0]["content"] == "Thread turn 0"
    assert request.conversation_history[-1]["content"] == "Thread turn 29"

    long_turn = "x" * 12000
    long_request = QueryRequest(
        query="Continue",
        conversation_history=[{"role": "assistant", "content": long_turn}],
    )
    assert long_request.conversation_history[0]["content"] == long_turn


def test_orchestrator_stops_before_retrieval_when_clarification_is_needed(
    monkeypatch,
):
    retriever = _Retriever()
    monkeypatch.setattr(generator_module, "rag_retriever", retriever)
    service = _generator(
        ClarityDecision(
            decision="clarify",
            confidence=0.98,
            reason="The company is missing.",
            follow_up="Which company should I analyze?",
            options=["Firecell", "Agent IQ"],
            resolved_query="What is the revenue for 2023?",
            extracted_slots=FinancialQuestionSlots(
                metric="revenue",
                period="2023",
            ),
        )
    )

    result = service.generate_response("What is the revenue for 2023?")

    assert result["status"] == "clarification_required"
    assert result["options"] == ["Firecell", "Agent IQ"]
    assert result["metrics"]["decision_stage"] == "clarity"
    assert retriever.calls == 0


def test_orchestrator_stops_before_analysis_on_evidence_ambiguity(monkeypatch):
    query = "What is Firecell's revenue in 2023?"
    retriever = _Retriever()
    audit = _Audit(
        EvidenceAuditResult(
            decision="clarify",
            confidence=0.97,
            reason="Two sources report conflicting figures.",
            follow_up="Should I show and compare both figures?",
            options=["Show and compare both", "Use latest source"],
        )
    )
    monkeypatch.setattr(generator_module, "rag_retriever", retriever)
    monkeypatch.setattr(
        generator_module, "financial_evidence_service", _Supplement()
    )
    monkeypatch.setattr(
        generator_module, "financial_evidence_audit_service", audit
    )
    service = _generator(_answer_clarity(query))
    service.call_llm_with_financial_tools = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("analyst must not run"))
    )

    result = service.generate_response(query)

    assert result["status"] == "clarification_required"
    assert result["metrics"]["decision_stage"] == "evidence_audit"
    assert audit.calls == 1


def test_orchestrator_returns_presentable_missing_disclosure(monkeypatch):
    query = "What is Firecell's revenue in 2035?"
    retriever = _Retriever()
    audit = _Audit(
        EvidenceAuditResult(
            decision="missing",
            confidence=0.98,
            reason="No evidence was found for 2035.",
            limitations=["The requested period is not disclosed."],
        )
    )
    monkeypatch.setattr(generator_module, "rag_retriever", retriever)
    monkeypatch.setattr(
        generator_module, "financial_evidence_service", _Supplement()
    )
    monkeypatch.setattr(
        generator_module, "financial_evidence_audit_service", audit
    )
    service = _generator(_answer_clarity(query))

    result = service.generate_response(query)

    assert result["status"] == "answered"
    assert "Information not verified" in result["answer"]
    assert "not estimated or inferred" in result["answer"]
    assert result["metrics"]["agent_trace"]["analyst"] == "not_run"


def test_orchestrator_passes_only_audited_context_to_analyst(monkeypatch):
    query = "What about 2024?"
    resolved_query = "What is Firecell's revenue in 2024?"
    history = [
        {
            "role": "user",
            "content": "Begin with Firecell's product differentiation.",
        },
        {
            "role": "assistant",
            "content": "Firecell provides private networking products.",
            "resolved_question": "What is Firecell's product differentiation?",
        },
        {
            "role": "user",
            "content": "What was its revenue in 2023?",
        },
        {
            "role": "assistant",
            "content": "The 2023 figure is disclosed in the model.",
            "resolved_question": "What is Firecell's revenue in 2023?",
        },
    ]
    retriever = _Retriever()
    audit = _Audit(
        EvidenceAuditResult(
            decision="answer",
            confidence=0.94,
            reason="Evidence matches.",
            verified_context=(
                "[Verified Evidence | Source: Firecell.pdf | Page: 4]\n"
                "Revenue 2023: EUR 10 million"
            ),
        )
    )
    monkeypatch.setattr(generator_module, "rag_retriever", retriever)
    monkeypatch.setattr(
        generator_module, "financial_evidence_service", _Supplement()
    )
    monkeypatch.setattr(
        generator_module, "financial_evidence_audit_service", audit
    )
    service = _generator(_answer_clarity(resolved_query))
    captured = {}

    def analyst(prompt, system_prompt):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "### Direct answer\n\nRevenue was **EUR 10 million**.", []

    service.call_llm_with_financial_tools = analyst

    result = service.generate_response(
        query,
        conversation_history=history,
    )

    assert result["status"] == "answered"
    assert "EUR 10 million" in result["answer"]
    assert "Verified Evidence" in captured["prompt"]
    assert retriever.query == resolved_query
    assert "Begin with Firecell's product differentiation." in captured["prompt"]
    assert "What is Firecell's revenue in 2023?" in captured["prompt"]
    assert "RESOLVED USER QUESTION" in captured["prompt"]
    assert resolved_query in captured["prompt"]
    assert result["metrics"]["agent_trace"] == {
        "clarity": "passed",
        "evidence_audit": "passed",
        "analyst": "answered",
    }
