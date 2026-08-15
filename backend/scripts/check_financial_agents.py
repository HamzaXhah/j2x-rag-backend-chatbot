"""Live, synthetic smoke test for Xbot's three LLM-backed analysis stages.

This script deliberately uses invented Acme figures. It never reads or sends
uploaded document content, so it is safe for checking the configured provider.
"""

from __future__ import annotations

import json

from app.services.financial_analysis.evidence_audit import (
    financial_evidence_audit_service,
)
from app.services.retrieval.generator import (
    FINANCIAL_ANALYST_SYSTEM_PROMPT,
    rag_generator,
)


def main() -> int:
    clarity = rag_generator.clarity_service.assess(
        "Which company is the better investment?",
        known_entities=["Acme", "BetaCo"],
    )
    contextual_follow_up = rag_generator.clarity_service.assess(
        "What about 2024?",
        conversation_history=[
            {
                "role": "user",
                "content": "What was Acme's revenue in 2023?",
            },
            {
                "role": "assistant",
                "content": "Acme's 2023 revenue is disclosed.",
                "resolved_question": "What was Acme's revenue in 2023?",
            },
        ],
        known_entities=["Acme", "BetaCo"],
    )

    evidence = [
        {
            "content": (
                "Acme reported revenue of EUR 10 million in 2022, "
                "EUR 12 million in 2023, and EUR 15 million in 2024."
            ),
            "metadata": {"filename": "Synthetic Acme financials"},
        }
    ]
    audit = financial_evidence_audit_service.audit(
        query="What was Acme's revenue in 2024?",
        documents=evidence,
        llm_callable=rag_generator._call_evidence_agent,
    )

    analyst_prompt = """
VERIFIED DOCUMENT CONTEXT
[Verified Evidence | Source: Synthetic Acme financials]
Acme reported revenue of EUR 10 million in 2022, EUR 12 million in 2023,
and EUR 15 million in 2024.

RESOLVED USER QUESTION
Show the revenue trend, calculate CAGR from 2022 to 2024, and include a chart.
""".strip()
    answer, tool_results = rag_generator.call_llm_with_financial_tools(
        analyst_prompt,
        FINANCIAL_ANALYST_SYSTEM_PROMPT,
    )

    summary = {
        "clarity": {
            "decision": clarity.decision,
            "has_follow_up": bool(clarity.follow_up),
        },
        "thread_memory": {
            "decision": contextual_follow_up.decision,
            "resolved_query": contextual_follow_up.resolved_query,
        },
        "evidence_audit": {
            "decision": audit.decision,
            "confidence": audit.confidence,
        },
        "analyst": {
            "has_answer": bool(answer.strip()),
            "tool_results": [
                result.get("operation")
                or ("financial_chart" if result.get("markdown") else "unknown")
                for result in tool_results
            ],
        },
    }
    print(json.dumps(summary, indent=2))

    if clarity.decision != "clarify":
        return 1
    resolved_follow_up = contextual_follow_up.resolved_query.lower()
    if (
        contextual_follow_up.decision != "answer"
        or "acme" not in resolved_follow_up
        or "revenue" not in resolved_follow_up
        or "2024" not in resolved_follow_up
    ):
        return 5
    if audit.decision != "answer":
        return 2
    if not answer.strip():
        return 3
    if "cagr" not in summary["analyst"]["tool_results"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
