from types import SimpleNamespace

from app.services.retrieval.generator import (
    COMPANY_INVESTOR_QUESTIONS,
    RAGGenerator,
)


def test_curated_question_banks_cover_requested_companies():
    assert len(COMPANY_INVESTOR_QUESTIONS["firecell"]) == 15
    assert len(COMPANY_INVESTOR_QUESTIONS["agent_iq"]) == 7
    assert "What is Firecell’s projected 2027 EBITDA?" in (
        COMPANY_INVESTOR_QUESTIONS["firecell"]
    )
    assert "What are the key terms of Agent IQ’s current raise?" in (
        COMPANY_INVESTOR_QUESTIONS["agent_iq"]
    )


def test_document_names_select_both_company_question_banks():
    generator = RAGGenerator.__new__(RAGGenerator)
    documents = [
        SimpleNamespace(
            filename="Firecell - Investor Deck.pdf",
            title="Firecell Investor Deck",
        ),
        SimpleNamespace(
            filename="AIQ Investor Deck.pdf",
            title="Agent IQ",
        ),
    ]

    questions = generator._company_questions_for_documents(documents)

    assert questions[:15] == COMPANY_INVESTOR_QUESTIONS["firecell"]
    assert questions[15:] == COMPANY_INVESTOR_QUESTIONS["agent_iq"]
