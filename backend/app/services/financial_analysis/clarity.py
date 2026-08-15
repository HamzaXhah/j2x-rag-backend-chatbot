"""Pre-retrieval clarity checks for financial questions.

This module deliberately has no dependency on the retriever.  It turns a user
message (and, when applicable, a response to a prior clarification) into either
an answerable canonical query or one focused follow-up question.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field


LLMClarityCallable = Callable[[str], object]


class FinancialQuestionSlots(BaseModel):
    """Financial dimensions extracted before retrieval."""

    entity: Optional[str] = None
    metric: Optional[str] = None
    period: Optional[str] = None
    basis: Optional[str] = None
    currency: Optional[str] = None
    scope: Optional[str] = None
    comparison_entities: List[str] = Field(default_factory=list)


class ClarityDecision(BaseModel):
    """Stable service result consumed by the query orchestration layer."""

    decision: Literal["answer", "clarify"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    follow_up: Optional[str] = None
    options: List[str] = Field(default_factory=list)
    resolved_query: str
    extracted_slots: FinancialQuestionSlots


class FinancialQuestionClarityService:
    """Screen questions for material ambiguity before document retrieval.

    Deterministic checks are intentionally conservative: they only block when a
    missing dimension is likely to materially change a financial answer.  An
    optional synchronous ``llm_callable(prompt)`` can be injected to detect
    subtler ambiguity.  LLM clarification requests are accepted only when the
    model explicitly marks the ambiguity as material.
    """

    _COMMON_ENTITIES = ("Firecell", "Agent IQ")
    _ENTITY_REQUIRED_METRICS = {
        "assets",
        "book value",
        "business model",
        "cash",
        "competition",
        "customers",
        "debt",
        "ebitda",
        "equity",
        "funding",
        "management",
        "market size",
        "milestones",
        "partners",
        "profit",
        "product",
        "raise terms",
        "revenue",
        "risks",
        "valuation",
    }
    _AMBIGUOUS_METRICS = (
        (
            re.compile(r"\breturns?\b", re.IGNORECASE),
            ("IRR", "ROI", "ROE"),
            ("irr", "roi", "return on equity", "return on invested capital"),
        ),
        (
            re.compile(r"\bmargins?\b", re.IGNORECASE),
            ("Gross margin", "EBITDA margin", "Net margin"),
            ("gross margin", "ebitda margin", "net margin", "operating margin"),
        ),
        (
            re.compile(r"\bprofits?\b", re.IGNORECASE),
            ("Gross profit", "Operating profit", "Net profit"),
            ("gross profit", "operating profit", "net profit"),
        ),
        (
            re.compile(r"\bvalue\b", re.IGNORECASE),
            ("Enterprise value", "Equity value", "Book value"),
            (
                "enterprise value",
                "equity value",
                "book value",
                "company value",
                "valuation",
            ),
        ),
    )
    _METRIC_PATTERNS = (
        ("raise terms", r"\b(?:raise|funding|financing)\s+terms?\b"),
        ("market size", r"\b(?:tam|sam|som|market size)\b"),
        ("book value", r"\bbook value\b"),
        ("ebitda", r"\bebitda\b"),
        ("revenue", r"\b(?:revenue|sales|turnover|arr)\b"),
        ("debt", r"\b(?:debt|borrowings?|loans?|leverage)\b"),
        ("assets", r"\bassets?\b"),
        ("equity", r"\b(?:equity|net assets?)\b"),
        ("cash", r"\b(?:cash|liquidity|runway|burn)\b"),
        ("valuation", r"\b(?:valuation|enterprise value|equity value)\b"),
        ("margin", r"\bmargins?\b"),
        ("profit", r"\b(?:profits?|net income|earnings)\b"),
        ("customers", r"\b(?:customers?|clients?)\b"),
        ("product", r"\b(?:products?|services?|offering|solution)\b"),
        ("business model", r"\b(?:business model|revenue model|moneti[sz]ation)\b"),
        ("competition", r"\b(?:competitors?|competition|competitive)\b"),
        ("partners", r"\b(?:partners?|integrators?|alliances?)\b"),
        ("milestones", r"\b(?:milestones?|traction|achievements?)\b"),
        ("risks", r"\b(?:risks?|downside|concerns?|red flags?)\b"),
        ("management", r"\b(?:management|leadership|founders?|team)\b"),
        ("funding", r"\b(?:raise|funding|financing|capital)\b"),
    )

    _LLM_PROMPT = """
You are a pre-retrieval clarity gate for a financial-document analyst.
Return one JSON object only.

Ask a follow-up only when a missing or ambiguous detail could materially change
the answer. Do not block broad but legitimate requests for an overview, trend,
customer list, product description, risk assessment, or all available periods.
Do not ask for information the documents can determine after retrieval. If the
question is answerable as written, use decision "answer".

Required JSON:
{{
  "decision": "answer" or "clarify",
  "confidence": 0.0 to 1.0,
  "material_ambiguity": true or false,
  "reason": "short explanation",
  "follow_up": "one focused question or null",
  "options": ["up to three grounded choices"],
  "resolved_query": "canonical answerable question"
}}

User question:
{query}

Known document entities:
{known_entities}

Deterministically extracted slots:
{slots}

Complete conversation so far:
{conversation_history}
""".strip()

    _CONTEXT_RESOLUTION_PROMPT = """
You rewrite a context-dependent follow-up for a financial-document analyst.
Return one JSON object only. Do not answer the question.

Use the complete conversation to make the current message a concise,
self-contained question. Preserve the user's intent, company, metric,
timeframe, reporting basis, currency, and comparison scope. A follow-up such
as "What about 2024?" must inherit the subject and company from the prior
question while replacing the prior period. A follow-up such as "And EBITDA?"
must inherit the active company and relevant timeframe.

Ask one clarification only if the conversation contains multiple plausible
antecedents and choosing one would materially change the answer. Do not ask the
user to repeat information already present in the conversation.

Required JSON:
{{
  "decision": "answer" or "clarify",
  "confidence": 0.0 to 1.0,
  "material_ambiguity": true or false,
  "reason": "short explanation",
  "follow_up": "one focused question or null",
  "options": ["up to four choices grounded in the conversation"],
  "resolved_query": "standalone current question"
}}

Complete conversation:
{conversation_history}

Current user message:
{query}

Known document entities:
{known_entities}
""".strip()

    def __init__(
        self,
        llm_callable: Optional[LLMClarityCallable] = None,
    ) -> None:
        self.llm_callable = llm_callable

    def assess(
        self,
        question: str,
        conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
        known_entities: Optional[Sequence[str]] = None,
    ) -> ClarityDecision:
        """Return whether ``question`` can safely proceed to retrieval."""

        normalized_question = self._normalize(question)
        entities = self._normalize_entities(known_entities)
        history = tuple(conversation_history or ())
        resolved_query = self._resolve_prior_clarification(
            normalized_question,
            history,
            entities,
        )
        if (
            resolved_query == normalized_question
            and self._looks_context_dependent(
                normalized_question,
                history,
                entities,
            )
        ):
            resolved_query, contextual_clarification = (
                self._resolve_conversation_context(
                    normalized_question,
                    history,
                    entities,
                )
            )
            if contextual_clarification is not None:
                return contextual_clarification

        slots = self._extract_slots(resolved_query, entities)

        deterministic = self._deterministic_decision(resolved_query, slots, entities)
        if deterministic is not None:
            return deterministic

        if self.llm_callable is not None:
            llm_decision = self._llm_decision(
                resolved_query,
                slots,
                entities,
                history,
            )
            if llm_decision is not None:
                return llm_decision

        return ClarityDecision(
            decision="answer",
            confidence=0.92,
            reason="The entity, requested measure, and material scope are clear.",
            resolved_query=resolved_query,
            extracted_slots=slots,
        )

    def _deterministic_decision(
        self,
        query: str,
        slots: FinancialQuestionSlots,
        known_entities: Sequence[str],
    ) -> Optional[ClarityDecision]:
        if not query:
            return self._clarify(
                query=query,
                slots=slots,
                reason="The question is empty.",
                follow_up="What would you like to evaluate?",
                options=[],
                confidence=1.0,
            )

        ambiguous = self._ambiguous_metric(query)
        if ambiguous is not None:
            label, options = ambiguous
            return self._clarify(
                query=query,
                slots=slots,
                reason=f"The requested {label} has multiple material definitions.",
                follow_up=f"Which {label} do you want me to analyze?",
                options=list(options),
            )

        if self._requires_entity(query, slots):
            options = list(known_entities[:3])
            return self._clarify(
                query=query,
                slots=slots,
                reason="The company or entity is not identified.",
                follow_up="Which company should I analyze?",
                options=options,
            )

        return None

    def _llm_decision(
        self,
        query: str,
        slots: FinancialQuestionSlots,
        known_entities: Sequence[str],
        conversation_history: Sequence[Mapping[str, Any]],
    ) -> Optional[ClarityDecision]:
        prompt = self._LLM_PROMPT.format(
            query=query,
            known_entities=json.dumps(list(known_entities)),
            slots=slots.model_dump_json(),
            conversation_history=self._history_json(conversation_history),
        )
        try:
            raw = self.llm_callable(prompt)  # type: ignore[misc]
            payload = self._parse_llm_payload(raw)
        except Exception:
            # Clarity screening must degrade safely without taking the RAG
            # application offline when the optional model call fails.
            return None

        if payload.get("decision") != "clarify":
            return None
        if payload.get("material_ambiguity") is not True:
            return None

        follow_up = self._normalize(str(payload.get("follow_up") or ""))
        confidence = self._bounded_confidence(payload.get("confidence"))
        if not follow_up or confidence < 0.75:
            return None
        if not follow_up.endswith("?"):
            follow_up += "?"

        options = [
            self._normalize(str(option))
            for option in (payload.get("options") or [])
            if self._normalize(str(option))
        ][:3]
        return ClarityDecision(
            decision="clarify",
            confidence=confidence,
            reason=self._normalize(
                str(payload.get("reason") or "A material ambiguity remains.")
            ),
            follow_up=follow_up,
            options=options,
            resolved_query=self._normalize(
                str(payload.get("resolved_query") or query)
            ),
            extracted_slots=slots,
        )

    def _looks_context_dependent(
        self,
        question: str,
        history: Sequence[Mapping[str, Any]],
        known_entities: Sequence[str],
    ) -> bool:
        if not history or self._is_educational_question(question):
            return False

        lowered = question.lower().strip()
        if re.search(
            r"^(?:and\b|also\b|then\b|now\b|what about\b|how about\b|"
            r"why\b|what else\b|same\b|compare (?:it|that|them)\b)",
            lowered,
        ):
            return True
        if re.search(
            r"\b(?:it|its|they|their|them|that|this|these|those|"
            r"former|latter|previous|above|same)\b",
            lowered,
        ):
            return True

        prior_query = self._last_resolved_question(history)
        if not prior_query:
            return False
        current_slots = self._extract_slots(question, known_entities)
        prior_slots = self._extract_slots(prior_query, known_entities)
        if self._question_mentions_entity(question, known_entities):
            return False
        if not prior_slots.entity and not prior_slots.comparison_entities:
            return False
        if current_slots.metric or current_slots.period:
            return True
        return len(question.strip(" .?!").split()) <= 8

    def _question_mentions_entity(
        self,
        question: str,
        known_entities: Sequence[str],
    ) -> bool:
        lowered = question.lower()
        if any(
            re.search(
                rf"(?<!\w){re.escape(entity.lower())}(?!\w)",
                lowered,
            )
            for entity in (*known_entities, *self._COMMON_ENTITIES)
        ):
            return True
        return self._generic_possessive_entity(question) is not None

    def _resolve_conversation_context(
        self,
        question: str,
        history: Sequence[Mapping[str, Any]],
        known_entities: Sequence[str],
    ) -> tuple[str, Optional[ClarityDecision]]:
        current_slots = self._extract_slots(question, known_entities)
        if self.llm_callable is not None:
            prompt = self._CONTEXT_RESOLUTION_PROMPT.format(
                query=question,
                known_entities=json.dumps(
                    list(known_entities),
                    ensure_ascii=False,
                ),
                conversation_history=self._history_json(history),
            )
            try:
                payload = self._parse_llm_payload(self.llm_callable(prompt))
                confidence = self._bounded_confidence(
                    payload.get("confidence")
                )
                if (
                    payload.get("decision") == "clarify"
                    and payload.get("material_ambiguity") is True
                    and confidence >= 0.75
                ):
                    follow_up = self._normalize(
                        str(payload.get("follow_up") or "")
                    )
                    if follow_up:
                        options = [
                            self._normalize(str(option))
                            for option in (payload.get("options") or [])
                            if self._normalize(str(option))
                        ][:4]
                        return question, ClarityDecision(
                            decision="clarify",
                            confidence=confidence,
                            reason=self._normalize(
                                str(
                                    payload.get("reason")
                                    or "The follow-up has multiple possible meanings."
                                )
                            ),
                            follow_up=self._ensure_question(follow_up),
                            options=options,
                            resolved_query=question,
                            extracted_slots=current_slots,
                        )
                if payload.get("decision") == "answer" and confidence >= 0.6:
                    resolved_query = self._normalize(
                        str(payload.get("resolved_query") or "")
                    )
                    if 3 <= len(resolved_query) <= 4000:
                        return self._ensure_question(resolved_query), None
            except Exception:
                pass

        return (
            self._fallback_contextual_query(
                question,
                history,
                known_entities,
            ),
            None,
        )

    def _fallback_contextual_query(
        self,
        question: str,
        history: Sequence[Mapping[str, Any]],
        known_entities: Sequence[str],
    ) -> str:
        prior_query = self._last_resolved_question(history)
        if not prior_query:
            return question

        current_slots = self._extract_slots(question, known_entities)
        prior_slots = self._extract_slots(prior_query, known_entities)
        current_years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        prior_years = re.findall(r"\b(?:19|20)\d{2}\b", prior_query)
        if current_years and prior_years:
            replacement = current_years[0]
            resolved = re.sub(
                r"\b(?:19|20)\d{2}\b",
                replacement,
                prior_query,
            )
            return self._ensure_question(resolved)

        entity = current_slots.entity or prior_slots.entity
        metric = current_slots.metric
        if entity and metric:
            plural_metrics = {
                "competition",
                "customers",
                "management",
                "milestones",
                "partners",
                "risks",
            }
            verb = "are" if metric in plural_metrics else "is"
            period = current_slots.period or prior_slots.period
            basis = current_slots.basis or prior_slots.basis
            qualifiers = " ".join(
                value for value in (basis, period) if value
            )
            return self._ensure_question(
                f"What {verb} {entity}'s {qualifiers} {metric}"
            )

        return self._ensure_question(
            f"{question.rstrip(' ?.')} regarding {prior_query.rstrip(' ?.')}?"
        )

    def _last_resolved_question(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        for message in reversed(history):
            resolved = self._normalize(
                str(message.get("resolved_question") or "")
            )
            if resolved:
                return resolved
        for message in reversed(history):
            if message.get("role") == "user":
                content = self._normalize(str(message.get("content") or ""))
                if content:
                    return content
        return None

    def _history_json(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        turns: List[Dict[str, str]] = []
        for message in history:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            turn = {"role": role, "content": content}
            resolved = self._normalize(
                str(message.get("resolved_question") or "")
            )
            if resolved:
                turn["resolved_question"] = resolved
            turns.append(turn)
        return json.dumps(turns, ensure_ascii=False)

    def _extract_slots(
        self,
        query: str,
        known_entities: Sequence[str],
    ) -> FinancialQuestionSlots:
        lowered = query.lower()
        mentioned_entities = [
            entity
            for entity in known_entities
            if re.search(
                rf"(?<!\w){re.escape(entity.lower())}(?!\w)",
                lowered,
            )
        ]
        if not mentioned_entities:
            mentioned_entities = [
                entity
                for entity in self._COMMON_ENTITIES
                if re.search(
                    rf"(?<!\w){re.escape(entity.lower())}(?!\w)",
                    lowered,
                )
            ]
        if not mentioned_entities:
            generic_entity = self._generic_possessive_entity(query)
            if generic_entity:
                mentioned_entities = [generic_entity]
        if not mentioned_entities and len(known_entities) == 1:
            mentioned_entities = [known_entities[0]]

        metric = next(
            (
                name
                for name, pattern in self._METRIC_PATTERNS
                if re.search(pattern, lowered, re.IGNORECASE)
            ),
            None,
        )

        years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", query)))
        period: Optional[str] = ", ".join(years) or None
        if not period:
            for phrase in (
                "latest available",
                "most recent",
                "current",
                "historical",
                "forecast period",
                "all years",
                "all periods",
            ):
                if phrase in lowered:
                    period = phrase
                    break

        basis = None
        if re.search(r"\b(?:projected|forecast|estimated|target)\b", lowered):
            basis = "projected"
        if re.search(r"\b(?:actual|reported|historical)\b", lowered):
            basis = "actual" if basis is None else "actual and projected"
        if re.search(r"\b(?:both|actuals? and forecasts?)\b", lowered):
            basis = "actual and projected"

        currency_match = re.search(
            r"(?:\bUSD\b|\bEUR\b|\bGBP\b|[$€£])",
            query,
            re.IGNORECASE,
        )
        currency = currency_match.group(0).upper() if currency_match else None

        scope = None
        if re.search(r"\bby sector\b", lowered):
            scope = "by sector"
        elif re.search(r"\bby (?:customer|client)\b", lowered):
            scope = "by customer"
        elif re.search(r"\bby (?:country|region|geography)\b", lowered):
            scope = "by geography"
        elif re.search(r"\b(?:overview|all available|comprehensive)\b", lowered):
            scope = "broad"

        return FinancialQuestionSlots(
            entity=mentioned_entities[0] if len(mentioned_entities) == 1 else None,
            metric=metric,
            period=period,
            basis=basis,
            currency=currency,
            scope=scope,
            comparison_entities=mentioned_entities if len(mentioned_entities) > 1 else [],
        )

    def _requires_entity(
        self,
        query: str,
        slots: FinancialQuestionSlots,
    ) -> bool:
        if slots.entity or slots.comparison_entities:
            return False
        if slots.metric not in self._ENTITY_REQUIRED_METRICS:
            return False

        # Educational questions about a financial term do not require a
        # company. Everything else involving a company-specific fact does.
        return not self._is_educational_question(query)

    def _is_educational_question(self, query: str) -> bool:
        lowered = query.lower().strip(" ?.")
        return (
            lowered.startswith("define ")
            or lowered.startswith("how do you calculate ")
            or lowered.startswith("what does ")
            or lowered in {
                "what is ebitda",
                "what is irr",
                "what is roi",
                "what is revenue",
                "what is net debt",
            }
        )

    def _ambiguous_metric(
        self,
        query: str,
    ) -> Optional[tuple[str, tuple[str, ...]]]:
        lowered = query.lower()
        for pattern, options, precise_phrases in self._AMBIGUOUS_METRICS:
            if not pattern.search(query):
                continue
            if any(phrase in lowered for phrase in precise_phrases):
                continue
            label = pattern.pattern.replace(r"\b", "").replace("?", "")
            label = re.sub(r"[\[\]()+*\\s]", "", label).replace("s", "")
            display = {
                "return": "return metric",
                "margin": "margin",
                "profit": "profit measure",
                "value": "value measure",
            }.get(label, "financial measure")
            return display, options
        return None

    def _resolve_prior_clarification(
        self,
        current_question: str,
        history: Sequence[Mapping[str, Any]],
        known_entities: Sequence[str],
    ) -> str:
        pending = self._pending_clarification(history)
        if pending is None:
            return current_question

        original, follow_up = pending
        original_slots = self._extract_slots(original, known_entities)
        reply_slots = self._extract_slots(current_question, known_entities)
        lowered_follow_up = follow_up.lower()

        if (
            "which company" in lowered_follow_up
            or "which entity" in lowered_follow_up
            or "should i analyze" in lowered_follow_up
        ):
            entity = reply_slots.entity or self._standalone_entity(
                current_question,
                known_entities,
            )
            if entity:
                if original_slots.entity:
                    return self._replace_entity(
                        original,
                        original_slots.entity,
                        entity,
                    )
                return self._inject_entity(original, entity)

        if (
            not original_slots.period
            and "which period" in lowered_follow_up
        ):
            return self._append_qualifier(original, current_question)

        if (
            self._ambiguous_metric(original) is not None
            and re.search(
                r"\b(?:which|what)\s+(?:return|margin|profit|value)",
                lowered_follow_up,
            )
        ):
            return self._replace_ambiguous_metric(original, current_question)

        if re.search(
            r"\b(?:actual|projected|forecast|historical)\b",
            lowered_follow_up,
        ):
            return self._append_qualifier(original, current_question)

        if self._looks_like_short_clarification_reply(current_question):
            return self._append_qualifier(original, current_question)

        return current_question

    def _pending_clarification(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> Optional[tuple[str, str]]:
        if len(history) < 2:
            return None

        # Only the most recent assistant turn can be pending. Reusing an older
        # clarification after a later answer makes unrelated follow-ups attach
        # to the wrong question.
        assistant_index: Optional[int] = None
        for index in range(len(history) - 1, -1, -1):
            if history[index].get("role") == "assistant":
                assistant_index = index
                break
        if assistant_index is None:
            return None

        assistant_message = history[assistant_index]
        content = self._normalize(str(assistant_message.get("content") or ""))
        if not self._looks_like_clarification(content):
            return None
        resolved_question = self._normalize(
            str(assistant_message.get("resolved_question") or "")
        )
        if resolved_question:
            return resolved_question, content
        for user_index in range(assistant_index - 1, -1, -1):
            user_message = history[user_index]
            if user_message.get("role") == "user":
                original = self._normalize(
                    str(user_message.get("content") or "")
                )
                if original:
                    return original, content
        return None

    def _looks_like_clarification(self, content: str) -> bool:
        lowered = content.lower()
        return content.endswith("?") and any(
            phrase in lowered
            for phrase in (
                "which company",
                "which entity",
                "which period",
                "which year",
                "which return",
                "which margin",
                "which profit",
                "which value",
                "do you want",
                "should i use",
                "should i analyze",
                "should i show",
                "should i keep",
                "which currency",
            )
        )

    def _looks_like_short_clarification_reply(self, content: str) -> bool:
        words = content.strip(" .?!").split()
        if not words or len(words) > 10:
            return False
        return not re.match(
            r"^(?:what|why|who|when|where|how|can|could|would|"
            r"explain|analyze|analyse|provide|list)\b",
            content,
            re.IGNORECASE,
        )

    def _standalone_entity(
        self,
        reply: str,
        known_entities: Sequence[str],
    ) -> Optional[str]:
        lowered = reply.lower().strip(" .?!")
        for entity in (*known_entities, *self._COMMON_ENTITIES):
            if lowered == entity.lower():
                return entity
        if re.fullmatch(r"[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,2}", reply):
            return reply.strip()
        return None

    def _generic_possessive_entity(self, query: str) -> Optional[str]:
        match = re.search(
            r"([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3})['’]s\b",
            query,
        )
        if not match:
            return None
        words = match.group(1).split()
        while words and words[0].lower() in {
            "what",
            "which",
            "who",
            "how",
            "is",
            "are",
            "does",
        }:
            words.pop(0)
        return " ".join(words) or None

    def _inject_entity(self, query: str, entity: str) -> str:
        replacements = (
            (r"^(what|which)\s+is\s+(?:the\s+)?", rf"\1 is {entity}'s "),
            (r"^(what|which)\s+are\s+(?:the\s+)?", rf"\1 are {entity}'s "),
            (r"^(who)\s+are\s+(?:the\s+)?", rf"\1 are {entity}'s "),
            (r"^(how)\s+(?:big|large)\s+is\s+", rf"\1 big is {entity}'s "),
        )
        for pattern, replacement in replacements:
            resolved, count = re.subn(
                pattern,
                replacement,
                query,
                count=1,
                flags=re.IGNORECASE,
            )
            if count:
                return self._ensure_question(resolved)
        return self._append_qualifier(query, f"for {entity}")

    def _replace_entity(
        self,
        query: str,
        existing_entity: str,
        replacement_entity: str,
    ) -> str:
        resolved = re.sub(
            re.escape(existing_entity),
            replacement_entity,
            query,
            count=1,
            flags=re.IGNORECASE,
        )
        return self._ensure_question(resolved)

    def _replace_ambiguous_metric(self, query: str, replacement: str) -> str:
        resolved = re.sub(
            r"\b(?:returns?|margins?|profits?|value)\b",
            replacement.strip(" .?"),
            query,
            count=1,
            flags=re.IGNORECASE,
        )
        return self._ensure_question(resolved)

    def _append_qualifier(self, query: str, qualifier: str) -> str:
        return self._ensure_question(
            f"{query.rstrip(' ?.')} — {qualifier.strip(' ?.')}?"
        )

    def _clarify(
        self,
        query: str,
        slots: FinancialQuestionSlots,
        reason: str,
        follow_up: str,
        options: List[str],
        confidence: float = 0.96,
    ) -> ClarityDecision:
        return ClarityDecision(
            decision="clarify",
            confidence=confidence,
            reason=reason,
            follow_up=self._ensure_question(follow_up),
            options=options[:3],
            resolved_query=query,
            extracted_slots=slots,
        )

    def _parse_llm_payload(self, raw: object) -> Dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        text = str(raw or "").strip()
        fenced = re.search(r"\{[\s\S]*\}", text)
        if not fenced:
            raise ValueError("LLM clarity response did not contain a JSON object")
        payload = json.loads(fenced.group(0))
        if not isinstance(payload, dict):
            raise ValueError("LLM clarity response must be a JSON object")
        return payload

    def _normalize_entities(
        self,
        entities: Optional[Sequence[str]],
    ) -> List[str]:
        return list(
            dict.fromkeys(
                self._normalize(str(entity))
                for entity in (entities or ())
                if self._normalize(str(entity))
            )
        )

    def _normalize(self, value: str) -> str:
        return " ".join(value.strip().split())

    def _ensure_question(self, value: str) -> str:
        normalized = self._normalize(value).rstrip(".")
        return normalized if normalized.endswith("?") else f"{normalized}?"

    def _bounded_confidence(self, value: object) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0


financial_question_clarity_service = FinancialQuestionClarityService()
