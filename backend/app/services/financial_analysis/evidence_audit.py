"""Post-retrieval evidence checks for financial questions.

This module deliberately does not retrieve data.  It inspects the output of the
existing retriever and decides whether the evidence is safe to pass to the
financial analyst, requires a user clarification, or simply does not disclose
the requested information.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


AuditLLMCallable = Callable[[Dict[str, Any]], Mapping[str, Any] | str]


@dataclass
class EvidenceAuditResult:
    """Structured result consumed by the query orchestration layer."""

    decision: str
    confidence: float
    reason: str
    follow_up: Optional[str] = None
    options: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    verified_context: str = ""
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _EvidenceItem:
    text: str
    metadata: Mapping[str, Any]


METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "turnover", "arr", "top line", "top-line"),
    "ebitda": ("ebitda",),
    "profit": ("net income", "net profit", "profit", "earnings", "loss"),
    "margin": ("margin",),
    "debt": ("debt", "borrowings", "loan", "loans", "leverage"),
    "assets": ("asset", "assets"),
    "book_value": ("book value", "shareholders' equity", "shareholder equity"),
    "cash": ("cash", "liquidity", "runway", "burn"),
    "cash_flow": ("cash flow", "free cash flow", "fcf"),
    "valuation": ("valuation", "enterprise value", "equity value", "multiple"),
    "tam": ("tam", "total addressable market", "market size"),
    "customers": ("customer", "customers", "client", "clients"),
    "product": (
        "product",
        "offering",
        "solution",
        "feature",
        "features",
        "use case",
        "use cases",
    ),
    "competition": ("competitor", "competitors", "competition", "competitive"),
    "funding": (
        "raise",
        "funding",
        "financing round",
        "use of proceeds",
        "use of funds",
    ),
    "management": (
        "management",
        "leadership",
        "founder",
        "ceo",
        "cfo",
        "coo",
        "cto",
        "team",
    ),
    "partners": ("partner", "partners", "system integrator", "integrators"),
    "milestones": ("milestone", "milestones", "traction", "achievement"),
}

TIME_SENSITIVE_METRICS = {
    "revenue",
    "ebitda",
    "profit",
    "margin",
    "debt",
    "assets",
    "book_value",
    "cash",
    "cash_flow",
    "valuation",
}

BASIS_ALIASES: Dict[str, Tuple[str, ...]] = {
    "forecast": (
        "forecast",
        "projected",
        "projection",
        "estimate",
        "estimated",
        "budget",
        "target",
    ),
    "actual": ("actual", "historical", "reported", "audited"),
}

KNOWN_ENTITIES: Dict[str, Tuple[str, ...]] = {
    "Firecell": ("firecell",),
    "Agent IQ": ("agent iq", "agentiq"),
}

_YEAR_RE = re.compile(
    r"(?<![\d.])(?:19|20)\d{2}(?:\.0)?(?!\d)(?!\.\d)"
)
_BROAD_PERIOD_TERMS = (
    "all available",
    "all periods",
    "all years",
    "historical",
    "history",
    "latest",
    "most recent",
    "over time",
    "trend",
)
_POSSESSIVE_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})[’']s\b"
)
_AMOUNT_RE = re.compile(
    r"(?P<currency>[$€£])?\s*"
    r"(?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<scale>bn|billion|mm|m|million|k|thousand|%)\b",
    re.IGNORECASE,
)


class FinancialEvidenceAuditService:
    """Validate whether retrieved evidence can support a financial answer."""

    def __init__(self, llm_callable: Optional[AuditLLMCallable] = None):
        self.llm_callable = llm_callable

    def audit(
        self,
        query: str,
        context: str = "",
        documents: Optional[Sequence[Any]] = None,
        llm_callable: Optional[AuditLLMCallable] = None,
    ) -> EvidenceAuditResult:
        items = self._coerce_evidence(context=context, documents=documents)
        metrics = self._extract_metrics(query)
        requested_years = self._year_values(query)
        requested_entity = self._extract_query_entity(query)
        requested_basis = self._extract_basis(query)

        checks: Dict[str, Any] = {
            "requested_entity": requested_entity,
            "requested_metrics": sorted(metrics),
            "requested_periods": sorted(requested_years),
            "requested_basis": sorted(requested_basis),
            "evidence_items": len(items),
        }

        if not items or not any(item.text.strip() for item in items):
            return self._missing(
                "No relevant document evidence was retrieved.",
                "The requested information is not disclosed in the retrieved documents.",
                checks,
            )

        entity_items, observed_entities = self._entity_aligned_items(
            items, requested_entity
        )
        checks["observed_entities"] = sorted(observed_entities)
        if requested_entity and not entity_items:
            observed_label = ", ".join(sorted(observed_entities)) or "other documents"
            return EvidenceAuditResult(
                decision="clarify",
                confidence=0.98,
                reason=(
                    f"The question asks about {requested_entity}, but the retrieved "
                    f"evidence appears to concern {observed_label}."
                ),
                follow_up=(
                    f"Should I analyze {requested_entity}, or use the retrieved "
                    f"{observed_label} documents instead?"
                ),
                options=[requested_entity, *sorted(observed_entities)],
                limitations=["The retrieved evidence does not match the named company."],
                checks=checks,
            )
        candidate_items = entity_items or items

        metric_items, missing_metrics = self._metric_aligned_items(
            candidate_items, metrics
        )
        checks["missing_metrics"] = sorted(missing_metrics)
        if metrics and not metric_items:
            metric_label = ", ".join(sorted(metrics))
            return self._missing(
                f"The evidence does not contain the requested metric: {metric_label}.",
                (
                    f"The retrieved documents do not disclose {metric_label}. "
                    "No figure should be inferred."
                ),
                checks,
            )
        candidate_items = metric_items or candidate_items

        evidence_years = self._extract_years(candidate_items)
        checks["evidence_periods"] = sorted(evidence_years)
        if requested_years and not requested_years.intersection(evidence_years):
            requested = ", ".join(sorted(requested_years))
            available = ", ".join(sorted(evidence_years))
            limitation = f"The requested period ({requested}) is not disclosed."
            if available:
                limitation += f" Retrieved evidence covers {available}."
            return self._missing(
                f"No evidence was found for the requested period {requested}.",
                limitation,
                checks,
            )

        matched_items = self._period_aligned_items(candidate_items, requested_years)
        if requested_years and matched_items:
            candidate_items = matched_items

        if (
            not requested_years
            and metrics.intersection(TIME_SENSITIVE_METRICS)
            and len(evidence_years) > 1
            and not any(term in query.lower() for term in _BROAD_PERIOD_TERMS)
        ):
            periods = sorted(evidence_years)
            return EvidenceAuditResult(
                decision="clarify",
                confidence=0.95,
                reason=(
                    "The requested financial metric varies by reporting period, and "
                    "the retrieved evidence contains multiple periods."
                ),
                follow_up=(
                    f"The documents contain figures from {periods[0]} to "
                    f"{periods[-1]}. Which period or range should I analyze?"
                ),
                options=[*periods[-3:], "All available periods"],
                limitations=["A period is required to select the correct figure."],
                checks=checks,
            )

        evidence_basis = self._extract_basis(
            "\n".join(item.text for item in candidate_items)
        )
        checks["evidence_basis"] = sorted(evidence_basis)
        if requested_basis and evidence_basis and not requested_basis.intersection(
            evidence_basis
        ):
            requested_label = ", ".join(sorted(requested_basis))
            return self._missing(
                f"The evidence does not contain the requested {requested_label} basis.",
                (
                    f"Only {', '.join(sorted(evidence_basis))} evidence was retrieved; "
                    f"{requested_label} data is not disclosed."
                ),
                checks,
            )
        if not requested_basis and {"actual", "forecast"}.issubset(evidence_basis):
            return EvidenceAuditResult(
                decision="clarify",
                confidence=0.92,
                reason=(
                    "The retrieved evidence contains both historical actuals and "
                    "management forecasts for the requested metric."
                ),
                follow_up="Should I use historical actuals, management forecasts, or both?",
                options=["Historical actuals", "Management forecasts", "Both"],
                limitations=["Actual and forecast figures must not be mixed silently."],
                checks=checks,
            )

        requested_units = self._extract_units(query)
        evidence_units = self._extract_units(
            "\n".join(item.text for item in candidate_items)
        )
        checks["requested_units"] = sorted(requested_units)
        checks["evidence_units"] = sorted(evidence_units)
        currency_units = evidence_units.intersection({"USD", "EUR", "GBP"})
        unit_limitations: List[str] = []
        if requested_units and evidence_units and not requested_units.intersection(
            evidence_units
        ):
            return EvidenceAuditResult(
                decision="clarify",
                confidence=0.9,
                reason="The requested and retrieved currency or unit do not match.",
                follow_up=(
                    "Should I keep the values in the reported unit, or convert them "
                    "to your requested unit?"
                ),
                options=["Keep reported units", "Convert units"],
                limitations=["A conversion basis or exchange rate may be required."],
                checks=checks,
            )
        if not requested_units and len(currency_units) > 1:
            requires_common_currency = bool(
                re.search(
                    r"\b(?:compare|comparison|calculate|combine|combined|"
                    r"consolidate|consolidated|aggregate|total across|cagr|"
                    r"growth rate|valuation multiple)\b",
                    query,
                    re.IGNORECASE,
                )
            )
            if requires_common_currency:
                return EvidenceAuditResult(
                    decision="clarify",
                    confidence=0.9,
                    reason=(
                        "The requested comparison or calculation uses figures "
                        "reported in more than one currency."
                    ),
                    follow_up=(
                        "Should I keep each figure in its reported currency, or "
                        "convert them using an exchange rate you provide?"
                    ),
                    options=[
                        "Keep reported currencies",
                        "I will provide an exchange rate",
                    ],
                    limitations=[
                        "Figures in different currencies are not directly comparable."
                    ],
                    checks=checks,
                )
            unit_limitations.append(
                "The retrieved sources use multiple currencies; each figure "
                "must remain in its reported currency unless a conversion "
                "basis is supplied."
            )

        conflicts = self._find_conflicts(candidate_items, metrics, requested_years)
        checks["conflicts"] = conflicts
        if conflicts:
            return EvidenceAuditResult(
                decision="clarify",
                confidence=0.97,
                reason="The retrieved sources report conflicting figures.",
                follow_up=(
                    "The documents contain different values for the requested item. "
                    "Should I show both and compare their source dates?"
                ),
                options=["Show and compare both", "Use the latest source"],
                limitations=[
                    "The conflicting values must not be reconciled without a stated basis."
                ],
                verified_context=self._format_verified_context(candidate_items),
                checks=checks,
            )

        limitations: List[str] = list(unit_limitations)
        if missing_metrics:
            limitations.append(
                "Some requested metrics were not disclosed: "
                + ", ".join(sorted(missing_metrics))
                + "."
            )
        if requested_basis and not evidence_basis:
            limitations.append(
                "The source does not explicitly label the figure as actual or forecast."
            )

        result = EvidenceAuditResult(
            decision="answer",
            confidence=0.94 if not limitations else 0.8,
            reason=(
                "The retrieved evidence matches the requested company, metric, "
                "period, unit, and reporting basis closely enough to answer."
            ),
            limitations=limitations,
            verified_context=self._format_verified_context(candidate_items),
            checks=checks,
        )
        return self._apply_optional_llm_review(
            result,
            query=query,
            items=candidate_items,
            llm_callable=llm_callable or self.llm_callable,
        )

    def _missing(
        self, reason: str, limitation: str, checks: Dict[str, Any]
    ) -> EvidenceAuditResult:
        return EvidenceAuditResult(
            decision="missing",
            confidence=0.98,
            reason=reason,
            limitations=[limitation],
            checks=checks,
        )

    def _coerce_evidence(
        self, context: str, documents: Optional[Sequence[Any]]
    ) -> List[_EvidenceItem]:
        items: List[_EvidenceItem] = []
        for document in documents or []:
            if isinstance(document, Mapping):
                text = str(
                    document.get("page_content")
                    or document.get("content")
                    or document.get("text")
                    or ""
                )
                metadata = document.get("metadata") or document.get(
                    "chunk_metadata"
                ) or {}
            else:
                text = str(
                    getattr(document, "page_content", None)
                    or getattr(document, "content", None)
                    or ""
                )
                metadata = getattr(document, "metadata", None) or getattr(
                    document, "chunk_metadata", None
                ) or {}
            if text.strip():
                items.append(_EvidenceItem(text=text, metadata=metadata))

        if not items and context.strip():
            source_blocks = re.split(r"(?=\[Source:[^\]]+\])", context)
            for block in source_blocks:
                if not block.strip():
                    continue
                source_match = re.match(r"\[Source:\s*([^\]]+)\]", block.strip())
                metadata = (
                    {"source": source_match.group(1).strip()} if source_match else {}
                )
                items.append(_EvidenceItem(text=block.strip(), metadata=metadata))
        return items

    def _extract_metrics(self, text: str) -> Set[str]:
        lowered = text.lower()
        return {
            metric
            for metric, aliases in METRIC_ALIASES.items()
            if any(alias in lowered for alias in aliases)
        }

    def _extract_query_entity(self, query: str) -> Optional[str]:
        lowered = query.lower()
        for canonical, aliases in KNOWN_ENTITIES.items():
            if any(alias in lowered for alias in aliases):
                return canonical
        match = _POSSESSIVE_ENTITY_RE.search(query)
        return match.group(1).strip() if match else None

    def _entity_aligned_items(
        self, items: Sequence[_EvidenceItem], requested_entity: Optional[str]
    ) -> Tuple[List[_EvidenceItem], Set[str]]:
        observed: Set[str] = set()
        aligned: List[_EvidenceItem] = []
        for item in items:
            haystack = self._item_haystack(item)
            for canonical, aliases in KNOWN_ENTITIES.items():
                if any(alias in haystack for alias in aliases):
                    observed.add(canonical)
            if requested_entity and requested_entity.lower() in haystack:
                aligned.append(item)
                continue
            if requested_entity:
                aliases = KNOWN_ENTITIES.get(requested_entity, ())
                if any(alias in haystack for alias in aliases):
                    aligned.append(item)
        return aligned, observed

    def _metric_aligned_items(
        self, items: Sequence[_EvidenceItem], metrics: Set[str]
    ) -> Tuple[List[_EvidenceItem], Set[str]]:
        if not metrics:
            return list(items), set()
        matched: List[_EvidenceItem] = []
        found: Set[str] = set()
        for item in items:
            item_metrics = self._extract_metrics(item.text)
            overlap = metrics.intersection(item_metrics)
            if overlap:
                matched.append(item)
                found.update(overlap)
        return matched, metrics - found

    def _extract_years(self, items: Sequence[_EvidenceItem]) -> Set[str]:
        years: Set[str] = set()
        for item in items:
            years.update(self._year_values(self._item_haystack(item)))
        return years

    def _period_aligned_items(
        self, items: Sequence[_EvidenceItem], requested_years: Set[str]
    ) -> List[_EvidenceItem]:
        if not requested_years:
            return list(items)
        return [
            item
            for item in items
            if requested_years.intersection(
                self._year_values(self._item_haystack(item))
            )
        ]

    def _year_values(self, text: str) -> Set[str]:
        return {match.group(0)[:4] for match in _YEAR_RE.finditer(text)}

    def _extract_basis(self, text: str) -> Set[str]:
        lowered = text.lower()
        return {
            basis
            for basis, aliases in BASIS_ALIASES.items()
            if any(alias in lowered for alias in aliases)
        }

    def _extract_units(self, text: str) -> Set[str]:
        lowered = text.lower()
        units: Set[str] = set()
        if "$" in text or re.search(r"\b(?:usd|us dollars?)\b", lowered):
            units.add("USD")
        if "€" in text or re.search(r"\b(?:eur|euros?)\b", lowered):
            units.add("EUR")
        if "£" in text or re.search(r"\b(?:gbp|pounds?)\b", lowered):
            units.add("GBP")
        if "%" in text or "percent" in lowered:
            units.add("percent")
        if re.search(r"\b(?:bn|billion)\b", lowered):
            units.add("billions")
        if re.search(r"\b(?:mm|million)\b", lowered):
            units.add("millions")
        if re.search(r"\b(?:thousand)\b", lowered):
            units.add("thousands")
        return units

    def _find_conflicts(
        self,
        items: Sequence[_EvidenceItem],
        metrics: Set[str],
        requested_years: Set[str],
    ) -> List[Dict[str, Any]]:
        by_key: Dict[Tuple[str, str, str], Set[float]] = {}
        for item in items:
            for line in item.text.splitlines():
                lowered = line.lower()
                line_metrics = {
                    metric
                    for metric in metrics
                    if any(alias in lowered for alias in METRIC_ALIASES[metric])
                }
                if not line_metrics:
                    continue
                line_years = self._year_values(line)
                # A table row or sentence can contain a complete multi-period
                # series. Without reliable column mapping, assigning every
                # amount on that line to one requested year creates false
                # conflicts. Single-period source lines are safe to compare.
                if len(line_years) > 1:
                    continue
                if requested_years and not requested_years.intersection(line_years):
                    continue
                periods = requested_years.intersection(line_years) or line_years or {""}
                for amount_match in _AMOUNT_RE.finditer(line):
                    amount = self._normalise_amount(amount_match)
                    unit = self._amount_unit(amount_match)
                    for metric in line_metrics:
                        for period in periods:
                            by_key.setdefault((metric, period, unit), set()).add(amount)

        conflicts: List[Dict[str, Any]] = []
        for (metric, period, unit), values in by_key.items():
            if len(values) < 2:
                continue
            ordered = sorted(values)
            if abs(ordered[-1] - ordered[0]) <= max(1e-9, abs(ordered[-1]) * 1e-6):
                continue
            conflicts.append(
                {
                    "metric": metric,
                    "period": period or None,
                    "unit": unit,
                    "values": ordered,
                }
            )
        return conflicts

    def _normalise_amount(self, match: re.Match[str]) -> float:
        value = float(match.group("number").replace(",", ""))
        scale = (match.group("scale") or "").lower()
        if scale in {"k", "thousand"}:
            return value * 1_000
        if scale in {"m", "mm", "million"}:
            return value * 1_000_000
        if scale in {"bn", "billion"}:
            return value * 1_000_000_000
        return value

    def _amount_unit(self, match: re.Match[str]) -> str:
        currency = {"$": "USD", "€": "EUR", "£": "GBP"}.get(
            match.group("currency") or "", ""
        )
        if (match.group("scale") or "") == "%":
            return "percent"
        return currency or "reported"

    def _item_haystack(self, item: _EvidenceItem) -> str:
        metadata_text = " ".join(str(value) for value in item.metadata.values())
        return f"{metadata_text}\n{item.text}".lower()

    def _format_verified_context(self, items: Sequence[_EvidenceItem]) -> str:
        blocks: List[str] = []
        seen: Set[Tuple[str, str]] = set()
        for index, item in enumerate(items, start=1):
            source = (
                item.metadata.get("filename")
                or item.metadata.get("source")
                or item.metadata.get("title")
                or f"retrieved evidence {index}"
            )
            locator_parts = [f"Source: {source}"]
            for key, label in (
                ("page_num", "Page"),
                ("slide_num", "Slide"),
                ("sheet", "Sheet"),
                ("row_range", "Rows"),
            ):
                if item.metadata.get(key) is not None:
                    locator_parts.append(f"{label}: {item.metadata[key]}")
            key = (" | ".join(locator_parts), item.text)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(f"[Verified Evidence | {' | '.join(locator_parts)}]\n{item.text}")
        return "\n\n".join(blocks)

    def _apply_optional_llm_review(
        self,
        result: EvidenceAuditResult,
        query: str,
        items: Sequence[_EvidenceItem],
        llm_callable: Optional[AuditLLMCallable],
    ) -> EvidenceAuditResult:
        if llm_callable is None:
            return result
        payload = {
            "task": (
                "Audit whether the evidence answers the question. Return decision "
                "(answer, clarify, or missing), confidence, reason, follow_up, "
                "options, and limitations. Never invent evidence."
            ),
            "query": query,
            "deterministic_audit": result.to_dict(),
            "evidence": [
                {"text": item.text, "metadata": dict(item.metadata)} for item in items
            ],
        }
        try:
            review = llm_callable(payload)
            if isinstance(review, str):
                review = json.loads(review)
            if not isinstance(review, Mapping):
                return result
            decision = str(review.get("decision", "")).lower()
            if decision not in {"answer", "clarify", "missing"}:
                return result
            if decision == "answer":
                # The LLM may confirm, but never replace the deterministic context.
                result.confidence = min(
                    1.0, max(result.confidence, float(review.get("confidence", 0)))
                )
                if review.get("limitations"):
                    result.limitations.extend(
                        str(value) for value in review["limitations"]
                    )
                return result
            return EvidenceAuditResult(
                decision=decision,
                confidence=float(review.get("confidence", 0.75)),
                reason=str(review.get("reason") or result.reason),
                follow_up=(
                    str(review["follow_up"]) if review.get("follow_up") else None
                ),
                options=[str(value) for value in review.get("options", [])],
                limitations=[
                    str(value) for value in review.get("limitations", [])
                ],
                verified_context=result.verified_context,
                checks={**result.checks, "llm_reviewed": True},
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Evidence-audit LLM returned an invalid response", exc_info=True)
        except Exception:
            logger.exception("Evidence-audit LLM call failed")
        return result


financial_evidence_audit_service = FinancialEvidenceAuditService()
