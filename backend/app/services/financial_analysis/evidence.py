import logging
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.models.document import Document as DBDocument, DocumentChunk

logger = logging.getLogger(__name__)


TOPIC_RULES: Dict[str, Dict[str, Sequence[str]]] = {
    "revenue": {
        "triggers": ("revenue", "sales", "arr", "turnover", "top line", "top-line"),
        "labels": (
            "Revenue ($M)",
            "INVOICED SALES",
            "TURNOVER",
            "SW recurring sales",
            "HW sales",
            "Revenue Model split",
        ),
    },
    "profitability": {
        "triggers": (
            "ebitda",
            "profit",
            "loss",
            "margin",
            "gross profit",
            "break-even",
            "breakeven",
        ),
        "labels": (
            "EBITDA",
            "GROSS PROFIT",
            "GROSS MARGIN",
            "OPERATING EXPENSES",
            "Personnel expenses",
            "PROFIT/(LOSS) FOR THE PERIOD",
        ),
    },
    "balance_sheet": {
        "triggers": (
            "balance sheet",
            "asset",
            "equity",
            "debt",
            "cash",
            "borrowings",
            "leverage",
            "liquidity",
        ),
        "labels": (
            "ASSETS,",
            "EQUITY,",
            "TOTAL ASSETS",
            "TOTAL EQUITY",
            "CASH AND SHORT-TERM DEPOSITS",
            "INTEREST-BEARING LOANS AND BORROWINGS",
            "NET DEBT",
            "TOTAL LIABILITIES",
        ),
    },
    "cash_flow": {
        "triggers": (
            "cash flow",
            "free cash flow",
            "runway",
            "burn",
            "working capital",
            "investing",
            "financing",
        ),
        "labels": (
            "NET CASH FLOW FROM OPERATING ACTIVITIES",
            "NET CASH FLOW FROM INVESTING ACTIVITIES",
            "NET CASH FLOW FROM FINANCING ACTIVITIES",
            "CASH AND CASH EQUIVALENTS",
            "CHANGE IN WORKING CAPITAL",
        ),
    },
    "company_profile": {
        "triggers": (
            "company profile",
            "product",
            "service",
            "solution",
            "offering",
            "feature",
            "use case",
            "architecture",
            "technology",
            "target customer",
            "competitive",
            "differentiation",
            "business model",
            "revenue model",
        ),
        "labels": (
            "Solution:",
            "main product",
            "service category",
            "key advantages",
            "comprehensive banking solution",
            "Virtual Branch",
            "target customers",
            "Warehouses",
            "Manufacturing plants",
            "Industrial sites",
            "Full 5G Software Stack",
            "Open Source and O-RAN",
            "Unique competitive advantage",
            "Business model",
        ),
    },
    "market": {
        "triggers": (
            "market",
            "industry",
            "competition",
            "competitor",
            "tam",
            "sam",
            "2030",
        ),
        "labels": (
            "Target Market",
            "Substantial TAM",
            "initial market size",
            "Competitive landscape",
            "Competition",
            "market",
            "2030",
            "frequency dedicated to enterprises",
        ),
    },
    "traction": {
        "triggers": (
            "traction",
            "customer",
            "client",
            "booking",
            "milestone",
            "partner",
            "system integrator",
            "patent",
            "sector",
            "geographic",
        ),
        "labels": (
            "Traction to date",
            "Customers",
            "main customers",
            "Customer traction",
            "Retention and Expansion",
            "pilots converted",
            "Sales have accelerated",
            "Bookings",
            "patent pending",
            "Committed",
            "Partner",
            "System Integrator",
            "Launch",
        ),
    },
    "funding": {
        "triggers": (
            "raise",
            "funding",
            "capital",
            "use of funds",
            "investment terms",
            "valuation",
            "dilution",
        ),
        "labels": (
            "Raising",
            "Raise Details",
            "Amount:",
            "pre-money valuation",
            "Initial close",
            "Use of funds",
            "Milestones",
            "Capital increase",
            "Proceeds from share issues",
        ),
    },
    "management": {
        "triggers": (
            "management",
            "team",
            "founder",
            "ceo",
            "cto",
            "cfo",
            "biography",
            "leadership",
        ),
        "labels": (
            "Team",
            "Experienced team",
            "CEO",
            "President",
            "COO",
            "CRO",
            "CSO",
            "CPO",
            "CBO",
            "CTO",
        ),
    },
}


class FinancialEvidenceService:
    """Adds exact-label evidence without changing semantic RAG retrieval."""

    max_context_chars = 30000
    max_matches_per_label = 2

    def supplement(
        self,
        query: str,
        document_ids: Optional[List[str]],
        retrieved_document_ids: Sequence[str],
        db: Optional[Session],
    ) -> Tuple[str, List[DocumentChunk]]:
        if db is None:
            return "", []

        scoped_ids = list(document_ids or retrieved_document_ids)
        if not scoped_ids:
            scoped_ids = [
                row[0]
                for row in db.query(DBDocument.id)
                .filter(DBDocument.is_indexed.is_(True))
                .order_by(DBDocument.created_at.desc())
                .limit(20)
                .all()
            ]
        if not scoped_ids:
            return "", []
        scoped_ids = self._scope_to_named_company(query, scoped_ids, db)

        labels = self._labels_for_query(query)
        if not labels:
            return "", []

        selected: "OrderedDict[str, DocumentChunk]" = OrderedDict()
        for label in labels:
            matches = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id.in_(scoped_ids),
                    DocumentChunk.content.ilike(f"%{label}%"),
                )
                .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
                .limit(self.max_matches_per_label)
                .all()
            )
            for match in matches:
                self._add_with_neighbors(db, match, selected)
            if sum(len(chunk.content) for chunk in selected.values()) >= self.max_context_chars:
                break

        chunks = list(selected.values())
        return self._format_chunks(chunks, query=query, db=db), chunks

    def _labels_for_query(self, query: str) -> List[str]:
        lowered = query.lower()
        labels: List[str] = []
        for rule in TOPIC_RULES.values():
            if any(trigger in lowered for trigger in rule["triggers"]):
                labels.extend(rule["labels"])

        # Broad investor evaluation should receive the canonical profile.
        if any(
            phrase in lowered
            for phrase in (
                "investment opportunity",
                "evaluate the company",
                "company evaluation",
                "investor-style",
                "investor style",
                "due diligence",
            )
        ):
            for topic in (
                "company_profile",
                "market",
                "traction",
                "revenue",
                "profitability",
                "balance_sheet",
                "cash_flow",
                "funding",
                "management",
            ):
                labels.extend(TOPIC_RULES[topic]["labels"])

        return list(dict.fromkeys(labels))

    def _scope_to_named_company(
        self,
        query: str,
        scoped_ids: List[str],
        db: Session,
    ) -> List[str]:
        """Prevent supplemental evidence from mixing recognized companies."""
        lowered = query.lower()
        aliases: Sequence[str] = ()
        if "firecell" in lowered:
            aliases = ("firecell",)
        elif "agent iq" in lowered or "aiq" in lowered:
            aliases = ("agent iq", "aiq")
        if not aliases:
            return scoped_ids

        documents = (
            db.query(DBDocument)
            .filter(DBDocument.id.in_(scoped_ids))
            .all()
        )
        matching_ids = []
        for document in documents:
            searchable = f"{document.filename} {document.title or ''}".lower()
            if any(alias in searchable for alias in aliases):
                matching_ids.append(document.id)
        return matching_ids or scoped_ids

    def _add_with_neighbors(
        self,
        db: Session,
        match: DocumentChunk,
        selected: "OrderedDict[str, DocumentChunk]",
    ) -> None:
        neighbors = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == match.document_id,
                DocumentChunk.chunk_index.between(
                    max(0, match.chunk_index - 1),
                    match.chunk_index + 1,
                ),
            )
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        for chunk in neighbors:
            if sum(len(item.content) for item in selected.values()) >= self.max_context_chars:
                return
            selected.setdefault(chunk.id, chunk)

    def _format_chunks(
        self,
        chunks: List[DocumentChunk],
        query: str = "",
        db: Optional[Session] = None,
    ) -> str:
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.chunk_metadata or {}
            source = metadata.get("filename") or f"document:{chunk.document_id}"
            locator_parts = [f"Source: {source}"]
            if metadata.get("page_num") is not None:
                locator_parts.append(f"Page: {metadata['page_num']}")
            if metadata.get("slide_num") is not None:
                locator_parts.append(f"Slide: {metadata['slide_num']}")
            if metadata.get("sheet"):
                locator_parts.append(f"Sheet: {metadata['sheet']}")
            if metadata.get("row_range"):
                locator_parts.append(f"Rows: {metadata['row_range']}")
            locator = " | ".join(locator_parts)
            period_note = self._requested_period_note(
                chunk=chunk,
                query=query,
                db=db,
            )
            blocks.append(
                f"[Financial Evidence {index} | {locator}]\n"
                f"{chunk.content}{period_note}"
            )
        return "\n\n".join(blocks)

    def _requested_period_note(
        self,
        chunk: DocumentChunk,
        query: str,
        db: Optional[Session],
    ) -> str:
        """Pair monthly Excel rows with the exact requested year-end column."""
        if db is None:
            return ""
        year_match = re.search(r"\b(20\d{2})\b", query)
        metadata = chunk.chunk_metadata or {}
        if not year_match or metadata.get("format") != "excel":
            return ""

        parts = [part.strip() for part in chunk.content.split(",")]
        if len(parts) < 3:
            return ""
        try:
            values = [float(part) for part in parts[1:] if part]
        except ValueError:
            return ""
        if not values:
            return ""

        header_candidates = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == chunk.document_id,
                DocumentChunk.chunk_index <= chunk.chunk_index,
                DocumentChunk.content.ilike("%Dec 2022%"),
            )
            .order_by(DocumentChunk.chunk_index.desc())
            .limit(4)
            .all()
        )
        month_pattern = re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+20\d{2}\b"
        )
        target_period = f"Dec {year_match.group(1)}"
        for header in header_candidates:
            periods = month_pattern.findall(header.content)
            if target_period not in periods:
                continue
            position = periods.index(target_period)
            if position >= len(values):
                continue
            return (
                f"\nExact period alignment: {target_period} is the "
                f"{position + 1}th numeric value in this row, equal to "
                f"{values[position]}."
            )
        return ""


financial_evidence_service = FinancialEvidenceService()
