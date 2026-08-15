import math
from typing import Any, Dict, List


class FinancialPresentationError(ValueError):
    """Raised when a requested financial presentation cannot be built safely."""


class FinancialPresentation:
    """Deterministic Markdown presentation helpers for financial answers."""

    tool_definition = {
        "type": "function",
        "function": {
            "name": "create_financial_chart",
            "description": (
                "Create a compact Markdown text bar chart from verified, "
                "comparable financial values. Use it only when at least three "
                "periods or categories materially improve the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 12,
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 12,
                    },
                    "unit": {"type": "string"},
                },
                "required": ["title", "labels", "values"],
                "additionalProperties": False,
            },
        },
    }

    def create_chart(
        self,
        title: str,
        labels: List[str],
        values: List[float],
        unit: str = "",
    ) -> Dict[str, Any]:
        clean_title = " ".join(str(title).split())
        clean_unit = " ".join(str(unit).split())
        if not clean_title:
            raise FinancialPresentationError("Chart title cannot be empty")
        if not isinstance(labels, list) or not isinstance(values, list):
            raise FinancialPresentationError("labels and values must be arrays")
        if len(labels) != len(values):
            raise FinancialPresentationError(
                "labels and values must contain the same number of items"
            )
        if not 3 <= len(labels) <= 12:
            raise FinancialPresentationError(
                "A chart requires between three and twelve comparable values"
            )

        clean_labels: List[str] = []
        clean_values: List[float] = []
        for label, value in zip(labels, values):
            clean_label = " ".join(str(label).split())
            if not clean_label:
                raise FinancialPresentationError("Chart labels cannot be empty")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FinancialPresentationError("Chart values must be numeric")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise FinancialPresentationError("Chart values must be finite")
            clean_labels.append(clean_label[:40])
            clean_values.append(numeric_value)

        maximum = max(abs(value) for value in clean_values)
        label_width = max(len(label) for label in clean_labels)
        lines = [clean_title]
        for label, value in zip(clean_labels, clean_values):
            bar_width = 1 if maximum == 0 else round(abs(value) / maximum * 24)
            if value == 0:
                bar = "·"
            else:
                bar = ("−" if value < 0 else "") + ("█" * max(1, bar_width))
            formatted_value = f"{value:,.2f}".rstrip("0").rstrip(".")
            suffix = f" {clean_unit}" if clean_unit else ""
            lines.append(
                f"{label:<{label_width}} | {bar} {formatted_value}{suffix}"
            )

        markdown = "```text\n" + "\n".join(lines) + "\n```"
        return {
            "title": clean_title,
            "labels": clean_labels,
            "values": clean_values,
            "unit": clean_unit,
            "markdown": markdown,
        }


financial_presentation = FinancialPresentation()
