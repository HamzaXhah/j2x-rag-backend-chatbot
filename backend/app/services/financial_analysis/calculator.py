import math
from typing import Any, Dict, List


class FinancialCalculationError(ValueError):
    """Raised when a requested financial calculation is invalid."""


class FinancialCalculator:
    """Deterministic, validated calculations for the financial analyst."""

    tool_definition = {
        "type": "function",
        "function": {
            "name": "calculate_financial_metric",
            "description": (
                "Perform deterministic financial arithmetic. Use this tool for "
                "every derived numeric result instead of calculating mentally. "
                "Inputs must come from the supplied document evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "cagr",
                            "growth_rate",
                            "margin",
                            "percentage",
                            "net_debt",
                            "cash_runway_months",
                            "valuation_multiple",
                            "dilution",
                            "npv",
                            "irr",
                        ],
                    },
                    "values": {
                        "type": "object",
                        "description": (
                            "Named numeric inputs for the selected operation. "
                            "cagr: beginning_value, ending_value, periods; "
                            "growth_rate: previous_value, current_value; "
                            "margin/percentage: numerator, denominator; "
                            "net_debt: total_debt, cash; cash_runway_months: "
                            "cash, monthly_burn; valuation_multiple: valuation, "
                            "financial_metric; dilution: new_shares, "
                            "existing_shares; npv: discount_rate, cash_flows; "
                            "irr: cash_flows."
                        ),
                        "additionalProperties": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            ]
                        },
                    },
                },
                "required": ["operation", "values"],
                "additionalProperties": False,
            },
        },
    }

    def calculate(self, operation: str, values: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "cagr": self._cagr,
            "growth_rate": self._growth_rate,
            "margin": self._margin,
            "percentage": self._percentage,
            "net_debt": self._net_debt,
            "cash_runway_months": self._cash_runway,
            "valuation_multiple": self._valuation_multiple,
            "dilution": self._dilution,
            "npv": self._npv,
            "irr": self._irr,
        }
        if operation not in handlers:
            raise FinancialCalculationError(
                f"Unsupported financial operation: {operation}"
            )

        result, formula = handlers[operation](values)
        if not math.isfinite(result):
            raise FinancialCalculationError("Calculation produced a non-finite result")

        return {
            "operation": operation,
            "inputs": values,
            "result": result,
            "formula": formula,
            "result_type": (
                "percentage"
                if operation in {
                    "cagr",
                    "growth_rate",
                    "margin",
                    "percentage",
                    "dilution",
                    "irr",
                }
                else "number"
            ),
        }

    def _number(self, values: Dict[str, Any], key: str) -> float:
        if key not in values:
            raise FinancialCalculationError(f"Missing input: {key}")
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FinancialCalculationError(f"{key} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise FinancialCalculationError(f"{key} must be finite")
        return value

    def _cash_flows(self, values: Dict[str, Any]) -> List[float]:
        raw = values.get("cash_flows")
        if not isinstance(raw, list) or len(raw) < 2:
            raise FinancialCalculationError(
                "cash_flows must contain at least two numeric values"
            )
        flows = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FinancialCalculationError("cash_flows must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FinancialCalculationError("cash_flows must be finite")
            flows.append(numeric)
        return flows

    def _cagr(self, values: Dict[str, Any]):
        beginning = self._number(values, "beginning_value")
        ending = self._number(values, "ending_value")
        periods = self._number(values, "periods")
        if beginning <= 0 or ending < 0 or periods <= 0:
            raise FinancialCalculationError(
                "CAGR requires beginning_value > 0, ending_value >= 0, periods > 0"
            )
        result = (ending / beginning) ** (1 / periods) - 1
        return result, "(ending_value / beginning_value)^(1 / periods) - 1"

    def _growth_rate(self, values: Dict[str, Any]):
        previous = self._number(values, "previous_value")
        current = self._number(values, "current_value")
        if previous == 0:
            raise FinancialCalculationError(
                "Growth rate is undefined when previous_value is zero"
            )
        return (
            (current - previous) / abs(previous),
            "(current_value - previous_value) / abs(previous_value)",
        )

    def _margin(self, values: Dict[str, Any]):
        numerator = self._number(values, "numerator")
        denominator = self._number(values, "denominator")
        if denominator == 0:
            raise FinancialCalculationError("Margin denominator cannot be zero")
        return numerator / denominator, "numerator / denominator"

    def _percentage(self, values: Dict[str, Any]):
        numerator = self._number(values, "numerator")
        denominator = self._number(values, "denominator")
        if denominator == 0:
            raise FinancialCalculationError("Percentage denominator cannot be zero")
        return numerator / denominator, "numerator / denominator"

    def _net_debt(self, values: Dict[str, Any]):
        total_debt = self._number(values, "total_debt")
        cash = self._number(values, "cash")
        return total_debt - cash, "total_debt - cash"

    def _cash_runway(self, values: Dict[str, Any]):
        cash = self._number(values, "cash")
        monthly_burn = self._number(values, "monthly_burn")
        if cash < 0 or monthly_burn <= 0:
            raise FinancialCalculationError(
                "Runway requires cash >= 0 and monthly_burn > 0"
            )
        return cash / monthly_burn, "cash / monthly_burn"

    def _valuation_multiple(self, values: Dict[str, Any]):
        valuation = self._number(values, "valuation")
        metric = self._number(values, "financial_metric")
        if metric == 0:
            raise FinancialCalculationError(
                "Valuation multiple denominator cannot be zero"
            )
        return valuation / metric, "valuation / financial_metric"

    def _dilution(self, values: Dict[str, Any]):
        new_shares = self._number(values, "new_shares")
        existing_shares = self._number(values, "existing_shares")
        if new_shares < 0 or existing_shares <= 0:
            raise FinancialCalculationError(
                "Dilution requires new_shares >= 0 and existing_shares > 0"
            )
        return (
            new_shares / (existing_shares + new_shares),
            "new_shares / (existing_shares + new_shares)",
        )

    def _npv(self, values: Dict[str, Any]):
        discount_rate = self._number(values, "discount_rate")
        cash_flows = self._cash_flows(values)
        if discount_rate <= -1:
            raise FinancialCalculationError("discount_rate must be greater than -1")
        result = sum(
            cash_flow / ((1 + discount_rate) ** period)
            for period, cash_flow in enumerate(cash_flows)
        )
        return result, "sum(cash_flow[t] / (1 + discount_rate)^t)"

    def _irr(self, values: Dict[str, Any]):
        cash_flows = self._cash_flows(values)
        if not any(value < 0 for value in cash_flows) or not any(
            value > 0 for value in cash_flows
        ):
            raise FinancialCalculationError(
                "IRR requires at least one negative and one positive cash flow"
            )

        def npv(rate: float) -> float:
            return sum(
                cash_flow / ((1 + rate) ** period)
                for period, cash_flow in enumerate(cash_flows)
            )

        low, high = -0.9999, 10.0
        low_value, high_value = npv(low), npv(high)
        if low_value * high_value > 0:
            raise FinancialCalculationError(
                "Unable to bracket a unique IRR between -99.99% and 1000%"
            )

        for _ in range(200):
            midpoint = (low + high) / 2
            midpoint_value = npv(midpoint)
            if abs(midpoint_value) < 1e-10:
                return midpoint, "rate where NPV(cash_flows, rate) = 0"
            if low_value * midpoint_value <= 0:
                high = midpoint
                high_value = midpoint_value
            else:
                low = midpoint
                low_value = midpoint_value

        return (low + high) / 2, "rate where NPV(cash_flows, rate) = 0"


financial_calculator = FinancialCalculator()

