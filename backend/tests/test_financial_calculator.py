import math

import pytest

from app.services.financial_analysis.calculator import (
    FinancialCalculationError,
    financial_calculator,
)


def test_cagr():
    result = financial_calculator.calculate(
        "cagr",
        {
            "beginning_value": 1.07780382,
            "ending_value": 68.16070278247962,
            "periods": 5,
        },
    )
    assert result["result"] == pytest.approx(1.2919169065, rel=1e-6)
    assert result["result_type"] == "percentage"


def test_net_debt():
    result = financial_calculator.calculate(
        "net_debt",
        {"total_debt": 1035.4055, "cash": 627.04417},
    )
    assert result["result"] == pytest.approx(408.36133)


def test_margin():
    result = financial_calculator.calculate(
        "margin",
        {"numerator": 6520.196335075091, "denominator": 31717.22171045555},
    )
    assert result["result"] == pytest.approx(0.20557, rel=1e-4)


def test_npv_and_irr():
    npv = financial_calculator.calculate(
        "npv",
        {"discount_rate": 0.10, "cash_flows": [-100, 60, 60]},
    )
    assert npv["result"] == pytest.approx(4.1322314)

    irr = financial_calculator.calculate(
        "irr",
        {"cash_flows": [-100, 60, 60]},
    )
    assert irr["result"] == pytest.approx(0.13066, rel=1e-4)


def test_invalid_denominator():
    with pytest.raises(FinancialCalculationError):
        financial_calculator.calculate(
            "margin",
            {"numerator": 10, "denominator": 0},
        )
