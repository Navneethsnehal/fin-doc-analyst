"""Deterministic tools the agent calls instead of doing arithmetic in the model."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def sum_currency(amounts: list[str]) -> dict[str, object]:
    """Sum currency amounts. Each amount is rounded half-up to cents, then added."""
    cents = [money(Decimal(str(amount))) for amount in amounts]
    total = money(sum(cents, Decimal("0")))
    return {
        "amounts": [f"{value:.2f}" for value in cents],
        "total": f"{total:.2f}",
    }


def amortized_payment(principal: str, annual_rate: str, months: int) -> dict[str, object]:
    """Standard fully amortizing monthly payment, rounded half-up to cents."""
    principal_value = Decimal(str(principal))
    rate = Decimal(str(annual_rate))
    count = int(months)
    if count <= 0:
        raise ValueError("months must be positive")
    if principal_value < 0 or rate < 0:
        raise ValueError("principal and rate must be non-negative")
    if rate == 0:
        payment = money(principal_value / Decimal(count))
    else:
        monthly = rate / Decimal(12)
        factor = (Decimal(1) + monthly) ** count
        raw = principal_value * monthly * factor / (factor - Decimal(1))
        payment = money(raw)
    return {
        "principal": f"{principal_value:.2f}",
        "annual_rate": str(rate),
        "months": count,
        "monthly_payment": f"{payment:.2f}",
    }
