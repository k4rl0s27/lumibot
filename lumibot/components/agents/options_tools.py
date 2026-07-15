"""
Options-specific agent tools for LumiBot AI agents.

Provides @agent_tool functions that wrap Tradier options chain data and
portfolio Greeks into compact, LLM-friendly summaries. These tools give
AI agents the ability to:

- Query live options chains with Greeks and liquidity data
- Calculate portfolio-level Greek exposures
- Analyze option strategy risk/reward profiles
- Assess market conditions for options trading

Usage in a strategy:
    from lumibot.components.agents.options_tools import (
        get_option_chain_summary,
        get_portfolio_greeks_summary,
        get_option_strategy_analysis,
        get_options_market_snapshot,
    )

    def initialize(self):
        # Custom tools are auto-discovered when defined at module level
        # and registered via self.agents.create(tools=[...]) or auto-loaded
        ...
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from lumibot.components.agents import agent_tool
from lumibot.entities import Asset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of strikes above/below ATM to include in summaries
_DEFAULT_STRIKE_COUNT = 5

# Maximum spread percentage to consider an option liquid
_MAX_LIQUID_SPREAD_PCT = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Safely convert a value to float, returning default on failure."""
    if value is None:
        return default
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely convert a value to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_iv(row: Any) -> float | None:
    """Extract implied volatility from a chain row, handling column name variations.

    Tradier uses ``greeks.mid_iv``; other brokers may use ``greeks.implied_volatility``.
    """
    for col in ("greeks.mid_iv", "greeks.implied_volatility", "greeks.bid_iv"):
        value = row.get(col) if hasattr(row, "get") else None
        if value is not None:
            result = _safe_float(value)
            if result is not None:
                return result
    return None


def _format_price(value: float | None, decimals: int = 2) -> str:
    """Format a price value for display."""
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def _format_pct(value: float | None, decimals: int = 1) -> str:
    """Format a percentage value for display."""
    if value is None:
        return "N/A"
    return f"{value * 100:+.{decimals}f}%" if abs(value) < 10 else f"{value * 100:+.{decimals}f}%"


def _closest_strikes(strikes: list[float], target: float, count: int = _DEFAULT_STRIKE_COUNT) -> list[float]:
    """Return the `count` strikes closest to the target price."""
    if not strikes:
        return []
    sorted_strikes = sorted(strikes, key=lambda s: abs(s - target))
    return sorted(sorted_strikes[:count])


def _build_options_asset(
    symbol: str,
    expiration: str,
    strike: float,
    right: str,
) -> Asset:
    """Build an Asset object for an option contract."""
    exp_date = datetime.fromisoformat(expiration).date() if isinstance(expiration, str) else expiration
    return Asset(
        symbol=symbol,
        asset_type=Asset.AssetType.OPTION,
        expiration=exp_date,
        strike=strike,
        right=right.lower(),
    )


# ---------------------------------------------------------------------------
# Agent Tools
# ---------------------------------------------------------------------------


@agent_tool(
    name="get_option_chain_summary",
    description=(
        "Get a compact summary of the options chain for a stock symbol. "
        "Returns ATM strikes with key Greeks (delta, gamma, theta, vega), "
        "bid/ask spreads, volume, and open interest for both calls and puts. "
        "Use this to evaluate option liquidity, find strikes, and compare premiums "
        "before constructing a specific strategy."
    ),
)
def get_option_chain_summary(
    self,
    symbol: str,
    expiration: str | None = None,
    strikes_above_below: int = _DEFAULT_STRIKE_COUNT,
) -> dict[str, Any]:
    """Summarize the options chain for a stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g., SPY, QQQ, AAPL).
        expiration: Expiration date in YYYY-MM-DD format. If None, uses the
            nearest monthly expiration (closest to 30 DTE).
        strikes_above_below: Number of strikes above and below ATM to include.
            Default 5 gives ~11 strikes total.

    Returns:
        A dictionary with chain summary data.
    """
    try:
        # Get the underlying price
        stock_asset = Asset(symbol=symbol.upper(), asset_type=Asset.AssetType.STOCK)
        underlying_price = _safe_float(self.get_last_price(stock_asset))

        if underlying_price is None:
            return {
                "ok": False,
                "error": f"Could not get underlying price for {symbol}. Market may be closed.",
            }

        # Get the options chain
        chains = self.get_chains(stock_asset)

        if not chains or "Chains" not in chains:
            return {
                "ok": False,
                "error": f"No options chain data available for {symbol}.",
            }

        # Find the right expiration
        available_expirations = sorted(chains["Chains"]["CALL"].keys())
        if not available_expirations:
            return {
                "ok": False,
                "error": f"No option expirations available for {symbol}.",
            }

        if expiration:
            target_expiry = expiration
        else:
            # Default: nearest expiration with >= 20 DTE (avoid 0DTE for safety)
            today = datetime.now().date()
            target_expiry = None
            for exp in available_expirations:
                try:
                    exp_date = datetime.fromisoformat(exp).date()
                    dte = (exp_date - today).days
                    if dte >= 20:
                        target_expiry = exp
                        break
                except ValueError:
                    continue
            if target_expiry is None:
                target_expiry = available_expirations[0]  # fallback to nearest

        # Get full chain info with Greeks from Tradier (single API call)
        try:
            chain_df = self.get_chain_full_info(
                stock_asset,
                target_expiry,
                chains=chains,
                underlying_price=underlying_price,
            )
        except Exception:
            # Fallback: use the chains dict without Greeks
            chain_df = None

        # Get the strikes around ATM
        call_strikes = chains["Chains"]["CALL"].get(target_expiry, [])
        put_strikes = chains["Chains"]["PUT"].get(target_expiry, [])
        all_strikes = sorted(set(call_strikes) | set(put_strikes))
        atm_strikes = _closest_strikes(all_strikes, underlying_price, strikes_above_below * 2 + 1)

        # Build the summary
        dte = None
        try:
            exp_date = datetime.fromisoformat(target_expiry).date()
            dte = (exp_date - datetime.now().date()).days
        except ValueError:
            pass

        summary = {
            "ok": True,
            "symbol": symbol.upper(),
            "underlying_price": round(underlying_price, 2),
            "expiration": target_expiry,
            "dte": dte,
            "strikes_analyzed": len(atm_strikes),
            "strike_range": [min(atm_strikes), max(atm_strikes)] if atm_strikes else [],
        }

        # Add per-strike data if we have the full chain DataFrame
        strikes_data = []
        if chain_df is not None and not chain_df.empty:
            for strike in atm_strikes:
                strike_rows = chain_df[chain_df["strike"] == strike]
                if strike_rows.empty:
                    continue

                call_row = strike_rows[strike_rows["option_type"].str.upper() == "CALL"]
                put_row = strike_rows[strike_rows["option_type"].str.upper() == "PUT"]

                strike_info = {"strike": round(strike, 2)}

                # Call data
                if not call_row.empty:
                    c = call_row.iloc[0]
                    strike_info["call"] = {
                        "bid": _safe_float(c.get("bid")),
                        "ask": _safe_float(c.get("ask")),
                        "last": _safe_float(c.get("last")),
                        "delta": _safe_float(c.get("greeks.delta")),
                        "gamma": _safe_float(c.get("greeks.gamma")),
                        "theta": _safe_float(c.get("greeks.theta")),
                        "vega": _safe_float(c.get("greeks.vega")),
                        "implied_volatility": _get_iv(c),
                        "volume": _safe_int(c.get("volume"), 0),
                        "open_interest": _safe_int(c.get("open_interest"), 0),
                    }
                else:
                    strike_info["call"] = None

                # Put data
                if not put_row.empty:
                    p = put_row.iloc[0]
                    strike_info["put"] = {
                        "bid": _safe_float(p.get("bid")),
                        "ask": _safe_float(p.get("ask")),
                        "last": _safe_float(p.get("last")),
                        "delta": _safe_float(p.get("greeks.delta")),
                        "gamma": _safe_float(p.get("greeks.gamma")),
                        "theta": _safe_float(p.get("greeks.theta")),
                        "vega": _safe_float(p.get("greeks.vega")),
                        "implied_volatility": _get_iv(p),
                        "volume": _safe_int(p.get("volume"), 0),
                        "open_interest": _safe_int(p.get("open_interest"), 0),
                    }
                else:
                    strike_info["put"] = None

                strikes_data.append(strike_info)
        else:
            # No Greeks available — just list strikes
            for strike in atm_strikes:
                strikes_data.append({"strike": round(strike, 2), "call": None, "put": None})

        summary["strikes"] = strikes_data

        # Add ATM IV as a quick reference
        atm_call_iv = None
        atm_put_iv = None
        for s in strikes_data:
            if s["call"] and s["call"].get("implied_volatility"):
                atm_call_iv = s["call"]["implied_volatility"]
                break
        for s in strikes_data:
            if s["put"] and s["put"].get("implied_volatility"):
                atm_put_iv = s["put"]["implied_volatility"]
                break
        summary["atm_implied_volatility"] = {
            "call": atm_call_iv,
            "put": atm_put_iv,
        }

        return summary

    except Exception as e:
        logger.error(f"Error in get_option_chain_summary for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "error": f"Failed to get options chain for {symbol}: {e}",
        }


@agent_tool(
    name="get_portfolio_greeks_summary",
    description=(
        "Get a summary of the current portfolio's total Greek exposures. "
        "Returns net delta, gamma, theta, and vega across all option positions, "
        "plus total portfolio value and option buying power utilization. "
        "Use this to understand portfolio risk before adding new positions."
    ),
)
def get_portfolio_greeks_summary(self) -> dict[str, Any]:
    """Summarize the total Greek exposure of the current options portfolio.

    Returns:
        A dictionary with portfolio Greek summary and position count.
    """
    try:
        positions = self.get_positions() or []
        cash = _safe_float(self.get_cash(), 0.0)
        portfolio_value = _safe_float(self.get_portfolio_value(), 0.0)

        # Aggregate Greeks across option positions
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        option_positions = 0
        stock_positions = 0
        position_details = []

        for pos in positions:
            asset = getattr(pos, "asset", None)
            if asset is None:
                continue

            qty = _safe_float(getattr(pos, "quantity", 0), 0.0)
            symbol = getattr(asset, "symbol", "?")
            asset_type = str(getattr(asset, "asset_type", ""))

            if "option" not in asset_type.lower():
                # Stock positions contribute delta = qty (1.0 per share)
                total_delta += qty
                stock_positions += 1
                continue

            # Option position — get Greeks
            option_positions += 1
            try:
                greeks = self.get_greeks(asset)
            except Exception:
                greeks = None

            if greeks is None:
                position_details.append({
                    "symbol": symbol,
                    "quantity": qty,
                    "delta": "unknown",
                    "gamma": "unknown",
                    "theta": "unknown",
                    "vega": "unknown",
                })
                continue

            delta = _safe_float(greeks.get("delta"), 0.0)
            gamma = _safe_float(greeks.get("gamma"), 0.0)
            theta = _safe_float(greeks.get("theta"), 0.0)
            vega = _safe_float(greeks.get("vega"), 0.0)

            # Multiply by quantity and contract multiplier (100 shares per contract)
            multiplier = _safe_float(getattr(asset, "multiplier", 100), 100.0)
            position_delta = delta * qty * multiplier
            position_gamma = gamma * qty * multiplier
            position_theta = theta * qty * multiplier
            position_vega = vega * qty * multiplier

            total_delta += position_delta
            total_gamma += position_gamma
            total_theta += position_theta
            total_vega += position_vega

            position_details.append({
                "symbol": symbol,
                "quantity": qty,
                "delta": round(position_delta, 4),
                "gamma": round(position_gamma, 4),
                "theta": round(position_theta, 4),
                "vega": round(position_vega, 4),
            })

        # Calculate utilization
        option_bp_utilization = (
            round((portfolio_value - cash) / portfolio_value * 100, 1)
            if portfolio_value and portfolio_value > 0
            else 0.0
        )

        return {
            "ok": True,
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "buying_power_utilization_pct": option_bp_utilization,
            "stock_positions": stock_positions,
            "option_positions": option_positions,
            "total_delta": round(total_delta, 4),
            "total_gamma": round(total_gamma, 4),
            "total_theta": round(total_theta, 4),
            "total_vega": round(total_vega, 4),
            "position_details": position_details,
            "interpretation": _interpret_portfolio_greeks(
                total_delta, total_gamma, total_theta, total_vega, portfolio_value
            ),
        }

    except Exception as e:
        logger.error(f"Error in get_portfolio_greeks_summary: {e}", exc_info=True)
        return {
            "ok": False,
            "error": f"Failed to get portfolio Greeks: {e}",
        }


def _interpret_portfolio_greeks(
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    portfolio_value: float,
) -> str:
    """Provide a human-readable interpretation of portfolio Greeks."""
    parts = []

    # Delta interpretation
    delta_pct = (delta / portfolio_value * 100) if portfolio_value > 0 else 0
    if abs(delta_pct) < 5:
        parts.append(f"Delta exposure is small ({_format_pct(delta_pct / 100)} of portfolio) — mostly market-neutral.")
    elif delta_pct > 0:
        parts.append(f"Net long bias ({_format_pct(delta_pct / 100)} of portfolio) — benefits from upward moves.")
    else:
        parts.append(f"Net short bias ({_format_pct(delta_pct / 100)} of portfolio) — benefits from downward moves.")

    # Gamma interpretation
    if gamma > 0.01:
        parts.append("Positive gamma — position accelerates in your favor as the underlying moves.")
    elif gamma < -0.01:
        parts.append("Negative gamma — position decelerates; risk accelerates against you on large moves.")

    # Theta interpretation
    if theta > 1:
        parts.append(f"Positive theta (${theta:.1f}/day) — collecting time decay.")
    elif theta < -1:
        parts.append(f"Negative theta (${theta:.1f}/day) — paying time decay.")

    # Vega interpretation
    if abs(vega) > 1:
        direction = "benefits from" if vega > 0 else "hurt by"
        parts.append(f"Vega exposure ({vega:.1f}) — {direction} rising volatility.")

    return " ".join(parts) if parts else "Portfolio Greeks are flat — no significant options exposure."


@agent_tool(
    name="get_option_strategy_analysis",
    description=(
        "Analyze the risk/reward profile of a specific options strategy. "
        "Given a strategy type (vertical_spread, iron_condor, straddle, strangle, "
        "calendar_spread, butterfly, covered_call, cash_secured_put) and the "
        "specific strikes/expiration, returns max profit, max loss, breakevens, "
        "and probability of profit estimates. "
        "Use this to evaluate a trade idea before submitting orders."
    ),
)
def get_option_strategy_analysis(
    self,
    symbol: str,
    strategy_type: str,
    strikes: list[float],
    expiration: str,
    option_type: str = "call",
    leg_types: list[str] | None = None,
    leg_sides: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze an options strategy's risk/reward profile.

    Args:
        symbol: Stock ticker symbol (e.g., SPY, QQQ).
        strategy_type: One of: vertical_spread, iron_condor, straddle, strangle,
            calendar_spread, butterfly, covered_call, cash_secured_put.
        strikes: List of strike prices for the strategy legs, in order.
        expiration: Expiration date in YYYY-MM-DD format.
        option_type: "call" or "put" — default for all legs when leg_types is
            not provided. Ignored if leg_types is supplied.
        leg_types: Per-leg option type ("call" or "put"). Required for mixed
            strategies like iron condors that combine calls and puts.
            Length must match strikes.
        leg_sides: Per-leg trade side ("buy" or "sell", or the full
            "buy_to_open"/"sell_to_open" forms). Used to compute net debit/credit
            correctly. Length must match strikes. If None, all legs are treated
            as buys (debit).

    Returns:
        A dictionary with risk/reward analysis.
    """
    try:
        strategy_type = strategy_type.lower().strip()

        # Validate basic inputs
        if not strikes:
            return {"ok": False, "error": "At least one strike price is required."}

        num_legs = len(strikes)

        # Normalize leg_types: use per-leg list or fall back to option_type for all
        if leg_types is None:
            leg_types = [option_type] * num_legs
        elif len(leg_types) != num_legs:
            return {
                "ok": False,
                "error": f"leg_types length ({len(leg_types)}) must match strikes length ({num_legs}).",
            }

        # Normalize leg_sides: default to all buys if not provided
        if leg_sides is None:
            leg_sides = ["buy"] * num_legs
        elif len(leg_sides) != num_legs:
            return {
                "ok": False,
                "error": f"leg_sides length ({len(leg_sides)}) must match strikes length ({num_legs}).",
            }

        # Get the underlying price
        stock_asset = Asset(symbol=symbol.upper(), asset_type=Asset.AssetType.STOCK)
        underlying_price = _safe_float(self.get_last_price(stock_asset))

        if underlying_price is None:
            return {
                "ok": False,
                "error": f"Could not get underlying price for {symbol}.",
            }

        # Get option prices for each leg, using the correct option type per leg
        leg_prices = []
        leg_greeks = []
        for i, strike in enumerate(strikes):
            leg_opt_type = leg_types[i].lower()
            opt_asset = _build_options_asset(symbol, expiration, strike, leg_opt_type)
            try:
                price = _safe_float(self.get_last_price(opt_asset))
                greeks = self.get_greeks(opt_asset, underlying_price=underlying_price)
            except Exception:
                price = None
                greeks = None

            leg_prices.append(price)
            leg_greeks.append(greeks)

        # Calculate strategy P/L based on type
        result = _analyze_strategy_pl(
            strategy_type=strategy_type,
            strikes=strikes,
            leg_prices=leg_prices,
            leg_greeks=leg_greeks,
            leg_sides=leg_sides,
            leg_types=leg_types,
            underlying_price=underlying_price,
            expiration=expiration,
        )

        # If _analyze_strategy_pl returned an error, propagate it immediately
        if result.get("ok") is False:
            return result

        result["ok"] = True
        result["symbol"] = symbol.upper()
        result["strategy_type"] = strategy_type
        result["underlying_price"] = round(underlying_price, 2)
        result["expiration"] = expiration

        try:
            exp_date = datetime.fromisoformat(expiration).date()
            result["dte"] = (exp_date - datetime.now().date()).days
        except ValueError:
            result["dte"] = None

        return result

    except Exception as e:
        logger.error(f"Error in get_option_strategy_analysis: {e}", exc_info=True)
        return {
            "ok": False,
            "error": f"Failed to analyze strategy: {e}",
        }


def _analyze_strategy_pl(
    strategy_type: str,
    strikes: list[float],
    leg_prices: list[float | None],
    leg_greeks: list[dict | None],
    leg_sides: list[str],
    leg_types: list[str],
    underlying_price: float,
    expiration: str,
) -> dict[str, Any]:
    """Calculate P/L profile for a given options strategy.

    This is a simplified analytical model. For production use, the full
    OptionsHelper should be used for precise multi-leg pricing.

    Args:
        strategy_type: The strategy type identifier.
        strikes: Strike prices per leg (in order).
        leg_prices: Absolute (positive) option mid/last prices per leg.
        leg_greeks: Greeks dicts per leg.
        leg_sides: "buy" or "sell" per leg — used to sign net_cost.
        leg_types: "call" or "put" per leg.
        underlying_price: Current underlying price.
        expiration: Expiration date string.
    """
    multiplier = 100.0

    # Compute signed net cost: sell legs contribute +credit, buy legs contribute -debit
    signed_prices: list[float] = []
    for price, side in zip(leg_prices, leg_sides):
        if price is None:
            continue
        side_lower = side.lower().strip()
        if "sell" in side_lower:
            signed_prices.append(price)   # credit received
        else:
            signed_prices.append(-price)  # debit paid

    known_prices = [abs(p) for p in signed_prices]

    if not known_prices:
        return {
            "max_profit": None,
            "max_loss": None,
            "breakevens": [],
            "probability_of_profit": None,
            "risk_reward_ratio": None,
            "warning": "Could not determine option prices. Market may be closed or strikes are illiquid.",
        }

    # net_cost > 0 = net credit, net_cost < 0 = net debit
    net_cost = sum(signed_prices)

    if strategy_type in ("vertical_spread", "call_spread", "put_spread"):
        if len(strikes) < 2:
            return {"ok": False, "error": "Vertical spread requires exactly 2 strikes."}
        width = abs(strikes[0] - strikes[1])
        if net_cost > 0:
            # Net credit spread
            max_profit = net_cost * multiplier
            max_loss = (width * multiplier) - max_profit
        else:
            # Net debit spread
            max_loss = abs(net_cost) * multiplier
            max_profit = (width * multiplier) - max_loss
        # Breakeven: for calls, short strike + net credit (or long strike + net debit);
        # for puts, short strike - net credit (or long strike - net debit).
        # We use a general formula: the breakeven is the short strike adjusted by the
        # net credit/debit. For debit spreads the short leg is the higher strike (calls)
        # or lower strike (puts). For credit spreads, it's the opposite.
        short_strike = strikes[0] if leg_sides[0].lower().startswith("sell") else strikes[1]
        if net_cost > 0:
            breakeven = short_strike - net_cost if "put" in leg_types[0].lower() else short_strike + net_cost
        else:
            breakeven = short_strike - net_cost if "put" in leg_types[0].lower() else short_strike + net_cost
        breakevens = [round(breakeven, 2)]

    elif strategy_type in ("straddle",):
        if len(strikes) < 1:
            return {"ok": False, "error": "Straddle requires 1 strike."}
        # Long straddle: both legs bought
        total_cost = abs(net_cost) * multiplier
        max_profit = None  # unlimited
        max_loss = total_cost
        breakevens = [
            round(strikes[0] + abs(net_cost), 2),
            round(strikes[0] - abs(net_cost), 2),
        ]

    elif strategy_type in ("strangle",):
        if len(strikes) < 2:
            return {"ok": False, "error": "Strangle requires 2 strikes."}
        total_cost = abs(net_cost) * multiplier
        max_profit = None  # unlimited
        max_loss = total_cost
        breakevens = [
            round(max(strikes) + abs(net_cost), 2),
            round(min(strikes) - abs(net_cost), 2),
        ]

    elif strategy_type in ("iron_condor",):
        if len(strikes) < 4:
            return {"ok": False, "error": "Iron condor requires 4 strikes."}
        sorted_strikes = sorted(strikes)
        # Iron condor: short put at K2, long put at K1 (lower wing),
        # short call at K3, long call at K4 (upper wing).
        # Max loss = max(put_wing_width, call_wing_width) * 100 - net_credit
        put_wing_width = sorted_strikes[1] - sorted_strikes[0]
        call_wing_width = sorted_strikes[3] - sorted_strikes[2]
        max_wing_width = max(put_wing_width, call_wing_width)
        max_profit = abs(net_cost) * multiplier  # net credit received
        max_loss = (max_wing_width * multiplier) - max_profit
        breakevens = [
            round(sorted_strikes[1] - abs(net_cost), 2),   # lower BE: short put - credit
            round(sorted_strikes[2] + abs(net_cost), 2),   # upper BE: short call + credit
        ]

    elif strategy_type in ("butterfly",):
        if len(strikes) < 3:
            return {"ok": False, "error": "Butterfly requires 3 strikes."}
        sorted_strikes = sorted(strikes)
        wing_width = sorted_strikes[2] - sorted_strikes[1]
        max_profit = (wing_width * multiplier) - abs(net_cost) * multiplier
        max_loss = abs(net_cost) * multiplier
        breakevens = [
            round(sorted_strikes[0] + abs(net_cost), 2),
            round(sorted_strikes[2] - abs(net_cost), 2),
        ]

    elif strategy_type in ("covered_call",):
        if len(strikes) < 1:
            return {"ok": False, "error": "Covered call requires 1 strike."}
        # Assume 100 shares purchased at underlying, call sold against them
        stock_cost = underlying_price * multiplier
        premium = abs(net_cost) * multiplier  # net_cost for a sold call is positive
        max_profit = (strikes[0] - underlying_price) * multiplier + premium
        max_loss = stock_cost - premium
        breakevens = [round(underlying_price - abs(net_cost), 2)]

    elif strategy_type in ("cash_secured_put",):
        if len(strikes) < 1:
            return {"ok": False, "error": "Cash-secured put requires 1 strike."}
        premium = abs(net_cost) * multiplier  # net_cost for a sold put is positive
        max_profit = premium
        max_loss = (strikes[0] * multiplier) - premium
        breakevens = [round(strikes[0] - abs(net_cost), 2)]

    else:
        return {
            "ok": False,
            "error": f"Unknown strategy type: {strategy_type}. "
            f"Supported: vertical_spread, iron_condor, straddle, strangle, "
            f"butterfly, covered_call, cash_secured_put.",
        }

    # Calculate probability of profit (simplified: use delta approximation)
    pop = None
    if breakevens and leg_greeks:
        # Approximate POP from delta of the closest ATM leg
        atm_delta = None
        for greeks in leg_greeks:
            if greeks and greeks.get("delta") is not None:
                atm_delta = abs(greeks["delta"])
                break
        if atm_delta is not None:
            pop = round(max(0.0, min(1.0, 1.0 - atm_delta)) * 100, 1)

    # Risk/reward ratio
    risk_reward = None
    if max_profit is not None and max_loss is not None and max_loss > 0:
        risk_reward = round(max_profit / max_loss, 2)

    return {
        "net_debit_credit": round(net_cost, 2),
        "net_cost_total": round(net_cost * 100.0, 2),
        "max_profit": round(max_profit, 2) if max_profit is not None else "unlimited",
        "max_loss": round(max_loss, 2) if max_loss is not None else "unlimited",
        "breakevens": breakevens,
        "probability_of_profit_pct": pop,
        "risk_reward_ratio": risk_reward,
    }


@agent_tool(
    name="get_options_market_snapshot",
    description=(
        "Get a snapshot of current market conditions relevant to options trading. "
        "Returns VIX level, underlying price, implied vs historical volatility "
        "comparison, and flags for earnings risk or unusual activity. "
        "Use this as the first tool to assess whether the current environment "
        "favors selling premium, buying volatility, or staying neutral."
    ),
)
def get_options_market_snapshot(
    self,
    symbol: str,
) -> dict[str, Any]:
    """Get market conditions relevant to options trading for a symbol.

    Args:
        symbol: Stock ticker symbol (e.g., SPY, QQQ, AAPL).

    Returns:
        A dictionary with market snapshot data.
    """
    try:
        symbol = symbol.upper()
        stock_asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
        underlying_price = _safe_float(self.get_last_price(stock_asset))

        if underlying_price is None:
            return {
                "ok": False,
                "error": f"Could not get underlying price for {symbol}. Market may be closed.",
            }

        result: dict[str, Any] = {
            "ok": True,
            "symbol": symbol,
            "underlying_price": round(underlying_price, 2),
            "timestamp": datetime.now().isoformat(),
        }

        # VIX (market fear gauge)
        try:
            vix_asset = Asset(symbol="VIX", asset_type=Asset.AssetType.INDEX)
            vix_price = _safe_float(self.get_last_price(vix_asset))
            if vix_price is not None:
                result["vix"] = round(vix_price, 2)
                if vix_price < 15:
                    result["vix_regime"] = "low — market complacent, favors premium selling"
                elif vix_price < 20:
                    result["vix_regime"] = "normal — balanced environment"
                elif vix_price < 30:
                    result["vix_regime"] = "elevated — cautious, wider spreads expected"
                else:
                    result["vix_regime"] = "high — fear/panic, options expensive, favors premium selling if capital sufficient"
        except Exception:
            result["vix"] = None
            result["vix_regime"] = "unavailable"

        # Get ATM IV from near-term options
        try:
            chains = self.get_chains(stock_asset)
            if chains and "Chains" in chains:
                expirations = sorted(chains["Chains"]["CALL"].keys())
                if expirations:
                    # Nearest expiration with >= 7 DTE
                    today = datetime.now().date()
                    near_exp = None
                    for exp in expirations:
                        try:
                            exp_date = datetime.fromisoformat(exp).date()
                            if (exp_date - today).days >= 7:
                                near_exp = exp
                                break
                        except ValueError:
                            continue
                    if near_exp is None:
                        near_exp = expirations[0]

                    chain_df = self.get_chain_full_info(
                        stock_asset, near_exp, chains=chains, underlying_price=underlying_price
                    )
                    if chain_df is not None and not chain_df.empty:
                        atm_rows = chain_df[
                            (chain_df["strike"] >= underlying_price * 0.95)
                            & (chain_df["strike"] <= underlying_price * 1.05)
                        ]
                        if not atm_rows.empty:
                            call_rows = atm_rows[atm_rows["option_type"].str.upper() == "CALL"]
                            if not call_rows.empty:
                                atm_iv = _get_iv(call_rows.iloc[0])
                                if atm_iv is not None:
                                    result["atm_implied_volatility"] = round(atm_iv, 4)
                                    # Rough IV percentile assessment
                                    if atm_iv < 0.15:
                                        result["iv_assessment"] = "low — options are cheap, consider debit strategies"
                                    elif atm_iv < 0.30:
                                        result["iv_assessment"] = "moderate — balanced for most strategies"
                                    elif atm_iv < 0.50:
                                        result["iv_assessment"] = "high — options expensive, favor credit/selling strategies"
                                    else:
                                        result["iv_assessment"] = "extreme — very expensive, selling premium attractive if risk-managed"
        except Exception:
            result["atm_implied_volatility"] = None
            result["iv_assessment"] = "unavailable"

        # Estimate historical volatility (20-day)
        try:
            bars = self.get_historical_prices(stock_asset, length=21, timestep="day")
            if bars is not None and hasattr(bars, "df"):
                df = bars.df
            elif isinstance(bars, pd.DataFrame):
                df = bars
            else:
                df = None

            if df is not None and not df.empty and "close" in df.columns:
                returns = df["close"].pct_change().dropna()
                if len(returns) >= 5:
                    hv_20 = float(returns.std() * math.sqrt(252))
                    result["historical_volatility_20d"] = round(hv_20, 4)

                    # Compare IV vs HV
                    atm_iv = result.get("atm_implied_volatility")
                    if atm_iv is not None and hv_20 > 0:
                        iv_hv_ratio = atm_iv / hv_20
                        result["iv_hv_ratio"] = round(iv_hv_ratio, 2)
                        if iv_hv_ratio > 1.2:
                            result["iv_hv_signal"] = "IV rich — options expensive relative to recent realized volatility"
                        elif iv_hv_ratio < 0.8:
                            result["iv_hv_signal"] = "IV cheap — options inexpensive relative to recent realized volatility"
                        else:
                            result["iv_hv_signal"] = "IV fairly priced relative to realized volatility"
        except Exception:
            result["historical_volatility_20d"] = None

        # Liquidity check: can we get option quotes?
        try:
            chains = self.get_chains(stock_asset)
            if chains and "Chains" in chains:
                expirations = sorted(chains["Chains"]["CALL"].keys())
                result["available_expirations"] = len(expirations)
                result["options_available"] = len(expirations) > 0
        except Exception:
            result["options_available"] = False

        return result

    except Exception as e:
        logger.error(f"Error in get_options_market_snapshot for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "error": f"Failed to get market snapshot for {symbol}: {e}",
        }
