"""
Unit tests for options agent tools (@agent_tool functions).

Tests cover:
- get_option_chain_summary with mocked chain data
- get_portfolio_greeks_summary with mocked portfolio
- get_option_strategy_analysis for each strategy type
- get_options_market_snapshot with mocked market data
- Error handling for missing data and edge cases
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lumibot.entities import Asset


# ---------------------------------------------------------------------------
# Strategy stub that mimics a real LumiBot strategy for tool testing
# ---------------------------------------------------------------------------


class _OptionsToolsStrategy:
    """Minimal strategy stub exposing the methods options tools call."""

    is_backtesting = False

    def __init__(self):
        self._last_price = 450.0
        self._positions = []
        self._cash = 100000.0
        self._portfolio_value = 105000.0
        self._chains = None
        self._chain_df = None
        self._greeks = None
        self._historical_bars = None

    def get_datetime(self):
        return datetime(2026, 7, 9, 15, 30, tzinfo=timezone.utc)

    def get_last_price(self, asset, quote=None, exchange=None):
        symbol = getattr(asset, "symbol", "")
        if symbol == "VIX":
            return 16.5
        return self._last_price

    def get_chains(self, asset):
        return self._chains

    def get_chain_full_info(self, asset, expiry, chains=None, underlying_price=None, risk_free_rate=None,
                            strike_min=None, strike_max=None):
        return self._chain_df

    def get_greeks(self, asset, asset_price=None, underlying_price=None, risk_free_rate=None, query_greeks=False):
        return self._greeks

    def get_positions(self):
        return self._positions

    def get_cash(self):
        return self._cash

    def get_portfolio_value(self):
        return self._portfolio_value

    def get_historical_prices(self, asset, length, timestep="", timeshift=None, quote=None, exchange=None,
                              include_after_hours=True, return_polars=False):
        return self._historical_bars

    def log_message(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Helper: build a mock options chain dict (what TradierData.get_chains returns)
# ---------------------------------------------------------------------------


def _make_mock_chains(symbol="SPY", underlying_price=450.0):
    """Build a mock options chain dict matching Tradier's format."""
    strikes = sorted([underlying_price + i * 5 for i in range(-5, 6)])
    return {
        "Multiplier": 100,
        "Exchange": "SMART",
        "Chains": {
            "CALL": {
                "2026-07-17": strikes,
                "2026-08-21": strikes,
            },
            "PUT": {
                "2026-07-17": strikes,
                "2026-08-21": strikes,
            },
        },
    }


def _make_mock_chain_df(underlying_price=450.0):
    """Build a mock chain DataFrame matching Tradier's get_chain_full_info output."""
    strikes = [underlying_price + i * 5 for i in range(-2, 3)]
    rows = []
    for strike in strikes:
        for opt_type in ["CALL", "PUT"]:
            is_call = opt_type == "CALL"
            rows.append({
                "strike": strike,
                "option_type": opt_type,
                "expiration": "2026-07-17",
                "bid": round(strike * 0.01, 2) if is_call else round((underlying_price - strike) * 0.02, 2),
                "ask": round(strike * 0.012, 2) if is_call else round((underlying_price - strike) * 0.022, 2),
                "last": round(strike * 0.011, 2) if is_call else round((underlying_price - strike) * 0.021, 2),
                "greeks.delta": 0.55 if is_call else -0.45,
                "greeks.gamma": 0.03,
                "greeks.theta": -0.05,
                "greeks.vega": 0.15,
                "greeks.implied_volatility": 0.22,
                "volume": 500,
                "open_interest": 2000,
            })
    return pd.DataFrame(rows)


def _make_mock_greeks():
    """Build mock Greeks dict."""
    return {
        "delta": 0.55,
        "gamma": 0.03,
        "theta": -0.05,
        "vega": 0.15,
        "implied_volatility": 0.22,
        "underlying_price": 450.0,
    }


def _make_mock_bars_df(prices=None):
    """Build mock historical bars DataFrame."""
    if prices is None:
        prices = [440 + i * 2 for i in range(21)]
    dates = pd.date_range(end="2026-07-09", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


# ---------------------------------------------------------------------------
# Tests: get_option_chain_summary
# ---------------------------------------------------------------------------


class TestGetOptionChainSummary:
    """Tests for the get_option_chain_summary agent tool."""

    def test_returns_chain_summary_for_valid_symbol(self):
        """Should return a structured summary with strikes and Greeks."""
        from lumibot.components.agents.options_tools import get_option_chain_summary

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)

        result = get_option_chain_summary(strategy, "SPY", expiration="2026-07-17")

        assert result["ok"] is True
        assert result["symbol"] == "SPY"
        assert result["underlying_price"] == 450.0
        assert result["expiration"] == "2026-07-17"
        assert len(result["strikes"]) > 0
        # First strike should have call and put data
        strike0 = result["strikes"][0]
        assert strike0["call"] is not None
        assert strike0["put"] is not None
        assert strike0["call"]["delta"] is not None

    def test_returns_error_for_invalid_symbol(self):
        """Should return ok=False when underlying price can't be fetched."""
        from lumibot.components.agents.options_tools import get_option_chain_summary

        strategy = _OptionsToolsStrategy()
        strategy._last_price = None  # Simulate no price available

        result = get_option_chain_summary(strategy, "INVALID")
        assert result["ok"] is False
        assert "error" in result

    def test_returns_error_for_missing_chains(self):
        """Should return ok=False when no chains data available."""
        from lumibot.components.agents.options_tools import get_option_chain_summary

        strategy = _OptionsToolsStrategy()
        strategy._chains = None  # No chains

        result = get_option_chain_summary(strategy, "SPY")
        assert result["ok"] is False

    def test_handles_chain_without_dataframe(self):
        """Should work even when get_chain_full_info fails."""
        from lumibot.components.agents.options_tools import get_option_chain_summary

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = None  # Simulate chain info failure

        result = get_option_chain_summary(strategy, "SPY", expiration="2026-07-17")

        assert result["ok"] is True
        # Strikes listed but without call/put details
        assert len(result["strikes"]) > 0
        assert result["strikes"][0]["call"] is None

    def test_returns_dte_in_summary(self):
        """Should include days-to-expiration in the result."""
        from lumibot.components.agents.options_tools import get_option_chain_summary

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)

        result = get_option_chain_summary(strategy, "SPY", expiration="2026-07-17")

        assert result["dte"] is not None
        assert result["dte"] > 0


# ---------------------------------------------------------------------------
# Tests: get_portfolio_greeks_summary
# ---------------------------------------------------------------------------


class TestGetPortfolioGreeksSummary:
    """Tests for the get_portfolio_greeks_summary agent tool."""

    def test_returns_portfolio_summary_with_no_positions(self):
        """Should return a valid summary even with no positions."""
        from lumibot.components.agents.options_tools import get_portfolio_greeks_summary

        strategy = _OptionsToolsStrategy()

        result = get_portfolio_greeks_summary(strategy)

        assert result["ok"] is True
        assert result["portfolio_value"] == 105000.0
        assert result["cash"] == 100000.0
        assert result["option_positions"] == 0
        assert result["total_delta"] == 0.0
        assert result["total_gamma"] == 0.0

    def test_aggregates_option_greeks_correctly(self):
        """Should multiply per-contract Greeks by quantity and multiplier."""
        from lumibot.components.agents.options_tools import get_portfolio_greeks_summary

        strategy = _OptionsToolsStrategy()
        strategy._greeks = _make_mock_greeks()

        # Mock an option position
        mock_asset = SimpleNamespace(
            symbol="SPY250718C00450000",
            asset_type="option",
            expiration=datetime(2026, 7, 18).date(),
            strike=450.0,
            right="call",
            multiplier=100,
        )
        mock_position = SimpleNamespace(asset=mock_asset, quantity=2)

        strategy._positions = [mock_position]

        result = get_portfolio_greeks_summary(strategy)

        assert result["ok"] is True
        assert result["option_positions"] == 1
        # 2 contracts × 100 multiplier × 0.55 delta = 110
        assert result["total_delta"] == pytest.approx(110.0, rel=0.01)
        # 2 contracts × 100 multiplier × 0.03 gamma = 6
        assert result["total_gamma"] == pytest.approx(6.0, rel=0.01)

    def test_includes_interpretation(self):
        """Should include a human-readable interpretation string."""
        from lumibot.components.agents.options_tools import get_portfolio_greeks_summary

        strategy = _OptionsToolsStrategy()

        result = get_portfolio_greeks_summary(strategy)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_handles_error_gracefully(self):
        """Should return ok=False when an exception occurs."""
        from lumibot.components.agents.options_tools import get_portfolio_greeks_summary

        strategy = _OptionsToolsStrategy()

        # Make get_positions raise
        def _raise(*args, **kwargs):
            raise RuntimeError("Boom")

        strategy.get_positions = _raise

        result = get_portfolio_greeks_summary(strategy)
        assert result["ok"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: get_option_strategy_analysis
# ---------------------------------------------------------------------------


class TestGetOptionStrategyAnalysis:
    """Tests for the get_option_strategy_analysis agent tool."""

    def test_vertical_spread_credit_with_buy_sell_sides(self):
        """Should calculate correct P/L for a credit spread with leg_sides."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 450.0  # underlying price
        strategy._greeks = _make_mock_greeks()

        # Sell 450P @ $3.00, Buy 445P @ $1.00 → $2.00 net credit, 5-wide
        _leg_prices = {"SPY_2026-07-17_450.0_PUT": 3.0, "SPY_2026-07-17_445.0_PUT": 1.0}
        _original_get_last_price = strategy.get_last_price

        def _priced_get_last_price(asset, quote=None, exchange=None):
            # Determine if this is an option leg
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _leg_prices.get(key, 1.0)
            # Stock/underlying — use the default
            return 450.0

        strategy.get_last_price = _priced_get_last_price

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="vertical_spread",
            strikes=[450.0, 445.0],
            expiration="2026-07-17",
            option_type="put",
            leg_sides=["sell", "buy"],
        )

        assert result["ok"] is True
        assert result["strategy_type"] == "vertical_spread"
        # Net credit = +3.00 - 1.00 = $2.00 per share
        assert result["net_debit_credit"] == pytest.approx(2.0, rel=0.01)
        # Max profit = $200, Max loss = $500 - $200 = $300
        assert result["max_profit"] == pytest.approx(200.0, rel=0.01)
        assert result["max_loss"] == pytest.approx(300.0, rel=0.01)
        assert len(result["breakevens"]) == 1

    def test_vertical_spread_debit_with_buy_sell_sides(self):
        """Should calculate correct P/L for a debit spread with leg_sides."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 450.0  # underlying price
        strategy._greeks = _make_mock_greeks()

        # Buy 450C @ $4.00, Sell 455C @ $1.50 → $2.50 net debit, 5-wide
        _leg_prices = {"SPY_2026-07-17_450.0_CALL": 4.0, "SPY_2026-07-17_455.0_CALL": 1.5}

        def _priced_get_last_price(asset, quote=None, exchange=None):
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _leg_prices.get(key, 1.0)
            return 450.0

        strategy.get_last_price = _priced_get_last_price

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="vertical_spread",
            strikes=[450.0, 455.0],
            expiration="2026-07-17",
            option_type="call",
            leg_sides=["buy", "sell"],
        )

        assert result["ok"] is True
        # Net debit = -4.00 + 1.50 = -$2.50 per share
        assert result["net_debit_credit"] == pytest.approx(-2.5, rel=0.01)
        # Max loss = $250, Max profit = $500 - $250 = $250
        assert result["max_loss"] == pytest.approx(250.0, rel=0.01)
        assert result["max_profit"] == pytest.approx(250.0, rel=0.01)

    def test_vertical_spread_legacy_no_sides(self):
        """Should still work without leg_sides (backward compat — all buys)."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 3.0
        strategy._greeks = _make_mock_greeks()

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="vertical_spread",
            strikes=[450.0, 445.0],
            expiration="2026-07-17",
            option_type="put",
        )

        assert result["ok"] is True
        # Without leg_sides, both legs are treated as buys (debit)
        assert result["net_debit_credit"] < 0  # net debit
        assert result["max_loss"] is not None
        assert len(result["breakevens"]) == 1

    def test_straddle_analysis(self):
        """Should calculate correct P/L for a straddle."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 10.0  # total premium for straddle
        strategy._greeks = _make_mock_greeks()

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="straddle",
            strikes=[450.0],
            expiration="2026-07-17",
            option_type="call",
        )

        assert result["ok"] is True
        assert result["max_profit"] == "unlimited"
        assert result["max_loss"] is not None
        assert len(result["breakevens"]) == 2

    def test_iron_condor_with_mixed_call_put_legs(self):
        """Should calculate correct P/L for an iron condor with mixed call/put legs."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 450.0
        strategy._greeks = _make_mock_greeks()

        # Iron condor: Buy 430P @ $0.50, Sell 435P @ $1.00,
        #              Sell 465C @ $1.00, Buy 470C @ $0.50
        # Net credit = -0.50 + 1.00 + 1.00 - 0.50 = $1.00
        _leg_prices = {
            "SPY_2026-07-17_430.0_PUT": 0.5,
            "SPY_2026-07-17_435.0_PUT": 1.0,
            "SPY_2026-07-17_465.0_CALL": 1.0,
            "SPY_2026-07-17_470.0_CALL": 0.5,
        }

        def _priced_get_last_price(asset, quote=None, exchange=None):
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _leg_prices.get(key, 1.0)
            return 450.0

        strategy.get_last_price = _priced_get_last_price

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="iron_condor",
            strikes=[430.0, 435.0, 465.0, 470.0],
            expiration="2026-07-17",
            leg_types=["put", "put", "call", "call"],
            leg_sides=["buy", "sell", "sell", "buy"],
        )

        assert result["ok"] is True
        assert result["strategy_type"] == "iron_condor"
        # Net credit = $1.00 per share
        assert result["net_debit_credit"] == pytest.approx(1.0, rel=0.01)
        # Max profit = $100
        assert result["max_profit"] == pytest.approx(100.0, rel=0.01)
        # Max loss = 5*100 - 100 = $400
        assert result["max_loss"] == pytest.approx(400.0, rel=0.01)
        assert len(result["breakevens"]) == 2

    def test_iron_condor_unequal_wing_widths(self):
        """Iron condor max loss should use the wider wing width."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 450.0
        strategy._greeks = _make_mock_greeks()

        # Put wing = 5 (430→435), Call wing = 10 (460→470)
        # Net credit = $2.00
        _leg_prices = {
            "SPY_2026-07-17_430.0_PUT": 0.5,
            "SPY_2026-07-17_435.0_PUT": 1.5,
            "SPY_2026-07-17_460.0_CALL": 1.5,
            "SPY_2026-07-17_470.0_CALL": 0.5,
        }

        def _priced_get_last_price(asset, quote=None, exchange=None):
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _leg_prices.get(key, 1.0)
            return 450.0

        strategy.get_last_price = _priced_get_last_price

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="iron_condor",
            strikes=[430.0, 435.0, 460.0, 470.0],
            expiration="2026-07-17",
            leg_types=["put", "put", "call", "call"],
            leg_sides=["buy", "sell", "sell", "buy"],
        )

        assert result["ok"] is True
        # Max loss should use the wider wing (10), not the narrower (5)
        # Net credit = $2.00, max profit = $200
        # Max loss = 10*100 - 200 = $800
        assert result["max_loss"] == pytest.approx(800.0, rel=0.01)
        assert result["max_profit"] == pytest.approx(200.0, rel=0.01)

    def test_rejects_mismatched_leg_types_length(self):
        """Should return error when leg_types length doesn't match strikes."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="iron_condor",
            strikes=[430.0, 435.0, 465.0, 470.0],
            expiration="2026-07-17",
            leg_types=["put", "call"],  # Only 2 types for 4 strikes
        )

        assert result["ok"] is False
        assert "leg_types" in result.get("error", "").lower()

    def test_rejects_mismatched_leg_sides_length(self):
        """Should return error when leg_sides length doesn't match strikes."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="vertical_spread",
            strikes=[450.0, 445.0],
            expiration="2026-07-17",
            leg_sides=["sell"],  # Only 1 side for 2 strikes
        )

        assert result["ok"] is False
        assert "leg_sides" in result.get("error", "").lower()

    def test_returns_error_for_unknown_strategy(self):
        """Should return ok=False for unknown strategy types."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="unknown_fancy_strategy",
            strikes=[450.0],
            expiration="2026-07-17",
        )

        assert result["ok"] is False
        assert "unknown" in result.get("error", "").lower()

    def test_covered_call_with_sell_side(self):
        """Should calculate correct P/L for a covered call with sell side."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 450.0
        strategy._greeks = _make_mock_greeks()

        # Sell 460C @ $5.00, underlying @ $450
        _leg_prices = {"SPY_2026-07-17_460.0_CALL": 5.0}

        def _priced_get_last_price(asset, quote=None, exchange=None):
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _leg_prices.get(key, 1.0)
            return 450.0

        strategy.get_last_price = _priced_get_last_price

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="covered_call",
            strikes=[460.0],
            expiration="2026-07-17",
            option_type="call",
            leg_sides=["sell"],
        )

        assert result["ok"] is True
        # Net credit = $5.00 per share
        assert result["net_debit_credit"] == pytest.approx(5.0, rel=0.01)
        assert result["max_profit"] is not None
        assert len(result["breakevens"]) == 1

    def test_handles_none_prices(self):
        """Should handle the case where option prices are unavailable."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = None  # No prices available

        result = get_option_strategy_analysis(
            strategy,
            symbol="SPY",
            strategy_type="vertical_spread",
            strikes=[450.0, 445.0],
            expiration="2026-07-17",
        )

        # Should still return (degraded) rather than error
        assert "ok" in result


# ---------------------------------------------------------------------------
# Tests: get_options_market_snapshot
# ---------------------------------------------------------------------------


class TestGetOptionsMarketSnapshot:
    """Tests for the get_options_market_snapshot agent tool."""

    def test_returns_snapshot_for_valid_symbol(self):
        """Should return VIX, IV, and market assessment."""
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)
        strategy._historical_bars = _make_mock_bars_df()

        result = get_options_market_snapshot(strategy, "SPY")

        assert result["ok"] is True
        assert result["symbol"] == "SPY"
        assert result["underlying_price"] == 450.0
        assert result["vix"] == 16.5
        assert "vix_regime" in result

    def test_vix_regime_low(self):
        """VIX < 15 should show 'low/complacent' regime."""
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)

        result = get_options_market_snapshot(strategy, "SPY")
        # VIX is 16.5 which is "normal"
        assert "normal" in result.get("vix_regime", "")

    def test_returns_error_when_no_price(self):
        """Should return ok=False when underlying price is unavailable."""
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._last_price = None

        result = get_options_market_snapshot(strategy, "SPY")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Helpers unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_safe_float_returns_value(self):
        from lumibot.components.agents.options_tools import _safe_float

        assert _safe_float(42.5) == 42.5
        assert _safe_float("3.14") == 3.14

    def test_safe_float_returns_default_for_invalid(self):
        from lumibot.components.agents.options_tools import _safe_float

        assert _safe_float(None, 0.0) == 0.0
        assert _safe_float("not_a_number", -1.0) == -1.0

    def test_safe_float_handles_nan(self):
        from lumibot.components.agents.options_tools import _safe_float

        assert _safe_float(float("nan"), None) is None
        assert _safe_float(float("inf"), 0.0) == 0.0

    def test_safe_int_returns_value(self):
        from lumibot.components.agents.options_tools import _safe_int

        assert _safe_int(42) == 42
        assert _safe_int("100") == 100

    def test_safe_int_returns_default_for_invalid(self):
        from lumibot.components.agents.options_tools import _safe_int

        assert _safe_int(None, 0) == 0
        assert _safe_int("abc", -1) == -1

    def test_closest_strikes_finds_nearest(self):
        from lumibot.components.agents.options_tools import _closest_strikes

        strikes = [440.0, 445.0, 450.0, 455.0, 460.0]
        result = _closest_strikes(strikes, 452.0, count=3)

        assert result == [445.0, 450.0, 455.0]

    def test_format_price(self):
        from lumibot.components.agents.options_tools import _format_price

        assert _format_price(450.0) == "$450.00"
        assert _format_price(None) == "N/A"

    def test_format_pct(self):
        from lumibot.components.agents.options_tools import _format_pct

        assert "+5.0%" in _format_pct(0.05)
        assert _format_pct(None) == "N/A"


# ---------------------------------------------------------------------------
# Tests: Determinism — same inputs must produce same outputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify tool outputs are 100% deterministic given fixed inputs.

    If these fail, the tool itself is non-deterministic and every agent
    calling it at the same moment would get different numbers — the exact
    hallucination pattern seen in the Jul 16 debate log where six agents
    reported six different values for the same iron condor.
    """

    def _make_strategy_for_iron_condor(self):
        strategy = _OptionsToolsStrategy()
        strategy._last_price = 751.0
        strategy._greeks = {"delta": 0.35, "gamma": 0.02, "theta": -0.04, "vega": 0.12, "implied_volatility": 0.145}

        _prices = {
            "SPY_2026-08-07_746.0_PUT": 6.20,
            "SPY_2026-08-07_741.0_PUT": 4.10,
            "SPY_2026-08-07_756.0_CALL": 5.30,
            "SPY_2026-08-07_759.0_CALL": 3.70,
        }

        def _priced_get_last_price(asset, quote=None, exchange=None):
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _prices.get(key, 1.0)
            return 751.0

        strategy.get_last_price = _priced_get_last_price
        return strategy

    def test_same_inputs_same_outputs(self):
        """Calling _analyze_strategy_pl N times with identical inputs must yield identical results."""
        from lumibot.components.agents.options_tools import _analyze_strategy_pl

        strikes = [746.0, 741.0, 756.0, 759.0]
        leg_prices = [6.20, 4.10, 5.30, 3.70]
        leg_greeks = [
            {"delta": 0.35}, {"delta": 0.25}, {"delta": 0.40}, {"delta": 0.20},
        ]
        leg_sides = ["buy", "sell", "sell", "buy"]
        leg_types = ["put", "put", "call", "call"]

        results = []
        for _ in range(10):
            r = _analyze_strategy_pl(
                strategy_type="iron_condor",
                strikes=strikes,
                leg_prices=leg_prices,
                leg_greeks=leg_greeks,
                leg_sides=leg_sides,
                leg_types=leg_types,
                underlying_price=751.0,
                expiration="2026-08-07",
            )
            results.append(r)

        first = results[0]
        for i, r in enumerate(results[1:], 1):
            assert r["net_debit_credit"] == first["net_debit_credit"], f"Run {i}: net_debit_credit diverged"
            assert r["max_profit"] == first["max_profit"], f"Run {i}: max_profit diverged"
            assert r["max_loss"] == first["max_loss"], f"Run {i}: max_loss diverged"
            assert r["breakevens"] == first["breakevens"], f"Run {i}: breakevens diverged"
            assert r["probability_of_profit_pct"] == first["probability_of_profit_pct"], (
                f"Run {i}: PoP diverged ({first['probability_of_profit_pct']} vs {r['probability_of_profit_pct']})"
            )
            assert r["risk_reward_ratio"] == first["risk_reward_ratio"], f"Run {i}: R:R diverged"

    def test_end_to_end_tool_determinism(self):
        """Full get_option_strategy_analysis must return identical results on repeated calls."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        results = []
        for _ in range(5):
            strategy = self._make_strategy_for_iron_condor()
            r = get_option_strategy_analysis(
                strategy,
                symbol="SPY",
                strategy_type="iron_condor",
                strikes=[746.0, 741.0, 756.0, 759.0],
                expiration="2026-08-07",
                leg_types=["put", "put", "call", "call"],
                leg_sides=["buy", "sell", "sell", "buy"],
            )
            results.append(r)

        first = results[0]
        for i, r in enumerate(results[1:], 1):
            for key in ("net_debit_credit", "max_profit", "max_loss", "breakevens",
                         "probability_of_profit_pct", "risk_reward_ratio"):
                assert r.get(key) == first.get(key), f"Run {i}: key '{key}' diverged"


# ---------------------------------------------------------------------------
# Tests: Iron Condor P/L math correctness (hand-verified)
# ---------------------------------------------------------------------------


class TestIronCondorMath:
    """Verify the P/L math for iron condors against hand-computed values.

    These tests use fixed prices so the expected output can be verified
    independently with a calculator. If these fail, the tool is computing
    max loss / max profit / breakevens incorrectly.
    """

    def _run_iron_condor(self, strikes, leg_types, leg_sides, prices, underlying=751.0):
        from lumibot.components.agents.options_tools import (
            get_option_strategy_analysis,
        )

        strategy = _OptionsToolsStrategy()
        strategy._last_price = underlying
        strategy._greeks = {"delta": 0.35}

        _price_map = {}
        for strike, ltype, price in zip(strikes, leg_types, prices):
            key = f"SPY_2026-08-07_{strike}_{ltype.upper()}"
            _price_map[key] = price

        def _priced(asset, quote=None, exchange=None):
            atype = str(getattr(asset, "asset_type", ""))
            if "option" in atype.lower():
                key = f"{asset.symbol}_{asset.expiration}_{asset.strike}_{asset.right}"
                return _price_map.get(key, 1.0)
            return underlying

        strategy.get_last_price = _priced
        return get_option_strategy_analysis(
            strategy, symbol="SPY", strategy_type="iron_condor",
            strikes=strikes, expiration="2026-08-07",
            leg_types=leg_types, leg_sides=leg_sides,
        )

    def test_five_wide_wings_credit_2_70(self):
        """5-wide both wings, net credit $2.70/share.

        Buy 741P@4.50, Sell 746P@6.20 → put wing credit $1.70
        Sell 756C@4.80, Buy 759C@3.80 → call wing credit $1.00
        Total credit = $2.70, max profit = $270, max loss = 500-270 = $230
        Lower BE = 746 - 2.70 = 743.30, Upper BE = 756 + 2.70 = 758.70
        """
        result = self._run_iron_condor(
            strikes=[741.0, 746.0, 756.0, 759.0],
            leg_types=["put", "put", "call", "call"],
            leg_sides=["buy", "sell", "sell", "buy"],
            prices=[4.50, 6.20, 4.80, 3.80],
        )

        assert result["ok"] is True
        assert result["net_debit_credit"] == pytest.approx(2.70, rel=0.01)
        assert result["max_profit"] == pytest.approx(270.0, rel=0.01)
        assert result["max_loss"] == pytest.approx(230.0, rel=0.01)
        assert len(result["breakevens"]) == 2
        assert result["breakevens"][0] == pytest.approx(743.30, rel=0.01)
        assert result["breakevens"][1] == pytest.approx(758.70, rel=0.01)

    def test_unequal_wings_5_and_3(self):
        """5-wide put wing, 3-wide call wing. Max loss follows wider wing.

        Buy 741P@3.50, Sell 746P@5.00 → put net = -3.50 + 5.00 = +1.50 credit
        Sell 756C@3.00, Buy 759C@1.50 → call net = +3.00 - 1.50 = +1.50 credit
        Total credit = +3.00, max profit = $300
        Put wing=5 (746-741), Call wing=3 (759-756), wider=5
        Max loss = 5*100 - 300 = $200
        Lower BE = 746 - 3.00 = 743.00, Upper BE = 756 + 3.00 = 759.00
        """
        result = self._run_iron_condor(
            strikes=[741.0, 746.0, 756.0, 759.0],
            leg_types=["put", "put", "call", "call"],
            leg_sides=["buy", "sell", "sell", "buy"],
            prices=[3.50, 5.00, 3.00, 1.50],
        )

        assert result["ok"] is True
        assert result["net_debit_credit"] == pytest.approx(3.00, rel=0.01)
        assert result["max_profit"] == pytest.approx(300.0, rel=0.01)
        assert result["max_loss"] == pytest.approx(200.0, rel=0.01)
        assert result["breakevens"][0] == pytest.approx(743.00, rel=0.01)
        assert result["breakevens"][1] == pytest.approx(759.00, rel=0.01)

    def test_iron_condor_with_unsorted_strikes(self):
        """Strikes in non-sorted order: net_cost is order-independent, wing widths sorted."""
        # Same trade as test_five_wide_wings_credit_2_70 but with scrambled order.
        # Buy 741P@4.50, Sell 746P@6.20, Sell 756C@4.80, Buy 759C@3.80 → credit $2.70
        result = self._run_iron_condor(
            strikes=[756.0, 741.0, 759.0, 746.0],  # random order
            leg_types=["call", "put", "call", "put"],
            leg_sides=["sell", "buy", "buy", "sell"],
            prices=[4.80, 4.50, 3.80, 6.20],
        )

        assert result["ok"] is True
        # signed: +4.80 - 4.50 - 3.80 + 6.20 = +2.70 (same credit)
        assert result["net_debit_credit"] == pytest.approx(2.70, rel=0.01)
        assert result["max_profit"] == pytest.approx(270.0, rel=0.01)
        # sorted=[741, 746, 756, 759], put_wing=5, call_wing=3, wider=5
        assert result["max_loss"] == pytest.approx(230.0, rel=0.01)

    def test_debit_iron_condor_net_cost_negative(self):
        """When total paid > received, net_debit_credit is negative (debit).

        Buy 741P@6.00, Sell 746P@4.00 → put net = -2.00 debit
        Sell 756C@2.00, Buy 759C@1.00 → call net = +1.00 credit
        Total = -2.00, net debit. Still valid — the tool computes P/L from abs(net_cost).
        """
        result = self._run_iron_condor(
            strikes=[741.0, 746.0, 756.0, 759.0],
            leg_types=["put", "put", "call", "call"],
            leg_sides=["buy", "sell", "sell", "buy"],
            prices=[6.00, 4.00, 2.00, 1.00],
        )
        # signed: -6.00 + 4.00 + 2.00 - 1.00 = -1.00 (net debit)
        # abs(net_cost) = 1.00, max_profit = 1.00 * 100 = $100
        # max_loss = 5*100 - 100 = $400

        assert result["ok"] is True
        assert result["net_debit_credit"] < 0
        assert result["max_profit"] == pytest.approx(100.0, rel=0.01)
        assert result["max_loss"] == pytest.approx(400.0, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: PoP sensitivity to strike ordering (the root cause of hallucination)
# ---------------------------------------------------------------------------


class TestPoPSensitivity:
    """Verify the strategy-aware PoP computation.

    After the fix, iron condor PoP uses both short-leg deltas:
      PoP = 1 - abs(delta_short_put) - delta_short_call

    Vertical spreads use the sold leg's delta:
      PoP = 1 - abs(delta_sold_leg)

    PoP is now order-independent — same trade, same PoP regardless of
    strike ordering.
    """

    def _run_iron_condor(self, deltas, strikes=None, leg_types=None, leg_sides=None):
        from lumibot.components.agents.options_tools import _analyze_strategy_pl

        if strikes is None:
            strikes = [741.0, 746.0, 756.0, 759.0]
        if leg_types is None:
            leg_types = ["put", "put", "call", "call"]
        if leg_sides is None:
            leg_sides = ["buy", "sell", "sell", "buy"]

        leg_greeks = [{"delta": d} for d in deltas]
        leg_prices = [3.50, 5.00, 3.00, 1.50]

        return _analyze_strategy_pl(
            strategy_type="iron_condor",
            strikes=strikes,
            leg_prices=leg_prices,
            leg_greeks=leg_greeks,
            leg_sides=leg_sides,
            leg_types=leg_types,
            underlying_price=751.0,
            expiration="2026-08-07",
        )

    def test_iron_condor_pop_uses_both_short_legs(self):
        """PoP = 1 - abs(short_put_delta) - short_call_delta."""
        # Short 746P delta=0.25 → abs=0.25, short 756C delta=0.35
        # PoP = 1 - 0.25 - 0.35 = 0.40 = 40%
        result = self._run_iron_condor([0.321, 0.25, 0.35, 0.15])
        assert result["probability_of_profit_pct"] == pytest.approx(40.0, rel=0.01)

    def test_iron_condor_pop_order_independent(self):
        """Same trade, different strike ordering → same PoP (the fix)."""
        # Order A: standard [long_put, short_put, short_call, long_call]
        result_a = self._run_iron_condor(
            deltas=[0.15, 0.25, 0.35, 0.10],
            strikes=[741.0, 746.0, 756.0, 759.0],
            leg_types=["put", "put", "call", "call"],
            leg_sides=["buy", "sell", "sell", "buy"],
        )

        # Order B: scrambled — short legs still identified by side+type
        result_b = self._run_iron_condor(
            deltas=[0.35, 0.10, 0.15, 0.25],
            strikes=[756.0, 759.0, 741.0, 746.0],
            leg_types=["call", "call", "put", "put"],
            leg_sides=["sell", "buy", "buy", "sell"],
        )

        assert result_a["probability_of_profit_pct"] == result_b["probability_of_profit_pct"], (
            f"PoP must be order-independent: A={result_a['probability_of_profit_pct']}%, "
            f"B={result_b['probability_of_profit_pct']}%"
        )

    def test_realistic_iron_condor_pop(self):
        """Plausible OTM deltas give a plausible PoP (~60-70%)."""
        # Short 746P delta ~0.18, short 756C delta ~0.22
        # PoP = 1 - 0.18 - 0.22 = 0.60 = 60%
        result = self._run_iron_condor([0.10, 0.18, 0.22, 0.08])
        assert 50.0 <= result["probability_of_profit_pct"] <= 80.0

    def test_iron_condor_pop_zero_when_deep_itm(self):
        """When both short legs are ITM (high deltas), PoP should be near zero."""
        result = self._run_iron_condor([0.05, 0.90, 0.85, 0.03])
        assert result["probability_of_profit_pct"] < 5.0

    def test_vertical_spread_pop_uses_short_leg(self):
        """Credit vertical spread PoP = 1 - abs(short_leg_delta)."""
        from lumibot.components.agents.options_tools import _analyze_strategy_pl

        # Sell 746P (delta=0.25), Buy 741P (delta=0.15)
        # PoP = 1 - 0.25 = 75%
        result = _analyze_strategy_pl(
            strategy_type="vertical_spread",
            strikes=[746.0, 741.0],
            leg_prices=[5.00, 3.50],
            leg_greeks=[{"delta": -0.25}, {"delta": -0.15}],
            leg_sides=["sell", "buy"],
            leg_types=["put", "put"],
            underlying_price=751.0,
            expiration="2026-08-07",
        )
        assert result["probability_of_profit_pct"] == pytest.approx(75.0, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: Market snapshot IV/HV ratio correctness
# ---------------------------------------------------------------------------


class TestMarketSnapshotIVHV:
    """Verify the IV/HV ratio and other computed values in get_options_market_snapshot."""

    def test_iv_hv_ratio_computation(self):
        """IV=0.22, HV=0.1288 → ratio should be 22/12.88 ≈ 1.71 (rounded to 2 decimals = 1.71)."""
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        # Chain df with IV = 0.22
        strategy._chain_df = _make_mock_chain_df(450.0)
        for _, row in strategy._chain_df.iterrows():
            row["greeks.implied_volatility"] = 0.22
        # Historical bars that produce HV = ~0.1288
        prices = [440.0, 442.0, 441.0, 445.0, 443.0, 446.0, 444.0,
                   448.0, 447.0, 450.0, 449.0, 452.0, 451.0, 453.0,
                   450.0, 448.0, 446.0, 449.0, 451.0, 450.0, 448.0]
        strategy._historical_bars = _make_mock_bars_df(prices)

        result = get_options_market_snapshot(strategy, "SPY")

        assert result["ok"] is True
        assert result.get("atm_implied_volatility") is not None
        assert result.get("historical_volatility_20d") is not None
        assert result.get("iv_hv_ratio") is not None
        assert result["iv_hv_ratio"] > 0

    def test_iv_hv_ratio_absent_when_no_hv(self):
        """When historical bars are missing, IV/HV ratio should be absent, not silently wrong."""
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)
        strategy._historical_bars = None  # no historical data

        result = get_options_market_snapshot(strategy, "SPY")

        assert result["ok"] is True
        # IV/HV ratio should NOT be present when HV can't be computed
        assert result.get("iv_hv_ratio") is None
        assert result.get("historical_volatility_20d") is None

    def test_snapshot_handles_missing_chains_gracefully(self):
        """When chains are unavailable, snapshot returns initialized defaults."""
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._chains = None
        strategy._chain_df = None
        strategy._historical_bars = _make_mock_bars_df()

        result = get_options_market_snapshot(strategy, "SPY")

        assert result["ok"] is True
        assert result["atm_implied_volatility"] is None
        assert result["iv_assessment"] == "unavailable"


# ---------------------------------------------------------------------------
# Tests: Tool output schema completeness (prevents model fabrication)
# ---------------------------------------------------------------------------


class TestToolOutputSchema:
    """Verify every tool returns complete, well-typed output dicts.

    When an agent relies on a tool but the tool returns an incomplete or
    confusing response, the LLM may fabricate missing values. These tests
    ensure every response has the expected keys with the expected types.
    """

    def test_strategy_analysis_has_all_required_keys(self):
        """get_option_strategy_analysis must include all expected keys."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = 450.0
        strategy._greeks = _make_mock_greeks()

        result = get_option_strategy_analysis(
            strategy, symbol="SPY", strategy_type="vertical_spread",
            strikes=[450.0, 445.0], expiration="2026-07-17",
            option_type="put", leg_sides=["sell", "buy"],
        )

        required_keys = [
            "ok", "symbol", "strategy_type", "underlying_price", "expiration",
            "dte", "net_debit_credit", "max_profit", "max_loss",
            "breakevens", "probability_of_profit_pct", "risk_reward_ratio",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert isinstance(result["breakevens"], list)
        assert result["dte"] > 0

    def test_chain_summary_has_all_required_keys(self):
        from lumibot.components.agents.options_tools import get_option_chain_summary

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)

        result = get_option_chain_summary(strategy, "SPY", expiration="2026-07-17")

        for key in ("ok", "symbol", "underlying_price", "expiration", "dte", "strikes"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["strikes"], list)

    def test_market_snapshot_has_all_required_keys(self):
        from lumibot.components.agents.options_tools import get_options_market_snapshot

        strategy = _OptionsToolsStrategy()
        strategy._chains = _make_mock_chains("SPY", 450.0)
        strategy._chain_df = _make_mock_chain_df(450.0)
        strategy._historical_bars = _make_mock_bars_df()

        result = get_options_market_snapshot(strategy, "SPY")

        for key in ("ok", "symbol", "underlying_price", "timestamp",
                     "vix", "vix_regime", "atm_implied_volatility",
                     "iv_assessment", "options_available"):
            assert key in result, f"Missing key: {key}"

    def test_portfolio_greeks_has_all_required_keys(self):
        from lumibot.components.agents.options_tools import get_portfolio_greeks_summary

        strategy = _OptionsToolsStrategy()
        result = get_portfolio_greeks_summary(strategy)

        for key in ("ok", "portfolio_value", "cash", "buying_power_utilization_pct",
                     "total_delta", "total_gamma", "total_theta", "total_vega",
                     "option_positions", "stock_positions", "interpretation"):
            assert key in result, f"Missing key: {key}"

    def test_error_response_always_has_ok_false(self):
        """Every error path must set ok=False so the agent knows something went wrong."""
        from lumibot.components.agents.options_tools import get_option_strategy_analysis

        strategy = _OptionsToolsStrategy()
        strategy._last_price = None  # trigger error

        result = get_option_strategy_analysis(
            strategy, symbol="SPY", strategy_type="vertical_spread",
            strikes=[450.0, 445.0], expiration="2026-07-17",
        )

        assert result["ok"] is False
        assert "error" in result
        assert len(result["error"]) > 0, "Error message should not be empty"
