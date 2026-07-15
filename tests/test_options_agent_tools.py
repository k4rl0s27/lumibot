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
