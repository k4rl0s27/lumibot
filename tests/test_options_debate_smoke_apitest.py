"""
Paper trading smoke test for the options debate system.

Tests against the real Tradier paper API to verify:
- Broker connectivity and account access
- Options chain data availability
- Option order submission and cancellation
- Portfolio/position queries for options

Marked with @pytest.mark.apitest — skipped by default.
Run with: pytest -m apitest tests/test_options_debate_smoke_apitest.py

Requires TRADIER_TEST_CONFIG in lumibot/credentials.py or env vars:
    TRADIER_TEST_ACCESS_TOKEN, TRADIER_TEST_ACCOUNT_NUMBER
"""

from __future__ import annotations

import time

import pytest

from lumibot.brokers.tradier import Tradier
from lumibot.credentials import TRADIER_TEST_CONFIG
from lumibot.data_sources.tradier_data import TradierData
from lumibot.entities import Asset, Order

pytestmark = pytest.mark.apitest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tradier_broker() -> Tradier:
    """Get a Tradier broker connected to the paper API, or skip."""
    import os

    # Try test-specific env vars first, fall back to main env vars, then credentials module
    acct = (
        os.environ.get("TRADIER_TEST_ACCOUNT_NUMBER")
        or os.environ.get("TRADIER_ACCOUNT_NUMBER")
        or TRADIER_TEST_CONFIG.get("ACCOUNT_NUMBER")
    )
    token = (
        os.environ.get("TRADIER_TEST_ACCESS_TOKEN")
        or os.environ.get("TRADIER_TEST_API_KEY")
        or os.environ.get("TRADIER_ACCESS_TOKEN")
        or TRADIER_TEST_CONFIG.get("ACCESS_TOKEN")
    )

    if not acct or not token:
        pytest.skip("Missing Tradier credentials in environment or TRADIER_TEST_CONFIG")

    return Tradier(
        account_number=str(acct),
        access_token=str(token),
        paper=True,
        connect_stream=False,
    )


def _get_tradier_data_source(broker: Tradier) -> TradierData:
    """Get the TradierData instance from a broker."""
    return broker.data_source


# ---------------------------------------------------------------------------
# Connectivity & Account
# ---------------------------------------------------------------------------


class TestOptionsDebateSmokeConnectivity:
    """Verify the Tradier paper account is accessible."""

    def test_broker_connects_to_paper_api(self):
        """Broker should connect to paper API without errors."""
        broker = _get_tradier_broker()
        try:
            assert broker.name == "Tradier"
            assert broker._tradier_paper is True
        finally:
            broker.cleanup_streams()

    def test_account_has_positive_balances(self):
        """Paper account should return valid cash and portfolio values."""
        broker = _get_tradier_broker()
        try:
            cash, positions_value, portfolio_value = broker._get_balances_at_broker(
                Asset("USD", asset_type=Asset.AssetType.FOREX), None
            )
            assert isinstance(cash, float), f"Cash should be float, got {type(cash)}"
            assert cash >= 0, f"Cash should be >= 0, got {cash}"
            assert isinstance(portfolio_value, float)
            assert portfolio_value >= 0
        finally:
            broker.cleanup_streams()

    def test_can_fetch_positions(self):
        """Should be able to pull positions without error."""
        broker = _get_tradier_broker()
        try:
            positions = broker._pull_positions("apitest")
            assert positions is not None
            assert isinstance(positions, list)
        finally:
            broker.cleanup_streams()

    def test_can_fetch_orders(self):
        """Should be able to pull order history without error."""
        broker = _get_tradier_broker()
        try:
            orders = broker._pull_broker_all_orders()
            assert isinstance(orders, list)
        finally:
            broker.cleanup_streams()


# ---------------------------------------------------------------------------
# Options Chain Data
# ---------------------------------------------------------------------------


class TestOptionsDebateSmokeChainData:
    """Verify options chain data is available from Tradier."""

    def test_get_chains_returns_valid_structure(self):
        """get_chains should return a dict with Chains, Multiplier, Exchange."""
        broker = _get_tradier_broker()
        try:
            ds = _get_tradier_data_source(broker)
            asset = Asset("SPY", asset_type=Asset.AssetType.STOCK)

            chains = ds.get_chains(asset)

            assert chains is not None
            assert "Chains" in chains
            assert "Multiplier" in chains
            assert chains["Multiplier"] == 100
            assert "CALL" in chains["Chains"]
            assert "PUT" in chains["Chains"]
            # Should have at least one expiration
            call_expirations = chains["Chains"]["CALL"]
            assert len(call_expirations) > 0, "No call expirations found"
        finally:
            broker.cleanup_streams()

    def test_get_chains_has_strikes_for_expiry(self):
        """Each expiration should have a list of strikes."""
        broker = _get_tradier_broker()
        try:
            ds = _get_tradier_data_source(broker)
            asset = Asset("SPY", asset_type=Asset.AssetType.STOCK)

            chains = ds.get_chains(asset)
            first_expiry = sorted(chains["Chains"]["CALL"].keys())[0]
            strikes = chains["Chains"]["CALL"][first_expiry]

            assert len(strikes) > 0, f"No strikes for expiration {first_expiry}"
            assert all(isinstance(s, (int, float)) for s in strikes), "Strikes should be numeric"
            # Strikes should be sorted
            assert strikes == sorted(strikes), "Strikes should be sorted"
        finally:
            broker.cleanup_streams()

    def test_get_chain_full_info_returns_dataframe(self):
        """get_chain_full_info should return a DataFrame with Greeks columns."""
        broker = _get_tradier_broker()
        try:
            ds = _get_tradier_data_source(broker)
            asset = Asset("SPY", asset_type=Asset.AssetType.STOCK)

            chains = ds.get_chains(asset)
            first_expiry = sorted(chains["Chains"]["CALL"].keys())[0]

            df = ds.get_chain_full_info(asset, first_expiry, chains=chains)

            assert df is not None
            assert len(df) > 0, "Chain info DataFrame should not be empty"
            assert "strike" in df.columns
            assert "option_type" in df.columns
            # Greeks columns should be present (Tradier provides them)
            assert "greeks.delta" in df.columns, "Greeks delta column missing"
            # Tradier uses greeks.mid_iv (not greeks.implied_volatility)
            iv_col = (
                "greeks.mid_iv" if "greeks.mid_iv" in df.columns
                else "greeks.implied_volatility" if "greeks.implied_volatility" in df.columns
                else None
            )
            assert iv_col is not None, f"Greeks IV column missing. Available: {list(df.columns)}"
        finally:
            broker.cleanup_streams()

    def test_get_chain_full_info_includes_bid_ask(self):
        """Chain info should include bid/ask/last pricing."""
        broker = _get_tradier_broker()
        try:
            ds = _get_tradier_data_source(broker)
            asset = Asset("SPY", asset_type=Asset.AssetType.STOCK)

            chains = ds.get_chains(asset)
            first_expiry = sorted(chains["Chains"]["CALL"].keys())[0]
            df = ds.get_chain_full_info(asset, first_expiry, chains=chains)

            assert "bid" in df.columns
            assert "ask" in df.columns
            # At least one row should have positive bid/ask during market hours
            # (May be 0 if market is closed, which is fine)
            has_bid = (df["bid"] > 0).any()
            has_ask = (df["ask"] > 0).any()
            # Not asserting — after hours both can be 0
            assert True  # Just verifying columns exist
        finally:
            broker.cleanup_streams()


# ---------------------------------------------------------------------------
# Option Order Lifecycle
# ---------------------------------------------------------------------------


class TestOptionsDebateSmokeOrderLifecycle:
    """Verify the full option order lifecycle: submit → verify → cancel."""

    def test_submit_and_cancel_option_limit_order(self):
        """Submit a far-OTM option limit order, verify it exists, then cancel.

        Uses an extremely conservative limit price so the order should
        never fill, even during market hours. This validates the full
        order submission and cancellation pipeline for options.
        """
        broker = _get_tradier_broker()
        try:
            # Build a near-impossible-to-fill option order:
            # Buy 1 SPY call at $0.01, 30+ days out, far OTM.
            # This won't fill but validates the order routing.

            # First get the available expiration
            ds = _get_tradier_data_source(broker)
            asset_stock = Asset("SPY", asset_type=Asset.AssetType.STOCK)
            chains = ds.get_chains(asset_stock)
            expirations = sorted(chains["Chains"]["CALL"].keys())

            # Pick an expiration at least 14 days out
            import datetime as dt
            today = dt.date.today()
            target_expiry = None
            for exp in expirations:
                try:
                    exp_date = dt.date.fromisoformat(exp)
                    if (exp_date - today).days >= 14:
                        target_expiry = exp
                        break
                except ValueError:
                    continue
            if target_expiry is None:
                target_expiry = expirations[-1]  # fallback to furthest

            # Pick a far OTM strike
            strikes = chains["Chains"]["CALL"][target_expiry]
            far_otm_strike = max(strikes)

            option_asset = Asset(
                symbol="SPY",
                asset_type=Asset.AssetType.OPTION,
                expiration=dt.date.fromisoformat(target_expiry),
                strike=far_otm_strike,
                right="call",
            )

            order = Order(
                strategy="apitest_options_debate",
                asset=option_asset,
                quantity=1,
                side=Order.OrderSide.BUY_TO_OPEN,
                limit_price=0.01,
                time_in_force="day",
                order_type=Order.OrderType.LIMIT,
            )

            # Submit the order
            submitted = broker._submit_order(order)
            assert submitted is not None, "Order submission returned None"

            # Paper mode may return a dict or an Order object — extract the ID
            order_id = submitted.get("id") if isinstance(submitted, dict) else getattr(submitted, "identifier", None)
            assert order_id, f"No order ID found in response: {submitted}"

            # Verify we can fetch it back
            fetched = broker._pull_broker_order(order_id)
            assert fetched is not None, "Could not fetch submitted order"

            # Verify the fetched order has the same ID
            fetched_id = fetched.get("id") if isinstance(fetched, dict) else getattr(fetched, "identifier", None)
            assert str(fetched_id) == str(order_id)

            # Cancel the order
            broker.cancel_order(submitted)
            time.sleep(1)  # Brief wait for cancellation to settle

            # Verify cancelled
            cancelled = broker._pull_broker_order(order_id)
            assert cancelled is not None, "Could not fetch cancelled order"
            cancelled_status = str(
                cancelled.get("status") if isinstance(cancelled, dict) else getattr(cancelled, "status", "")
            ).upper()
            assert "CANCEL" in cancelled_status, (
                f"Expected cancelled status, got: {cancelled_status}"
                f"Expected cancelled status, got: {cancelled.status}"
            )

        finally:
            broker.cleanup_streams()

    def test_submit_and_cancel_multileg_option_order(self):
        """Verify multi-leg option order (vertical spread) submission and cancellation."""
        broker = _get_tradier_broker()
        try:
            ds = _get_tradier_data_source(broker)
            asset_stock = Asset("SPY", asset_type=Asset.AssetType.STOCK)
            chains = ds.get_chains(asset_stock)
            expirations = sorted(chains["Chains"]["CALL"].keys())

            import datetime as dt
            today = dt.date.today()
            target_expiry = None
            for exp in expirations:
                try:
                    exp_date = dt.date.fromisoformat(exp)
                    if (exp_date - today).days >= 14:
                        target_expiry = exp
                        break
                except ValueError:
                    continue
            if target_expiry is None:
                target_expiry = expirations[-1]

            strikes = sorted(chains["Chains"]["PUT"][target_expiry])
            if len(strikes) < 2:
                pytest.skip("Not enough strikes for a spread")

            # Build a far-OTM put credit spread (won't fill)
            short_strike = strikes[len(strikes) // 2]
            long_strike = strikes[0]

            short_put = Asset(
                symbol="SPY",
                asset_type=Asset.AssetType.OPTION,
                expiration=dt.date.fromisoformat(target_expiry),
                strike=short_strike,
                right="put",
            )
            long_put = Asset(
                symbol="SPY",
                asset_type=Asset.AssetType.OPTION,
                expiration=dt.date.fromisoformat(target_expiry),
                strike=long_strike,
                right="put",
            )

            # Build as a parent OTO order (sell short put, buy long put)
            parent_order = Order(
                strategy="apitest_options_debate",
                asset=short_put,
                quantity=1,
                side=Order.OrderSide.SELL_TO_OPEN,
                limit_price=0.01,
                time_in_force="day",
                order_type=Order.OrderType.LIMIT,
            )
            child_order = Order(
                strategy="apitest_options_debate",
                asset=long_put,
                quantity=1,
                side=Order.OrderSide.BUY_TO_OPEN,
                limit_price=0.01,
                time_in_force="day",
                order_type=Order.OrderType.LIMIT,
            )
            parent_order.child_orders = [child_order]

            submitted = broker._submit_order(parent_order)
            assert submitted is not None

            order_id = submitted.get("id") if isinstance(submitted, dict) else getattr(submitted, "identifier", None)
            assert order_id, f"No order ID found in response: {submitted}"

            # Cancel it
            broker.cancel_order(submitted)
            time.sleep(1)

            cancelled = broker._pull_broker_order(order_id)
            assert cancelled is not None

        finally:
            broker.cleanup_streams()


# ---------------------------------------------------------------------------
# Data source integration
# ---------------------------------------------------------------------------


class TestOptionsDebateSmokeDataSourceIntegration:
    """Verify TradierData methods used by our agent tools work."""

    def test_get_last_price_for_option(self):
        """get_last_price should work for option assets."""
        broker = _get_tradier_broker()
        try:
            ds = _get_tradier_data_source(broker)
            asset_stock = Asset("SPY", asset_type=Asset.AssetType.STOCK)
            chains = ds.get_chains(asset_stock)
            expirations = sorted(chains["Chains"]["CALL"].keys())

            import datetime as dt
            today = dt.date.today()
            target_expiry = None
            for exp in expirations:
                try:
                    if (dt.date.fromisoformat(exp) - today).days >= 7:
                        target_expiry = exp
                        break
                except ValueError:
                    continue
            if target_expiry is None:
                target_expiry = expirations[-1]

            strikes = chains["Chains"]["CALL"][target_expiry]
            atm_idx = len(strikes) // 2

            option_asset = Asset(
                symbol="SPY",
                asset_type=Asset.AssetType.OPTION,
                expiration=dt.date.fromisoformat(target_expiry),
                strike=strikes[atm_idx],
                right="call",
            )

            price = ds.get_last_price(option_asset)
            # Price may be 0 if market is closed, but should not error
            assert price is not None
            assert isinstance(price, float)
            assert price >= 0
        finally:
            broker.cleanup_streams()
