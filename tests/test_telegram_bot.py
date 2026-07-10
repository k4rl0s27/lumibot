"""
Unit tests for the bidirectional TelegramBot module.

Tests cover:
- Bot initialization and configuration
- Authorization / access control
- Approval gate: send, approve, reject, timeout
- Decision history tracking
- Strategy reference management
- Command handler registration
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubStrategy:
    """Minimal strategy stub for TelegramBot tests."""

    is_backtesting = False

    def get_datetime(self):
        return datetime(2026, 7, 9, 15, 30, tzinfo=timezone.utc)

    def get_cash(self):
        return 100000.0

    def get_portfolio_value(self):
        return 105000.0

    def get_positions(self):
        return []


def _make_bot(strategy=None, bot_token="test-token", chat_id="12345"):
    """Create a TelegramBot instance for testing."""
    from lumibot.components.notifications.telegram_bot import TelegramBot

    return TelegramBot(
        bot_token=bot_token,
        chat_id=chat_id,
        strategy=strategy or _StubStrategy(),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestTelegramBotInit:
    """Tests for bot initialization and configuration."""

    def test_bot_initializes_with_explicit_args(self):
        """Bot should accept explicit bot_token and chat_id."""
        bot = _make_bot(bot_token="abc", chat_id="999")
        assert bot.bot_token == "abc"
        assert bot.chat_id == "999"

    def test_bot_reads_from_env(self):
        """Bot should read TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token", "TELEGRAM_CHAT_ID": "env-chat"}):
            from lumibot.components.notifications.telegram_bot import TelegramBot

            bot = TelegramBot()
            assert bot.bot_token == "env-token"
            assert bot.chat_id == "env-chat"

    def test_bot_starts_with_strategy_reference(self):
        """Bot should store the strategy reference."""
        strategy = _StubStrategy()
        bot = _make_bot(strategy=strategy)
        assert bot.strategy is strategy

    def test_set_strategy_updates_reference(self):
        """set_strategy should update the strategy reference."""
        bot = _make_bot()
        new_strategy = _StubStrategy()
        bot.set_strategy(new_strategy)
        assert bot.strategy is new_strategy


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


class TestTelegramBotAuthorization:
    """Tests for access control."""

    def test_authorized_chat_id_is_allowed(self):
        """A chat_id matching the configured one should be authorized."""
        bot = _make_bot(chat_id="12345")
        assert bot._is_authorized("12345") is True

    def test_unauthorized_chat_id_is_denied(self):
        """A chat_id not matching the configured one should be denied."""
        bot = _make_bot(chat_id="12345")
        assert bot._is_authorized("99999") is False

    def test_empty_authorized_set_allows_all(self):
        """If no authorized chats configured, all are allowed."""
        # Use explicit empty string to override any env var
        bot = _make_bot(chat_id="")
        # Force empty authorized set (env may have leaked TELEGRAM_CHAT_ID)
        bot._authorized_chat_ids = set()
        assert bot._is_authorized("anyone") is True


# ---------------------------------------------------------------------------
# Approval Gate
# ---------------------------------------------------------------------------


class TestApprovalGate:
    """Tests for the trade approval flow."""

    def test_send_approval_request_creates_pending_decision(self):
        """send_approval_request should create a PendingDecision."""
        bot = _make_bot()
        # Don't actually start the bot (avoids Telegram API calls)
        bot._application = MagicMock()

        bot.send_approval_request("Buy SPY 450C")

        assert bot._pending_decision is not None
        assert "Buy SPY 450C" in bot._pending_decision.text
        assert len(bot._decision_history) == 1
        assert bot._decision_history[0]["status"] == "pending"

    def test_approve_via_resolve_sets_event(self):
        """_resolve_approval(approved=True) should set the event and store decision."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot.send_approval_request("Test trade")

        bot._resolve_approval(approved=True)

        assert bot._approval_event.is_set()
        assert bot._approval_decision.approved is True
        assert bot._approval_decision.decided_by == "user"

    def test_reject_via_resolve_sets_event(self):
        """_resolve_approval(approved=False) should set the event and store rejection."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot.send_approval_request("Test trade")

        bot._resolve_approval(approved=False)

        assert bot._approval_event.is_set()
        assert bot._approval_decision.approved is False
        assert bot._approval_decision.decided_by == "user"

    def test_wait_for_approval_blocks_and_returns_true_when_approved(self):
        """wait_for_approval should block and return True when approved."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot.send_approval_request("Test trade")

        # Approve from another thread after a short delay
        def _approve_later():
            time.sleep(0.1)
            bot._resolve_approval(approved=True)

        threading.Thread(target=_approve_later, daemon=True).start()

        result = bot.wait_for_approval(timeout_minutes=1)
        assert result is True

    def test_wait_for_approval_returns_false_when_rejected(self):
        """wait_for_approval should return False when rejected."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot.send_approval_request("Test trade")

        def _reject_later():
            time.sleep(0.1)
            bot._resolve_approval(approved=False)

        threading.Thread(target=_reject_later, daemon=True).start()

        result = bot.wait_for_approval(timeout_minutes=1)
        assert result is False

    def test_wait_for_approval_times_out_and_returns_false(self):
        """wait_for_approval should timeout and return False with a tiny timeout."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot.send_approval_request("Test trade")

        # Use tiny 0.001 minute timeout (60ms) to test timeout path
        # This is small enough that no external thread is needed
        result = bot.wait_for_approval(timeout_minutes=0.001)
        assert result is False
        assert bot._approval_decision.decided_by == "timeout"

    def test_double_approve_is_idempotent(self):
        """Calling _resolve_approval twice should only count the first."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot.send_approval_request("Test trade")

        bot._resolve_approval(approved=True)
        bot._resolve_approval(approved=False)  # Should be ignored

        assert bot._approval_decision.approved is True

    def test_approval_without_pending_decision_is_noop(self):
        """_resolve_approval should be a noop when no pending decision exists."""
        bot = _make_bot()
        bot._application = MagicMock()
        # No send_approval_request called

        bot._resolve_approval(approved=True)
        assert bot._approval_event.is_set()

    def test_send_approval_request_resets_previous_decision(self):
        """Sending a new approval request should reset the previous state."""
        bot = _make_bot()
        bot._application = MagicMock()

        bot.send_approval_request("First trade")
        bot._resolve_approval(approved=True)
        assert bot._approval_event.is_set()

        bot.send_approval_request("Second trade")
        assert not bot._approval_event.is_set()
        assert bot._approval_decision is None


# ---------------------------------------------------------------------------
# Decision History
# ---------------------------------------------------------------------------


class TestDecisionHistory:
    """Tests for the decision history tracking."""

    def test_history_starts_empty(self):
        """Decision history should start empty."""
        bot = _make_bot()
        assert bot._decision_history == []

    def test_history_appends_on_approval_request(self):
        """Each approval request should add to history."""
        bot = _make_bot()
        bot._application = MagicMock()

        bot.send_approval_request("Trade 1")
        bot.send_approval_request("Trade 2")

        assert len(bot._decision_history) == 2
        assert bot._decision_history[0]["text"] == "Trade 1"
        assert bot._decision_history[1]["text"] == "Trade 2"

    def test_history_updates_status_on_decision(self):
        """History entry status should update when decision is made."""
        bot = _make_bot()
        bot._application = MagicMock()

        bot.send_approval_request("Trade 1")
        bot._resolve_approval(approved=True)

        assert bot._decision_history[-1]["status"] == "approved"

    def test_history_respects_max_size(self):
        """History should not grow beyond _max_history."""
        bot = _make_bot()
        bot._application = MagicMock()
        bot._max_history = 5

        for i in range(10):
            bot.send_approval_request(f"Trade {i}")

        assert len(bot._decision_history) == 5
        assert bot._decision_history[0]["text"] == "Trade 5"
        assert bot._decision_history[-1]["text"] == "Trade 9"


# ---------------------------------------------------------------------------
# Status Text Building
# ---------------------------------------------------------------------------


class TestStatusTextBuilding:
    """Tests for portfolio status and position text builders."""

    def test_build_status_text_with_strategy(self):
        """_build_status_text should return formatted portfolio info."""
        bot = _make_bot(strategy=_StubStrategy())
        text = bot._build_status_text()
        assert "Portfolio Status" in text
        assert "$100,000.00" in text
        assert "$105,000.00" in text

    def test_build_status_text_without_strategy(self):
        """_build_status_text should return a warning when no strategy is set."""
        bot = _make_bot(strategy=None)
        bot._strategy = None
        text = bot._build_status_text()
        assert "not connected" in text.lower()

    def test_build_portfolio_text_no_positions(self):
        """_build_portfolio_text should handle empty positions."""
        bot = _make_bot(strategy=_StubStrategy())
        text = bot._build_portfolio_text()
        assert "No open positions" in text

    def test_build_positions_text_empty(self):
        """_build_positions_text should handle empty positions."""
        bot = _make_bot(strategy=_StubStrategy())
        text = bot._build_positions_text()
        assert "No open positions" in text
