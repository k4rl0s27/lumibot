"""
Unit tests for the TelegramBot monitoring module.

Tests cover:
- Bot initialization and configuration
- Authorization / access control
- Strategy reference management
- Portfolio status / positions text builders
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from lumibot.components.notifications.telegram_bot import TelegramBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubStrategy:
    """Minimal strategy stub for TelegramBot tests."""

    is_backtesting = False

    def get_datetime(self):
        from datetime import datetime, timezone
        return datetime(2026, 7, 9, 15, 30, tzinfo=timezone.utc)

    def get_cash(self):
        return 100000.0

    def get_portfolio_value(self):
        return 105000.0

    def get_positions(self):
        return []


def _make_bot(strategy=None, bot_token="test-token", chat_id="12345"):
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
        bot = _make_bot(bot_token="abc", chat_id="999")
        assert bot.bot_token == "abc"
        assert bot.chat_id == "999"

    def test_bot_reads_from_env(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token", "TELEGRAM_CHAT_ID": "env-chat"}):
            bot = TelegramBot()
            assert bot.bot_token == "env-token"
            assert bot.chat_id == "env-chat"

    def test_bot_starts_with_strategy_reference(self):
        strategy = _StubStrategy()
        bot = _make_bot(strategy=strategy)
        assert bot.strategy is strategy

    def test_set_strategy_updates_reference(self):
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
        bot = _make_bot(chat_id="12345")
        assert bot._is_authorized("12345") is True

    def test_unauthorized_chat_id_is_denied(self):
        bot = _make_bot(chat_id="12345")
        assert bot._is_authorized("99999") is False

    def test_empty_authorized_set_allows_all(self):
        bot = _make_bot(chat_id="")
        bot._authorized_chat_ids = set()
        assert bot._is_authorized("anyone") is True


# ---------------------------------------------------------------------------
# Status Text Building
# ---------------------------------------------------------------------------


class TestStatusTextBuilding:
    """Tests for portfolio status and position text builders."""

    def test_build_status_text_with_strategy(self):
        bot = _make_bot(strategy=_StubStrategy())
        text = bot._build_status_text()
        assert "Portfolio Status" in text
        assert "$100,000.00" in text
        assert "$105,000.00" in text

    def test_build_status_text_without_strategy(self):
        bot = _make_bot(strategy=None)
        bot._strategy = None
        text = bot._build_status_text()
        assert "not connected" in text.lower()

    def test_build_portfolio_text_no_positions(self):
        bot = _make_bot(strategy=_StubStrategy())
        text = bot._build_portfolio_text()
        assert "No open positions" in text

    def test_build_positions_text_empty(self):
        bot = _make_bot(strategy=_StubStrategy())
        text = bot._build_positions_text()
        assert "No open positions" in text
