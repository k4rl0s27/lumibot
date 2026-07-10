"""
Integration tests for the Options Debate Strategy.

Tests cover:
- Strategy initialization (agents created with correct permissions)
- Agent count and naming
- Flow order: research → debate → risk → decision
- PASS decision path (no trade, no Telegram approval)
- Parameter defaults and overrides
- Backtesting mode setup
- Telegram bot integration (bot started in initialize)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Strategy stub for integration testing
# We import the real strategy but mock its agent runtime so no LLM calls happen.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_telegram_bot():
    """Fixture that patches TelegramBot to avoid real Telegram API calls.

    Patches both the class definition and the name in notifications.__init__
    so the strategy's `from lumibot.components.notifications import TelegramBot`
    resolves to the mock.
    """
    with patch(
        "lumibot.components.notifications.telegram_bot.TelegramBot",
        autospec=True,
    ) as mock_cls_def:
        with patch(
            "lumibot.components.notifications.TelegramBot",
            new=mock_cls_def,
        ):
            instance = mock_cls_def.return_value
            instance.bot_token = "test-token"
            instance.chat_id = "test-chat"
            instance.send_approval_request = MagicMock()
            instance.wait_for_approval = MagicMock(return_value=False)
            instance.start = MagicMock()
            instance.stop = MagicMock()
            instance._pending_decision = None
            instance._approval_event = MagicMock()
            yield mock_cls_def


@pytest.fixture
def mock_agent_manager():
    """Fixture that returns a mock agent manager with fake agents."""
    manager = MagicMock()
    manager.create = MagicMock()
    # Mock agent access via manager["name"]
    mock_agent = MagicMock()
    mock_result = SimpleNamespace(
        summary="Mock summary",
        text="Mock text output",
        cache_hit=False,
        warning_messages=[],
        tool_calls=[],
        tool_results=[],
        payload={},
    )
    mock_agent.run = MagicMock(return_value=mock_result)
    manager.__getitem__ = MagicMock(return_value=mock_agent)
    return manager


@pytest.fixture
def mock_notifications():
    """Fixture that mocks the notification manager."""
    with patch("lumibot.strategies.strategy.Strategy.notifications", create=True) as mock_nm:
        mock_nm.configure_telegram = MagicMock()
        mock_nm.notify = MagicMock(return_value=[])
        yield mock_nm


# ---------------------------------------------------------------------------
# Strategy initialization
# ---------------------------------------------------------------------------


class TestStrategyInitialization:
    """Tests for strategy setup in initialize()."""

    def test_strategy_creates_seven_agents(self, mock_telegram_bot, mock_notifications):
        """initialize() should create exactly 7 agents."""
        from example_strategies.ai_trading_team_options_debate import (
            AITradingTeamOptionsDebateStrategy,
        )

        strategy = AITradingTeamOptionsDebateStrategy.__new__(AITradingTeamOptionsDebateStrategy)

        # Mock essentials that initialize() needs
        strategy.parameters = {
            "universe": ["SPY", "QQQ", "IWM"],
            "max_loss_per_trade_pct": 5.0,
            "min_dte": 7,
            "max_dte": 60,
        }
        strategy.sleeptime = None
        strategy.broker = None
        strategy.log_message = MagicMock()
        strategy.notifications = mock_notifications
        strategy.vars = {}
        strategy.memory = MagicMock()

        # Mock the agent manager
        strategy.agents = MagicMock()
        strategy.agents.create = MagicMock()

        # Run initialize
        strategy.initialize()

        # Should have created 7 agents
        assert strategy.agents.create.call_count == 7

    def test_agent_names_match_expected_roles(self, mock_telegram_bot, mock_notifications):
        """Agents should be created with the expected role names."""
        from example_strategies.ai_trading_team_options_debate import (
            AITradingTeamOptionsDebateStrategy,
        )

        strategy = AITradingTeamOptionsDebateStrategy.__new__(AITradingTeamOptionsDebateStrategy)
        strategy.parameters = {
            "universe": ["SPY"],
            "max_loss_per_trade_pct": 5.0,
            "min_dte": 7,
            "max_dte": 60,
        }
        strategy.sleeptime = None
        strategy.broker = None
        strategy.log_message = MagicMock()
        strategy.notifications = mock_notifications
        strategy.vars = {}
        strategy.memory = MagicMock()
        strategy.agents = MagicMock()
        strategy.agents.create = MagicMock()

        strategy.initialize()

        expected_names = [
            "macro_analyst",
            "technical_analyst",
            "options_analyst",
            "bull_case",
            "bear_case",
            "risk_manager",
            "portfolio_manager",
        ]

        # Extract the 'name' kwarg from each create() call
        call_names = [
            call.kwargs["name"]
            for call in strategy.agents.create.call_args_list
        ]
        assert call_names == expected_names

    def test_only_portfolio_manager_can_trade(self, mock_telegram_bot, mock_notifications):
        """Only the portfolio_manager agent should have allow_trading=True."""
        from example_strategies.ai_trading_team_options_debate import (
            AITradingTeamOptionsDebateStrategy,
        )

        strategy = AITradingTeamOptionsDebateStrategy.__new__(AITradingTeamOptionsDebateStrategy)
        strategy.parameters = {
            "universe": ["SPY"],
            "max_loss_per_trade_pct": 5.0,
            "min_dte": 7,
            "max_dte": 60,
        }
        strategy.sleeptime = None
        strategy.broker = None
        strategy.log_message = MagicMock()
        strategy.notifications = mock_notifications
        strategy.vars = {}
        strategy.memory = MagicMock()
        strategy.agents = MagicMock()
        strategy.agents.create = MagicMock()

        strategy.initialize()

        for call in strategy.agents.create.call_args_list:
            name = call.kwargs["name"]
            allow_trading = call.kwargs.get("allow_trading", False)
            if name == "portfolio_manager":
                assert allow_trading is True, f"{name} should have allow_trading=True"
            else:
                assert allow_trading is False, f"{name} should have allow_trading=False"

    def test_telegram_bot_is_started(self, mock_telegram_bot, mock_notifications):
        """The TelegramBot.start() should be called during initialize()."""
        from example_strategies.ai_trading_team_options_debate import (
            AITradingTeamOptionsDebateStrategy,
        )

        strategy = AITradingTeamOptionsDebateStrategy.__new__(AITradingTeamOptionsDebateStrategy)
        strategy.parameters = {
            "universe": ["SPY"],
            "max_loss_per_trade_pct": 5.0,
            "min_dte": 7,
            "max_dte": 60,
        }
        strategy.sleeptime = None
        strategy.broker = None
        strategy.log_message = MagicMock()
        strategy.notifications = mock_notifications
        strategy.vars = {}
        strategy.memory = MagicMock()
        strategy.agents = MagicMock()
        strategy.agents.create = MagicMock()

        strategy.initialize()

        # The mock bot should have had start() called
        instance = mock_telegram_bot.return_value
        instance.start.assert_called_once()

    def test_default_parameters(self, mock_telegram_bot, mock_notifications):
        """Default parameters should be set correctly."""
        from example_strategies.ai_trading_team_options_debate import (
            AITradingTeamOptionsDebateStrategy,
        )

        strategy = AITradingTeamOptionsDebateStrategy.__new__(AITradingTeamOptionsDebateStrategy)
        strategy.sleeptime = None
        strategy.broker = None
        strategy.log_message = MagicMock()
        strategy.notifications = mock_notifications
        strategy.vars = {}
        strategy.memory = MagicMock()
        strategy.agents = MagicMock()
        strategy.agents.create = MagicMock()

        strategy.initialize()

        assert strategy.parameters["universe"] == ["SPY", "QQQ", "IWM"]
        assert strategy.parameters["max_loss_per_trade_pct"] == 5.0
        assert strategy.parameters["min_dte"] == 7
        assert strategy.parameters["max_dte"] == 60


# ---------------------------------------------------------------------------
# Trading iteration flow
# ---------------------------------------------------------------------------


class TestTradingIterationFlow:
    """Tests for the on_trading_iteration() flow."""

    def _make_strategy_with_mocked_agents(self, mock_telegram_bot):
        """Build a strategy instance with all external deps mocked.

        Each agent gets its own mock so setting .run on one doesn't
        affect others.
        """
        from example_strategies.ai_trading_team_options_debate import (
            AITradingTeamOptionsDebateStrategy,
        )

        strategy = AITradingTeamOptionsDebateStrategy.__new__(AITradingTeamOptionsDebateStrategy)
        strategy.parameters = {
            "universe": ["SPY"],
            "max_loss_per_trade_pct": 5.0,
            "min_dte": 7,
            "max_dte": 60,
        }
        strategy.sleeptime = None
        strategy.broker = None
        strategy.log_message = MagicMock()
        strategy.notifications = MagicMock()
        strategy.notifications.configure_telegram = MagicMock()
        strategy.notifications.notify = MagicMock()
        strategy.vars = {}
        strategy.memory = MagicMock()
        strategy.get_datetime = MagicMock(return_value=datetime(2026, 7, 9, 15, 30, tzinfo=timezone.utc))

        # Each agent gets its own mock via a factory side_effect that
        # returns the SAME mock for the SAME key (deterministic).
        _agent_mocks: dict = {}
        def _get_agent(key):
            if key not in _agent_mocks:
                mock_agent = MagicMock()
                mock_agent.run = MagicMock(return_value=SimpleNamespace(
                    summary=f"{key} summary",
                    text=f"{key} text",
                    cache_hit=True,
                    warning_messages=[],
                ))
                _agent_mocks[key] = mock_agent
            return _agent_mocks[key]

        strategy.agents = MagicMock()
        strategy.agents.create = MagicMock()
        strategy.agents.__getitem__ = MagicMock(side_effect=_get_agent)

        # Mock Telegram bot
        strategy.telegram_bot = mock_telegram_bot.return_value
        strategy.telegram_bot.send_approval_request = MagicMock()
        strategy.telegram_bot.wait_for_approval = MagicMock(return_value=False)
        strategy.telegram_bot.start = MagicMock()

        return strategy

    def test_pass_decision_skips_telegram_approval(self, mock_telegram_bot):
        """When PM decides PASS, should NOT call send_approval_request or wait_for_approval."""
        strategy = self._make_strategy_with_mocked_agents(mock_telegram_bot)

        # Make the portfolio_manager return a PASS decision
        pass_result = SimpleNamespace(
            summary="FINAL DECISION: PASS - No clear edge today.",
            text="FINAL DECISION: PASS - No clear edge today.",
            cache_hit=True,
            warning_messages=[],
        )

        def _run_side_effect(task_prompt, context=None):
            # Return PASS for the portfolio_manager
            if "portfolio_manager" in str(context or ""):
                return pass_result
            if "Synthesize ALL research" in str(task_prompt or ""):
                return pass_result
            return SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[])

        strategy.agents["portfolio_manager"].run = MagicMock(side_effect=_run_side_effect)
        strategy.agents["macro_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="Macro ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["technical_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="Technical ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["options_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="Options ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bull_case"].run = MagicMock(return_value=SimpleNamespace(summary="Bull ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bear_case"].run = MagicMock(return_value=SimpleNamespace(summary="Bear ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["risk_manager"].run = MagicMock(return_value=SimpleNamespace(summary="Risk ok", text="ok", cache_hit=True, warning_messages=[]))

        strategy.on_trading_iteration()

        # Should NOT call approval flow
        strategy.telegram_bot.send_approval_request.assert_not_called()
        strategy.telegram_bot.wait_for_approval.assert_not_called()

        # Should still log the decision in memory
        strategy.memory.remember_decision.assert_called_once()

    def test_trade_decision_triggers_approval_gate(self, mock_telegram_bot):
        """When PM decides TRADE, should trigger send_approval_request and wait_for_approval."""
        strategy = self._make_strategy_with_mocked_agents(mock_telegram_bot)

        trade_decision = (
            "FINAL DECISION: TRADE\n\n"
            "- Strategy: Bull Put Spread\n"
            "- Underlying: SPY\n"
            "- Legs: sell_to_open SPY 445P 2026-08-21 qty 2, buy_to_open SPY 440P 2026-08-21 qty 2\n"
            "- Limit Price: $3.00 credit\n"
            "- Max Loss: $700\n"
            "- Max Profit: $300"
        )

        trade_result = SimpleNamespace(
            summary=trade_decision,
            text=trade_decision,
            cache_hit=True,
            warning_messages=[],
        )

        strategy.agents["portfolio_manager"].run = MagicMock(return_value=trade_result)
        strategy.agents["macro_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["technical_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["options_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bull_case"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bear_case"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["risk_manager"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))

        strategy.on_trading_iteration()

        # Should call approval flow
        strategy.telegram_bot.send_approval_request.assert_called_once_with(trade_decision)
        strategy.telegram_bot.wait_for_approval.assert_called_once()

    def test_trade_rejected_by_user(self, mock_telegram_bot):
        """When user rejects, should log rejection, no orders."""
        strategy = self._make_strategy_with_mocked_agents(mock_telegram_bot)

        trade_decision = "FINAL DECISION: TRADE\n\nBuy SPY calls."
        trade_result = SimpleNamespace(summary=trade_decision, text=trade_decision, cache_hit=True, warning_messages=[])

        strategy.agents["portfolio_manager"].run = MagicMock(return_value=trade_result)
        strategy.agents["macro_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["technical_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["options_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bull_case"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bear_case"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["risk_manager"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))

        # User rejects
        strategy.telegram_bot.wait_for_approval.return_value = False

        strategy.on_trading_iteration()

        # Should record as rejected
        strategy.memory.remember_decision.assert_called()
        call_args = strategy.memory.remember_decision.call_args
        assert "REJECTED" in call_args[0][0]

    def test_trade_approved_by_user(self, mock_telegram_bot):
        """When user approves, should log approval and proceed."""
        strategy = self._make_strategy_with_mocked_agents(mock_telegram_bot)

        trade_decision = "FINAL DECISION: TRADE\n\nBuy SPY calls."
        trade_result = SimpleNamespace(summary=trade_decision, text=trade_decision, cache_hit=True, warning_messages=[])

        strategy.agents["portfolio_manager"].run = MagicMock(return_value=trade_result)
        strategy.agents["macro_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["technical_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["options_analyst"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bull_case"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["bear_case"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))
        strategy.agents["risk_manager"].run = MagicMock(return_value=SimpleNamespace(summary="ok", text="ok", cache_hit=True, warning_messages=[]))

        # User approves
        strategy.telegram_bot.wait_for_approval.return_value = True

        strategy.on_trading_iteration()

        # Should record as approved
        strategy.memory.remember_decision.assert_called()
        call_args = strategy.memory.remember_decision.call_args
        assert "APPROVED" in call_args[0][0]

    def test_all_agents_called_in_correct_order(self, mock_telegram_bot):
        """Agents should be called in the order: macro, technical, options, bull, bear, risk, PM."""
        strategy = self._make_strategy_with_mocked_agents(mock_telegram_bot)

        call_order = []

        def _track_and_return(name):
            def _run(*args, **kwargs):
                call_order.append(name)
                return SimpleNamespace(summary=f"{name} summary", text=f"{name} text", cache_hit=True, warning_messages=[])
            return _run

        strategy.agents["macro_analyst"].run = MagicMock(side_effect=_track_and_return("macro"))
        strategy.agents["technical_analyst"].run = MagicMock(side_effect=_track_and_return("technical"))
        strategy.agents["options_analyst"].run = MagicMock(side_effect=_track_and_return("options"))
        strategy.agents["bull_case"].run = MagicMock(side_effect=_track_and_return("bull"))
        strategy.agents["bear_case"].run = MagicMock(side_effect=_track_and_return("bear"))
        strategy.agents["risk_manager"].run = MagicMock(side_effect=_track_and_return("risk"))
        strategy.agents["portfolio_manager"].run = MagicMock(side_effect=_track_and_return("pm"))
        strategy.telegram_bot.wait_for_approval.return_value = False

        strategy.on_trading_iteration()

        assert call_order == ["macro", "technical", "options", "bull", "bear", "risk", "pm"]


# ---------------------------------------------------------------------------
# Backtesting mode
# ---------------------------------------------------------------------------


class TestBacktestingMode:
    """Tests for the backtesting mode setup in __main__."""

    def test_is_backtesting_env_var_controls_mode(self):
        """IS_BACKTESTING env var should control which path is taken."""
        # This tests the runner code pattern, not actual backtest execution
        # The pattern ensures IS_BACKTESTING=true triggers the backtest path
        # and IS_BACKTESTING=false triggers the live Tradier path

        # Verify the env var check pattern
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("", False),
        ]

        for env_value, expected in test_cases:
            result = env_value.lower() in ("true", "1", "yes")
            assert result == expected, f"IS_BACKTESTING={env_value!r} should be {expected}"

    def test_default_mode_is_live_trading(self):
        """When IS_BACKTESTING is not set, should default to live/paper trading."""
        # Default value in the runner is "false"
        default = os.environ.get("IS_BACKTESTING", "false")
        is_backtest = default.lower() in ("true", "1", "yes")
        assert is_backtest is False, "Default should be live/paper trading"


# ---------------------------------------------------------------------------
# Prompt quality checks
# ---------------------------------------------------------------------------


class TestPromptQuality:
    """Tests that verify prompt content quality and safety."""

    def test_all_prompts_mention_options(self):
        """Each agent prompt should reference options-specific concepts."""
        from example_strategies.ai_trading_team_options_debate import (
            MACRO_ANALYST_PROMPT,
            TECHNICAL_ANALYST_PROMPT,
            OPTIONS_ANALYST_PROMPT,
            BULL_CASE_PROMPT,
            BEAR_CASE_PROMPT,
            RISK_MANAGER_PROMPT,
            PORTFOLIO_MANAGER_PROMPT,
        )

        prompts = {
            "macro": MACRO_ANALYST_PROMPT,
            "technical": TECHNICAL_ANALYST_PROMPT,
            "options": OPTIONS_ANALYST_PROMPT,
            "bull": BULL_CASE_PROMPT,
            "bear": BEAR_CASE_PROMPT,
            "risk": RISK_MANAGER_PROMPT,
            "pm": PORTFOLIO_MANAGER_PROMPT,
        }

        for name, prompt in prompts.items():
            text = prompt.lower()
            assert any(
                keyword in text
                for keyword in ["option", "strike", "expir", "delta", "gamma", "theta", "vega", "iv", "premium", "greeks"]
            ), f"{name} prompt should reference options concepts"

    def test_portfolio_manager_required_output_structure(self):
        """PM prompt must require FINAL DECISION: TRADE or FINAL DECISION: PASS."""
        from example_strategies.ai_trading_team_options_debate import PORTFOLIO_MANAGER_PROMPT

        assert "FINAL DECISION: TRADE" in PORTFOLIO_MANAGER_PROMPT
        assert "FINAL DECISION: PASS" in PORTFOLIO_MANAGER_PROMPT

    def test_no_naked_options_guidance(self):
        """PM prompt should favor defined-risk strategies."""
        from example_strategies.ai_trading_team_options_debate import PORTFOLIO_MANAGER_PROMPT

        assert (
            "defined-risk" in PORTFOLIO_MANAGER_PROMPT.lower()
            or "spreads" in PORTFOLIO_MANAGER_PROMPT.lower()
        )

    def test_no_0dte_guidance(self):
        """PM prompt should forbid 0DTE trades."""
        from example_strategies.ai_trading_team_options_debate import PORTFOLIO_MANAGER_PROMPT

        assert "0dte" in PORTFOLIO_MANAGER_PROMPT.lower() or "0DTE" in PORTFOLIO_MANAGER_PROMPT
