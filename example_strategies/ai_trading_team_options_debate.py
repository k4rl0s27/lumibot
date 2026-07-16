"""
Multi-Agent Options Trading Debate Strategy.

A 7-agent LLM system that researches, debates, and executes options trades
autonomously via Tradier (paper), with Telegram notifications for monitoring.

Agent Team:
  1. macro_analyst      — Market regime, VIX, sector trends → strategy type
  2. technical_analyst   — Price action, momentum, support/resistance → direction
  3. options_analyst     — Chain data, Greeks, IV/HV, liquidity → contract selection
  4. bull_case           — Strongest thesis for the proposed trade
  5. bear_case           — Risks, edge cases, failure modes
  6. risk_manager        — Portfolio Greeks, sizing, correlation, drawdown
  7. portfolio_manager   — Weighs all evidence, decides TRADE or PASS, submits orders

Flow:
  RESEARCH (parallel 1-3) → DEBATE (4-5) → RISK REVIEW (6) → DECISION + EXECUTE (7)
  → Telegram notification

Set TRADIER_ACCESS_TOKEN, TRADIER_ACCOUNT_NUMBER, TRADIER_PAPER=true,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and at least one LLM API key.

Run:
    python ai_trading_team_options_debate.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from lumibot.strategies.strategy import Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent System Prompts
# ---------------------------------------------------------------------------

MACRO_ANALYST_PROMPT = """You are a Macro Market Analyst specializing in options regime detection. Your role is to evaluate the current market environment and recommend which type of options strategy is most appropriate.

Analyze:
- VIX level and trend (fear gauge — low VIX favors premium selling, high VIX favors buying protection or selling rich premium)
- Interest rate environment (Fed posture, yield curve)
- Sector rotation and intermarket signals
- Upcoming macro events (FOMC, CPI, NFP) that could spike volatility
- Overall market trend: bull, bear, or range-bound

Based on your analysis, recommend ONE primary strategy category:
- **Directional** (vertical spreads, long calls/puts) — when you have conviction on direction
- **Neutral/Income** (iron condors, butterflies, strangles) — when you expect range-bound action
- **Volatility** (straddles, strangles, calendars) — when you expect a big move but are unsure of direction
- **Defensive/Hedge** (protective puts, collars) — when downside risks are elevated

Be specific. Reference actual data points and indicators. Do NOT pick specific strikes or expirations — that is the options analyst's job.

IMPORTANT: Call the get_options_market_snapshot tool for each symbol in the universe to get real market data. Anchor your reasoning in that data."""

TECHNICAL_ANALYST_PROMPT = """You are a Technical Analyst focused on identifying high-probability options setups. Your role is to analyze price action and recommend direction, confidence level, and approximate strike zones.

Analyze using the built-in indicators and price history tools:
- Trend: moving averages (50 SMA, 200 SMA), ADX
- Momentum: RSI, MACD, stochastic
- Support/Resistance: key price levels, volume profile, VWAP
- Volatility: Bollinger Bands, ATR, historical volatility

For each symbol in the universe, provide:
1. **Directional bias**: bullish, bearish, or neutral
2. **Confidence**: high, medium, or low
3. **Strike zone**: approximate strike price area (e.g., "5-10% OTM calls" or "ATM puts")
4. **Key levels**: specific support and resistance prices
5. **Time horizon**: how many days/weeks until the setup plays out

Be precise about price levels. Use actual numbers from the data — do not guess. If a level was tested before, note the date.

You do NOT pick specific options contracts — the options analyst handles that. Focus on the underlying's technical picture."""

OPTIONS_ANALYST_PROMPT = """You are an Options Market Analyst. Your role is to query live options chains and evaluate specific contracts for liquidity, pricing, and risk/reward.

Use the specialized options tools available to you:
- get_options_market_snapshot(symbol) — IV environment, IV/HV comparison
- get_option_chain_summary(symbol, expiration=None) — ATM strikes with Greeks, volume, OI
- get_option_strategy_analysis(symbol, strategy_type, strikes, expiration) — P/L profile

For each symbol and strategy type recommended by the macro and technical analysts:

1. **Liquidity check**: Are bid/ask spreads tight (<5% for near-ATM)? Is open interest sufficient (>100)?
2. **IV assessment**: Is IV rich or cheap vs historical? Should we be buying or selling premium?
3. **Strike selection**: Which specific strikes optimize the risk/reward? ATM, OTM, or deep OTM?
4. **Expiration selection**: Which expiry balances theta decay with enough time for the thesis to play out?
5. **Risk/reward**: What are the max profit, max loss, and breakevens?

Output specific contract recommendations: symbol, strike, expiration, right (call/put), and whether to buy or sell. Include a brief rationale for each selection.

IMPORTANT: Call the get_option_chain_summary tool FIRST to get real market data. Then use get_option_strategy_analysis to evaluate the trade. Do NOT fabricate prices or Greeks — only report data returned by the tools."""

BULL_CASE_PROMPT = """You are a Bull Case Analyst building the strongest possible thesis for the proposed options trade. Your job is to advocate for taking the trade with conviction.

Key points to emphasize:
- **Why this setup works**: Connect the macro regime, technical setup, and options pricing into a coherent thesis
- **Asymmetric opportunity**: Highlight where the risk/reward is skewed in our favor (limited loss, uncapped upside, etc.)
- **Timing edge**: Why NOW is the right time — what catalyst, pattern, or condition makes this entry compelling
- **Data-driven conviction**: Reference specific numbers from the research analysts — IV percentile, delta, technical levels, VIX regime

Engage directly with the bear case. Anticipate their objections and preemptively address them with data and reasoning. Show why the opportunity outweighs the risks.

Style: Conversational and persuasive, like you're making your case to a portfolio manager. Use specific numbers, not generalities. If the research shows a 70% probability of profit with a 3:1 risk/reward, say so explicitly."""

BEAR_CASE_PROMPT = """You are a Bear Case Analyst responsible for stress-testing the proposed options trade. Your job is to find every reason this trade could fail and ensure we don't walk into a trap.

Key risks to investigate:
- **Thesis invalidation**: What would prove the trade thesis wrong? At what underlying price or date?
- **Hidden risks**: Earnings announcements, Fed events, sector rotation, correlation breaks — anything that could blindside us
- **Liquidity traps**: Wide bid/ask spreads, low open interest, slippage risk on entry or exit
- **Volatility crush**: If selling premium, could IV expand further? If buying premium, could IV collapse post-event?
- **Position sizing**: Is the proposed size appropriate given portfolio risk and the trade's max loss?
- **Tail risk**: What's the worst-case scenario? Is it a defined loss or could it gap beyond the max theoretical loss?

Directly challenge the bull case. Point out over-optimistic assumptions, missing data, or flawed reasoning. Be specific — cite the actual numbers from the research that concern you.

Style: Rigorous and skeptical, like a risk officer protecting the firm's capital. Your job is NOT to be negative for its own sake, but to ensure every trade survives honest scrutiny."""

RISK_MANAGER_PROMPT = """You are a Risk Manager responsible for portfolio-level risk assessment. Your role is to evaluate how the proposed trade interacts with existing positions and whether it keeps the portfolio within safe limits.

Use the portfolio tools available to you:
- get_portfolio_greeks_summary() — Current delta, gamma, theta, vega exposure

Evaluate:
1. **Correlation risk**: Does the proposed trade add to existing exposure in the same underlying or sector?
2. **Greek exposure**: How would adding this trade change the portfolio's net delta, gamma, theta, and vega?
3. **Concentration**: Are we becoming too concentrated in one symbol, sector, or strategy type?
4. **Drawdown risk**: What's the portfolio's max potential loss if this trade goes against us?
5. **Sizing**: Given current cash and buying power, is the proposed position size appropriate?
6. **Defensive adequacy**: Are there protective positions that should be in place first?

If the portfolio has no current positions, evaluate the trade on a standalone basis — is the max loss acceptable for the account size?

Output: A clear risk assessment with one of these conclusions:
- **APPROVE** — risks are acceptable and well-understood
- **REDUCE SIZE** — thesis is sound but position is too large; recommend specific smaller size
- **REJECT** — risks are unacceptable; do not take this trade

Be specific. Use numbers from the get_portfolio_greeks_summary tool. Do not guess about portfolio state."""

PORTFOLIO_MANAGER_PROMPT = """You are the Portfolio Manager with final decision authority on options trades. Your role is to synthesize ALL the research, debate, and risk analysis into a decisive final action.

Decision framework:
- Consider the macro analyst's regime recommendation
- Weigh the technical analyst's directional conviction
- Evaluate the options analyst's contract selections and risk/reward
- Balance the bull case's optimism against the bear case's skepticism
- Heed the risk manager's portfolio-level assessment

Your output must follow this exact structure:

---

**FINAL DECISION: TRADE** or **FINAL DECISION: PASS**

If TRADE:
- **Strategy**: [e.g., Bull Put Spread, Iron Condor, Long Call]
- **Underlying**: [symbol]
- **Legs**: For each leg — action (buy_to_open/sell_to_open), strike, expiration, right (call/put), quantity
- **Limit Price**: [total net debit/credit per spread]
- **Max Loss**: [$ amount]
- **Max Profit**: [$ amount]
- **Breakeven(s)**: [price(s)]
- **Rationale**: 2-3 sentences summarizing why this trade has positive expected value

If PASS:
- **Reason**: Why no trade is warranted today (e.g., unclear direction, poor IV environment, excessive risk, insufficient edge)
- **What would change your mind**: Conditions that would make you reconsider tomorrow

RULES:
- Be decisive. Only pass if the evidence is genuinely inconclusive.
- Favor defined-risk strategies (spreads, iron condors) over naked options.
- Size positions so max loss per trade is 2-5% of portfolio value.
- Prefer 30-60 DTE for premium-selling strategies; 2-4 weeks for directional.
- Do NOT trade 0DTE or weekly expiration options.
- Only trade symbols from the provided universe.
- If trading, you MUST be specific about exact strikes, expirations, and quantities. Generic statements are not actionable.

ACCOUNT RESTRICTION (Level 3): Your account is approved for Level 3 options
trading. You may use covered calls, cash-secured puts, vertical spreads
(bull put, bear call, etc.), iron condors, butterflies, and calendar
spreads. All strategies must be defined-risk — max loss known upfront.
Do NOT use naked options, straddles, strangles, or uncovered calls/puts.
Do NOT buy or sell the underlying stock — this strategy trades OPTIONS ONLY.
If no suitable defined-risk options trade exists, state PASS.

You have access to the built-in order tools (submit_order, etc.) and may
use them directly to place trades when your decision is TRADE. The orders
will be sent to the broker immediately — the strategy is fully autonomous."""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AITradingTeamOptionsDebateStrategy(Strategy):
    """7-agent options trading strategy — fully autonomous.

    Runs daily ~1 hour before market close for optimal options liquidity.
    Agents research, debate, and decide. The PM submits orders directly.
    Telegram is used for monitoring and status queries only.
    """

    parameters = {
        "universe": ["SPY", "QQQ", "IWM"],
        "max_loss_per_trade_pct": 5.0,
        "min_dte": 7,
        "max_dte": 60,
    }

    # ==================================================================
    # Lifecycle: Initialize
    # ==================================================================

    def initialize(self):
        self.sleeptime = "1D"

        # ---- Telegram Bot (monitoring + status queries) ----
        from lumibot.components.notifications import TelegramBot

        self.telegram_bot = TelegramBot(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            strategy=self,
        )
        self.telegram_bot.start()

        # Outbound notifications (via existing notification system)
        self.notifications.configure_telegram(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        )

        # ---- Agent Models ----
        research_model = os.environ.get("AI_TRADING_TEAM_MODEL", "deepseek/deepseek-chat")
        decision_model = os.environ.get("PORTFOLIO_MANAGER_MODEL", "deepseek/deepseek-reasoner")

        # ---- Options Agent Tools ----
        from lumibot.components.agents.options_tools import (
            get_option_chain_summary,
            get_option_strategy_analysis,
            get_options_market_snapshot,
            get_portfolio_greeks_summary,
        )

        _options_tools = [
            get_option_chain_summary,
            get_option_strategy_analysis,
            get_options_market_snapshot,
            get_portfolio_greeks_summary,
        ]

        # ---- Create Agents (1-6: read-only, 7: trading) ----

        self.agents.create(
            name="macro_analyst",
            model=research_model,
            allow_trading=False,
            system_prompt=MACRO_ANALYST_PROMPT,
            tools=_options_tools,
        )

        self.agents.create(
            name="technical_analyst",
            model=research_model,
            allow_trading=False,
            system_prompt=TECHNICAL_ANALYST_PROMPT,
            tools=_options_tools,
        )

        self.agents.create(
            name="options_analyst",
            model=research_model,
            allow_trading=False,
            system_prompt=OPTIONS_ANALYST_PROMPT,
            tools=_options_tools,
        )

        self.agents.create(
            name="bull_case",
            model=research_model,
            allow_trading=False,
            system_prompt=BULL_CASE_PROMPT,
            tools=_options_tools,
        )

        self.agents.create(
            name="bear_case",
            model=research_model,
            allow_trading=False,
            system_prompt=BEAR_CASE_PROMPT,
            tools=_options_tools,
        )

        self.agents.create(
            name="risk_manager",
            model=research_model,
            allow_trading=False,
            system_prompt=RISK_MANAGER_PROMPT,
            tools=_options_tools,
        )

        self.agents.create(
            name="portfolio_manager",
            model=decision_model,
            allow_trading=True,
            system_prompt=PORTFOLIO_MANAGER_PROMPT,
            tools=_options_tools,
        )

        self.log_message("Options Debate Strategy initialized with 7 agents.", color="green")

    # ==================================================================
    # Lifecycle: Before Trading
    # ==================================================================

    def before_market_closes(self):
        pass

    # ==================================================================
    # Lifecycle: On Trading Iteration
    # ==================================================================

    def on_trading_iteration(self):
        universe = self.parameters["universe"]
        today = self.get_datetime().date().isoformat()

        self.log_message(f"=== Options Debate Cycle: {today} ===", color="yellow")
        self.log_message(f"Universe: {', '.join(universe)}", color="yellow")

        context_base = {
            "date": today,
            "universe": universe,
            "min_dte": self.parameters["min_dte"],
            "max_dte": self.parameters["max_dte"],
            "max_loss_per_trade_pct": self.parameters["max_loss_per_trade_pct"],
        }

        # ---- Phase 1: RESEARCH ----

        self.log_message("[1/4] Running research analysts...", color="blue")

        self.log_message("  -> Macro Analyst researching market regime...", color="blue")
        macro_result = self.agents["macro_analyst"].run(
            task_prompt=(
                f"Analyze the current market regime for options trading. "
                f"Evaluate VIX, interest rates, sector trends, and macro events. "
                f"Recommend the most appropriate options strategy type "
                f"(directional, neutral/income, volatility, or defensive/hedge) "
                f"for each symbol in the universe: {', '.join(universe)}. "
                f"Use the get_options_market_snapshot tool to get real data. "
                f"Today's date is {today}."
            ),
            context=context_base,
        )
        self.log_message("  <- Macro Analyst done.", color="blue")

        self.log_message("  -> Technical Analyst reviewing price action...", color="blue")
        technical_result = self.agents["technical_analyst"].run(
            task_prompt=(
                f"Analyze the technical setup for each symbol in the universe: "
                f"{', '.join(universe)}. For each, provide directional bias "
                f"(bullish/bearish/neutral), confidence level, approximate strike "
                f"zone, and key support/resistance levels. Use the built-in "
                f"indicator and price history tools. Today's date is {today}."
            ),
            context=context_base,
        )
        self.log_message("  <- Technical Analyst done.", color="blue")

        self.log_message("  -> Options Analyst querying chains and Greeks...", color="blue")
        options_result = self.agents["options_analyst"].run(
            task_prompt=(
                f"Query options chains for the symbols in the universe: "
                f"{', '.join(universe)}. Use get_option_chain_summary for each. "
                f"Evaluate liquidity, IV vs HV, and risk/reward. "
                f"Recommend specific contracts (strikes, expirations, call/put) "
                f"based on the strategy types suggested by the macro analyst. "
                f"Prefer {self.parameters['min_dte']}-{self.parameters['max_dte']} DTE. "
                f"Use get_options_market_snapshot first, then get_option_chain_summary, "
                f"then get_option_strategy_analysis to validate. Today's date is {today}."
            ),
            context=context_base,
        )
        self.log_message("  <- Options Analyst done.", color="blue")

        # ---- Phase 2: DEBATE ----

        self.log_message("[2/4] Running bull/bear debate...", color="blue")

        research_context = {
            **context_base,
            "macro_analysis": macro_result.summary or macro_result.text,
            "technical_analysis": technical_result.summary or technical_result.text,
            "options_analysis": options_result.summary or options_result.text,
        }

        self.log_message("  -> Bull Case building thesis...", color="blue")
        bull_result = self.agents["bull_case"].run(
            task_prompt=(
                f"Build the strongest possible bull case for the best options trade "
                f"identified by the research team. Use the macro, technical, and "
                f"options analysis below. Advocate with conviction — reference "
                f"specific data points, Greeks, and risk/reward numbers. "
                f"Engage with potential bear objections preemptively."
            ),
            context=research_context,
        )
        self.log_message("  <- Bull Case done.", color="blue")

        self.log_message("  -> Bear Case stress-testing...", color="blue")
        bear_result = self.agents["bear_case"].run(
            task_prompt=(
                f"Stress-test the options trade proposed by the research team. "
                f"Find every risk, failure mode, and reason to pass. "
                f"Challenge assumptions in the bull thesis. "
                f"Check for: earnings risk, volatility crush, liquidity traps, "
                f"correlation breaks, tail risk, and sizing concerns. "
                f"Directly reference the bull case below and counter its arguments. "
                f"Use specific numbers from the research — where is the thesis weakest?"
            ),
            context={
                **research_context,
                "bull_case": bull_result.summary or bull_result.text,
            },
        )
        self.log_message("  <- Bear Case done.", color="blue")

        # ---- Phase 3: RISK REVIEW ----

        self.log_message("[3/4] Running risk assessment...", color="blue")
        risk_result = self.agents["risk_manager"].run(
            task_prompt=(
                f"Assess portfolio-level risk for the proposed options trade. "
                f"Use the get_portfolio_greeks_summary tool to get current exposure. "
                f"Evaluate: correlation risk, Greek impact, concentration, drawdown, "
                f"sizing, and whether max loss is acceptable for the account. "
                f"If no current positions, evaluate on a standalone basis. "
                f"Output: APPROVE, REDUCE SIZE (specify new size), or REJECT. "
                f"Max loss per trade should not exceed "
                f"{self.parameters['max_loss_per_trade_pct']}% of portfolio."
            ),
            context={
                **research_context,
                "bull_case": bull_result.summary or bull_result.text,
                "bear_case": bear_result.summary or bear_result.text,
            },
        )
        self.log_message("  <- Risk Manager done.", color="blue")

        # ---- Phase 4: DECISION + EXECUTE ----

        self.log_message("[4/4] Portfolio Manager making final decision...", color="blue")
        pm_result = self.agents["portfolio_manager"].run(
            task_prompt=(
                f"Synthesize ALL research, debate, and risk analysis into a final "
                f"decision: TRADE or PASS.\n\n"
                f"If TRADE: Be specific about exact strikes, expirations, quantities, "
                f"and limit prices for each leg. Use defined-risk strategies. "
                f"Max loss {self.parameters['max_loss_per_trade_pct']}% of portfolio. "
                f"Submit orders directly using submit_order.\n\n"
                f"If PASS: Explain why and what would change your mind.\n\n"
                f"Universe: {', '.join(universe)}. Today: {today}."
            ),
            context={
                **research_context,
                "bull_case": bull_result.summary or bull_result.text,
                "bear_case": bear_result.summary or bear_result.text,
                "risk_assessment": risk_result.summary or risk_result.text,
            },
        )
        decision_text = pm_result.summary or pm_result.text
        self.log_message(f"  <- Portfolio Manager decision:\n{decision_text}", color="green")

        # Notify via Telegram + outbound notification
        is_trade = "FINAL DECISION: TRADE" in decision_text

        if is_trade:
            self.telegram_bot.send_message(
                f"<b>\U0001f4c8 Trade Executed</b>\n\n{decision_text[:1500]}"
            )
            self.notify(
                title="Options Debate — TRADE",
                message=decision_text[:800],
                severity="info",
            )
            self.memory.remember_decision(
                f"EXECUTED: {decision_text[:300]}",
                symbol=",".join(universe),
                action="buy",
            )
        else:
            self.telegram_bot.send_message(
                f"<b>\U0001f4ed PASS — No Trade Today</b>\n\n{decision_text[:1000]}"
            )
            self.notify(
                title="Options Debate — PASS",
                message=f"No trade today.\n\n{decision_text[:500]}",
                severity="info",
            )
            self.memory.remember_decision(
                f"PASS: {decision_text[:300]}",
                symbol=",".join(universe),
                action="hold",
            )

        self.log_message(f"=== Options Debate Cycle Complete: {today} ===", color="yellow")


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    IS_BACKTESTING = os.environ.get("IS_BACKTESTING", "false").lower() in ("true", "1", "yes")

    if IS_BACKTESTING:
        from lumibot.backtesting import YahooDataBacktesting

        AITradingTeamOptionsDebateStrategy.backtest(
            YahooDataBacktesting,
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
            benchmark_asset="SPY",
            parameters={
                "universe": ["SPY", "QQQ"],
            },
        )
    else:
        from lumibot.brokers import Tradier
        from lumibot.traders import Trader

        tradier_config = {
            "ACCESS_TOKEN": os.environ["TRADIER_ACCESS_TOKEN"],
            "ACCOUNT_NUMBER": os.environ["TRADIER_ACCOUNT_NUMBER"],
            "PAPER": os.environ.get("TRADIER_PAPER", "true").lower() != "false",
        }

        broker = Tradier(tradier_config)
        strategy = AITradingTeamOptionsDebateStrategy(
            broker=broker,
            parameters={
                "universe": os.environ.get("TRADING_UNIVERSE", "SPY,QQQ,IWM").split(","),
                "max_loss_per_trade_pct": float(os.environ.get("MAX_LOSS_PER_TRADE_PCT", "5.0")),
            },
        )

        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
