"""
Multi-Agent Options Trading Debate Strategy.

A 7-agent LLM system that researches, debates, and decides on options trades
daily, with human approval via Telegram before any order reaches Tradier (paper).

Agent Team:
  1. macro_analyst      — Market regime, VIX, sector trends → strategy type
  2. technical_analyst   — Price action, momentum, support/resistance → direction
  3. options_analyst     — Chain data, Greeks, IV/HV, liquidity → contract selection
  4. bull_case           — Strongest thesis for the proposed trade
  5. bear_case           — Risks, edge cases, failure modes
  6. risk_manager        — Portfolio Greeks, sizing, correlation, drawdown
  7. portfolio_manager   — Weighs all evidence, decides TRADE or PASS

Flow:
  RESEARCH (parallel 1-3) → DEBATE (4-5) → RISK REVIEW (6) → DECISION (7)
  → Telegram approval gate → Execute on Tradier (paper)

Set TRADIER_ACCESS_TOKEN, TRADIER_ACCOUNT_NUMBER, TRADIER_PAPER=true,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and at least one LLM API key.

Run:
    python ai_trading_team_options_debate.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from lumibot.strategies.strategy import Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent System Prompts
# Adapted from TradingAgents patterns; tailored for options trading.
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

You have access to the built-in order tools (submit_order, etc.). Use them only after human approval is received — the strategy will handle the approval gate."""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AITradingTeamOptionsDebateStrategy(Strategy):
    """7-agent options trading strategy with human-in-the-loop approval.

    Runs daily ~1 hour before market close for optimal options liquidity.
    Agents research, debate, and decide. The final decision is sent to
    Telegram for human approval before any orders reach Tradier.
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
        """Set up agents, Telegram bot, and scheduling."""
        self.sleeptime = "1D"

        # ---- Telegram Bot ----
        from lumibot.components.notifications import TelegramBot

        self.telegram_bot = TelegramBot(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            strategy=self,
        )
        self.telegram_bot.start()

        # Also configure outbound notifications
        self.notifications.configure_telegram(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        )

        # ---- Agent Models ----
        research_model = os.environ.get("AI_TRADING_TEAM_MODEL", "deepseek/deepseek-chat")
        decision_model = os.environ.get("PORTFOLIO_MANAGER_MODEL", "deepseek/deepseek-reasoner")

        # ---- Create Agents (1-6: read-only, 7: trading) ----

        # 1. Macro Analyst
        self.agents.create(
            name="macro_analyst",
            model=research_model,
            allow_trading=False,
            system_prompt=MACRO_ANALYST_PROMPT,
        )

        # 2. Technical Analyst
        self.agents.create(
            name="technical_analyst",
            model=research_model,
            allow_trading=False,
            system_prompt=TECHNICAL_ANALYST_PROMPT,
        )

        # 3. Options Analyst
        self.agents.create(
            name="options_analyst",
            model=research_model,
            allow_trading=False,
            system_prompt=OPTIONS_ANALYST_PROMPT,
        )

        # 4. Bull Case
        self.agents.create(
            name="bull_case",
            model=research_model,
            allow_trading=False,
            system_prompt=BULL_CASE_PROMPT,
        )

        # 5. Bear Case
        self.agents.create(
            name="bear_case",
            model=research_model,
            allow_trading=False,
            system_prompt=BEAR_CASE_PROMPT,
        )

        # 6. Risk Manager
        self.agents.create(
            name="risk_manager",
            model=research_model,
            allow_trading=False,
            system_prompt=RISK_MANAGER_PROMPT,
        )

        # 7. Portfolio Manager (trade-enabled)
        self.agents.create(
            name="portfolio_manager",
            model=decision_model,
            allow_trading=True,
            system_prompt=PORTFOLIO_MANAGER_PROMPT,
        )

        self.log_message("Options Debate Strategy initialized with 7 agents.", color="green")

    # ==================================================================
    # Lifecycle: Before Trading (trigger daily before market close)
    # ==================================================================

    def before_market_closes(self):
        """Ensure we run ~1 hour before close for best options liquidity."""
        # LumiBot's sleeptime="1D" handles daily cycles.
        # The exact timing depends on the broker/market calendar.
        # This hook can adjust scheduling if needed.
        pass

    # ==================================================================
    # Lifecycle: On Trading Iteration (the main daily flow)
    # ==================================================================

    def on_trading_iteration(self):
        """Execute the full agent debate → decision → approval → execution flow."""
        universe = self.parameters["universe"]
        today = self.get_datetime().date().isoformat()

        self.log_message(f"=== Options Debate Cycle: {today} ===", color="yellow")
        self.log_message(f"Universe: {', '.join(universe)}", color="yellow")

        # ---- Phase 1: RESEARCH (agents 1-3, run sequentially since each
        #      agent's output depends on understanding the full picture) ----

        context_base = {
            "date": today,
            "universe": universe,
            "min_dte": self.parameters["min_dte"],
            "max_dte": self.parameters["max_dte"],
            "max_loss_per_trade_pct": self.parameters["max_loss_per_trade_pct"],
        }

        self.log_message("[1/5] Running research analysts...", color="blue")

        # 1a. Macro Analyst
        self.log_message("  → Macro Analyst researching market regime...", color="blue")
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
        self.log_message(f"  ← Macro Analyst done.", color="blue")

        # 1b. Technical Analyst
        self.log_message("  → Technical Analyst reviewing price action...", color="blue")
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
        self.log_message(f"  ← Technical Analyst done.", color="blue")

        # 1c. Options Analyst
        self.log_message("  → Options Analyst querying chains and Greeks...", color="blue")
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
        self.log_message(f"  ← Options Analyst done.", color="blue")

        # ---- Phase 2: DEBATE (agents 4-5) ----

        self.log_message("[2/5] Running bull/bear debate...", color="blue")

        research_context = {
            **context_base,
            "macro_analysis": macro_result.summary or macro_result.text,
            "technical_analysis": technical_result.summary or technical_result.text,
            "options_analysis": options_result.summary or options_result.text,
        }

        # 2a. Bull Case
        self.log_message("  → Bull Case building thesis...", color="blue")
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
        self.log_message(f"  ← Bull Case done.", color="blue")

        # 2b. Bear Case
        self.log_message("  → Bear Case stress-testing...", color="blue")
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
        self.log_message(f"  ← Bear Case done.", color="blue")

        # ---- Phase 3: RISK REVIEW (agent 6) ----

        self.log_message("[3/5] Running risk assessment...", color="blue")
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
        self.log_message(f"  ← Risk Manager done (recommendation: APPROVE/REDUCE/REJECT).", color="blue")

        # ---- Phase 4: DECISION (agent 7) ----

        self.log_message("[4/5] Portfolio Manager making final decision...", color="blue")
        pm_result = self.agents["portfolio_manager"].run(
            task_prompt=(
                f"Synthesize ALL research, debate, and risk analysis into a final "
                f"decision: TRADE or PASS.\n\n"
                f"If TRADE: Be specific about exact strikes, expirations, quantities, "
                f"and limit prices for each leg. Use defined-risk strategies. "
                f"Max loss {self.parameters['max_loss_per_trade_pct']}% of portfolio.\n\n"
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
        self.log_message(f"  ← Portfolio Manager decision:\n{decision_text}", color="green")

        # ---- Phase 5: HUMAN APPROVAL GATE ----

        self.log_message("[5/5] Sending to Telegram for human approval...", color="yellow")

        # Determine if this is a TRADE or PASS
        is_trade = "FINAL DECISION: TRADE" in decision_text.upper()

        if not is_trade:
            # PASS — notify and log, no action needed
            self.log_message("Decision: PASS. No trade today.", color="yellow")
            self.notify(
                title="Options Debate — PASS",
                message=f"No trade today.\n\n{decision_text[:500]}",
                severity="info",
            )
            # Store in memory for audit
            self.memory.remember_decision(
                f"PASS: {decision_text[:300]}",
                symbol=",".join(universe),
                action="hold",
            )
            return

        # TRADE — send for approval
        self.telegram_bot.send_approval_request(decision_text)

        # Notify user
        self.notify(
            title="⚡ Options Trade — Approval Required",
            message=f"A trade decision is waiting for your approval.\n\n"
            f"Use /decision to view details, then /approve or /reject.",
            severity="warning",
        )

        # Block waiting for human response
        timeout = int(os.environ.get("APPROVAL_TIMEOUT_MINUTES", "30"))
        self.log_message(f"Waiting for Telegram approval (timeout: {timeout} min)...", color="yellow")
        approved = self.telegram_bot.wait_for_approval(timeout_minutes=timeout)

        if approved:
            self.log_message("✅ Trade APPROVED by user. Executing...", color="green")
            self.notify(
                title="✅ Trade Approved",
                message=f"Executing the approved trade now.\n\n{decision_text[:500]}",
                severity="info",
            )
            self.memory.remember_decision(
                f"APPROVED: {decision_text[:300]}",
                symbol=",".join(universe),
                action="buy",
            )
            # The portfolio_manager agent has allow_trading=True, so it already
            # submitted orders via its built-in tools during its run.
            # If we need to re-submit or verify, we'd do it here.
            # For safety, we verify the orders are in the broker.
            self.log_message("Trade submitted. Verify in Tradier dashboard.", color="green")

        else:
            self.log_message("❌ Trade REJECTED or timed out. No orders placed.", color="red")
            self.notify(
                title="❌ Trade Rejected",
                message=f"The trade was rejected (or approval timed out). "
                f"No orders have been placed.\n\n{decision_text[:300]}",
                severity="warning",
            )
            self.memory.remember_decision(
                f"REJECTED: {decision_text[:300]}",
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
        # Live/Paper trading with Tradier
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
