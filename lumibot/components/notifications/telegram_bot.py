"""
Bidirectional Telegram bot for LumiBot strategies.

Provides a two-way Telegram interface with:
- Command handlers for portfolio queries (/status, /portfolio, /positions, /history)
- Trade approval flow with inline keyboard [Approve] [Reject] buttons
- Thread-safe communication between bot thread and strategy thread via threading.Event

Usage in a strategy:
    from lumibot.components.notifications import TelegramBot

    def initialize(self):
        self.telegram_bot = TelegramBot(
            bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
            chat_id=os.environ["TELEGRAM_CHAT_ID"],
            strategy=self,
        )
        self.telegram_bot.start()

    def on_trading_iteration(self):
        # ... agents debate, produce a decision ...
        self.telegram_bot.send_approval_request(decision_text)
        approved = self.telegram_bot.wait_for_approval(timeout_minutes=30)
        if approved:
            self.submit_orders(orders)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approval result
# ---------------------------------------------------------------------------


@dataclass
class ApprovalResult:
    """Result of a human approval request."""

    approved: bool
    decided_by: str = "timeout"  # "user" or "timeout"
    user_response: str | None = None  # optional note from user
    decided_at: datetime | None = None


# ---------------------------------------------------------------------------
# Pending decision (stored for /decision and /approve /reject commands)
# ---------------------------------------------------------------------------


@dataclass
class PendingDecision:
    """A trade decision awaiting human approval."""

    text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: int | None = None


# ---------------------------------------------------------------------------
# Telegram Bot
# ---------------------------------------------------------------------------


class TelegramBot:
    """Bidirectional Telegram bot with command handlers and approval gate.

    Runs in a background thread using python-telegram-bot's polling mechanism.
    Communicates with the strategy thread via thread-safe events and locks.

    Parameters
    ----------
    bot_token : str
        Telegram bot token from @BotFather.
    chat_id : str
        Target chat ID for outgoing messages.
    strategy : Any, optional
        Reference to the LumiBot Strategy instance. Set later via set_strategy()
        if not available at construction time.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        strategy: Any = None,
    ) -> None:
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._strategy = strategy

        # Threading primitives for approval gate
        self._approval_event = threading.Event()
        self._approval_lock = threading.Lock()
        self._approval_decision: ApprovalResult | None = None

        # Pending decision storage
        self._pending_decision: PendingDecision | None = None

        # Decision history (in-memory, last N)
        self._decision_history: list[dict[str, Any]] = []
        self._max_history = 50

        # Bot state
        self._application: Any = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Track known chat IDs for access control
        self._authorized_chat_ids: set[str] = set()
        if self.chat_id:
            self._authorized_chat_ids.add(str(self.chat_id))

    # ------------------------------------------------------------------
    # Strategy reference
    # ------------------------------------------------------------------

    def set_strategy(self, strategy: Any) -> None:
        """Set or update the strategy reference for command handlers."""
        self._strategy = strategy

    @property
    def strategy(self) -> Any:
        """Get the current strategy reference."""
        return self._strategy

    # ------------------------------------------------------------------
    # Lifecycle: start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the bot in a background thread."""
        if self._running:
            logger.warning("TelegramBot is already running.")
            return

        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN is not set. TelegramBot will not start.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_bot, daemon=True, name="telegram-bot")
        self._thread.start()
        logger.info("TelegramBot started in background thread.")

    def stop(self) -> None:
        """Stop the bot gracefully."""
        self._running = False
        if self._application is not None:
            try:
                # Schedule shutdown on the event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._application.shutdown(), loop=loop)
            except Exception:
                pass
        logger.info("TelegramBot stopped.")

    def _run_bot(self) -> None:
        """Internal: run the bot polling loop in a background thread."""
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, CallbackQueryHandler
        except ImportError:
            logger.error(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )
            self._running = False
            return

        # Build the application
        self._application = Application.builder().token(self.bot_token).build()

        # Register command handlers
        self._application.add_handler(CommandHandler("start", self._cmd_start))
        self._application.add_handler(CommandHandler("help", self._cmd_help))
        self._application.add_handler(CommandHandler("status", self._cmd_status))
        self._application.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        self._application.add_handler(CommandHandler("positions", self._cmd_positions))
        self._application.add_handler(CommandHandler("decision", self._cmd_decision))
        self._application.add_handler(CommandHandler("approve", self._cmd_approve))
        self._application.add_handler(CommandHandler("reject", self._cmd_reject))
        self._application.add_handler(CommandHandler("history", self._cmd_history))

        # Register callback query handler (inline button presses)
        self._application.add_handler(CallbackQueryHandler(self._handle_callback))

        logger.info("TelegramBot polling started. Waiting for messages...")
        self._application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=[])

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _is_authorized(self, chat_id: str | int) -> bool:
        """Check if a chat_id is authorized to interact with this bot."""
        # If no authorized chats are configured, allow all (first-user-wins)
        if not self._authorized_chat_ids:
            return True
        return str(chat_id) in self._authorized_chat_ids

    def _ensure_authorized(self, chat_id: str | int) -> bool:
        """Check authorization and send a warning if not authorized."""
        if self._is_authorized(chat_id):
            return True
        # We can't easily reply here since we don't have the update object
        logger.warning(f"Unauthorized access attempt from chat_id={chat_id}")
        return False

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    async def _send_message_async(
        self,
        text: str,
        reply_markup: Any = None,
        parse_mode: str = "HTML",
    ) -> Any:
        """Send a message to the configured chat asynchronously."""
        if not self._application or not self.chat_id:
            return None
        try:
            return await self._application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return None

    def send_message(
        self,
        text: str,
        reply_markup: Any = None,
        parse_mode: str = "HTML",
    ) -> None:
        """Send a message to the configured chat (thread-safe)."""
        if not self._application:
            logger.warning("TelegramBot not started. Cannot send message.")
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            asyncio.ensure_future(self._send_message_async(text, reply_markup, parse_mode), loop=loop)
        else:
            loop.run_until_complete(self._send_message_async(text, reply_markup, parse_mode))

    # ------------------------------------------------------------------
    # Approval gate
    # ------------------------------------------------------------------

    def send_approval_request(self, decision_text: str) -> None:
        """Send a trade decision to Telegram with [Approve] [Reject] buttons.

        The strategy calls this after agents produce a decision.
        Then it calls wait_for_approval() to block until the user responds.

        Parameters
        ----------
        decision_text : str
            The trade decision summary to present to the user.
        """
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        # Store the pending decision
        self._pending_decision = PendingDecision(text=decision_text)

        # Reset approval state
        with self._approval_lock:
            self._approval_event.clear()
            self._approval_decision = None

        # Build inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Add to decision history
        self._decision_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": decision_text,
            "status": "pending",
        })
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]

        # Send the message
        timeout = int(os.environ.get("APPROVAL_TIMEOUT_MINUTES", "30"))
        full_text = (
            f"<b>🤖 Trade Decision — Approval Required</b>\n\n"
            f"{decision_text}\n\n"
            f"<i>⏰ Auto-reject in {timeout} minutes if no response.</i>\n"
            f"<i>Use /approve or /reject, or tap a button below.</i>"
        )
        self.send_message(full_text, reply_markup=reply_markup)

    def wait_for_approval(self, timeout_minutes: int | None = None) -> bool:
        """Block until the user approves or rejects the pending decision.

        Parameters
        ----------
        timeout_minutes : int, optional
            Maximum minutes to wait. Defaults to APPROVAL_TIMEOUT_MINUTES env var
            (30 minutes if not set).

        Returns
        -------
        bool
            True if approved, False if rejected or timed out.
        """
        if timeout_minutes is None:
            timeout_minutes = int(os.environ.get("APPROVAL_TIMEOUT_MINUTES", "30"))

        timeout_seconds = timeout_minutes * 60
        logger.info(f"Waiting for approval (timeout: {timeout_minutes} minutes)...")

        # Block on the event
        signaled = self._approval_event.wait(timeout=timeout_seconds)

        with self._approval_lock:
            if not signaled:
                # Timeout — auto-reject
                self._approval_decision = ApprovalResult(
                    approved=False,
                    decided_by="timeout",
                    decided_at=datetime.now(timezone.utc),
                )
                logger.warning("Approval timed out — auto-rejecting.")
                if self._pending_decision:
                    self._decision_history[-1]["status"] = "rejected (timeout)"
                # Notify user
                self.send_message("⏰ <b>Approval timed out.</b> Trade automatically rejected.")

            result = self._approval_decision

        return result.approved if result else False

    def _resolve_approval(self, approved: bool, user_note: str | None = None) -> None:
        """Resolve the pending approval (called from command/callback handlers)."""
        with self._approval_lock:
            if self._approval_event.is_set():
                # Already decided
                return
            self._approval_decision = ApprovalResult(
                approved=approved,
                decided_by="user",
                user_response=user_note,
                decided_at=datetime.now(timezone.utc),
            )
            if self._pending_decision:
                status = "approved" if approved else "rejected"
                self._decision_history[-1]["status"] = status
            self._approval_event.set()

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Any, context: Any) -> None:
        """Handle /start command."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return

        welcome = (
            "👋 <b>LumiBot Options Trading System</b>\n\n"
            "I monitor your options trading strategy and keep you in control.\n\n"
            "<b>Available commands:</b>\n"
            "/status — Portfolio summary (cash, value, P&amp;L)\n"
            "/portfolio — Full portfolio breakdown with Greeks\n"
            "/positions — List all open positions\n"
            "/decision — Show the latest pending trade decision\n"
            "/approve — Approve the pending trade\n"
            "/reject — Reject the pending trade\n"
            "/history — Recent trade decision history\n"
            "/help — Show this message"
        )
        await update.message.reply_text(welcome, parse_mode="HTML")

    async def _cmd_help(self, update: Any, context: Any) -> None:
        """Handle /help command — same as /start."""
        await self._cmd_start(update, context)

    async def _cmd_status(self, update: Any, context: Any) -> None:
        """Handle /status command — portfolio summary."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        status_text = self._build_status_text()
        await update.message.reply_text(status_text, parse_mode="HTML")

    async def _cmd_portfolio(self, update: Any, context: Any) -> None:
        """Handle /portfolio command — full portfolio breakdown."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        portfolio_text = self._build_portfolio_text()
        await update.message.reply_text(portfolio_text, parse_mode="HTML")

    async def _cmd_positions(self, update: Any, context: Any) -> None:
        """Handle /positions command — list open positions."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        positions_text = self._build_positions_text()
        await update.message.reply_text(positions_text, parse_mode="HTML")

    async def _cmd_decision(self, update: Any, context: Any) -> None:
        """Handle /decision command — show the latest pending decision."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        if self._pending_decision is None or self._approval_event.is_set():
            await update.message.reply_text(
                "📭 <b>No pending decision.</b>\n\n"
                "The strategy runs daily. You'll receive a decision here when the agents "
                "have completed their analysis.",
                parse_mode="HTML",
            )
            return

        timeout = int(os.environ.get("APPROVAL_TIMEOUT_MINUTES", "30"))
        elapsed = (datetime.now(timezone.utc) - self._pending_decision.created_at).total_seconds() / 60
        remaining = max(0, timeout - elapsed)

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = [
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        text = (
            f"<b>🤖 Pending Trade Decision</b>\n\n"
            f"{self._pending_decision.text}\n\n"
            f"<i>⏰ {remaining:.0f} minutes remaining before auto-reject.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=reply_markup)

    async def _cmd_approve(self, update: Any, context: Any) -> None:
        """Handle /approve command."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        if self._pending_decision is None or self._approval_event.is_set():
            await update.message.reply_text("📭 No pending decision to approve.")
            return

        self._resolve_approval(approved=True)
        await update.message.reply_text("✅ <b>Trade approved!</b> Executing now...", parse_mode="HTML")

    async def _cmd_reject(self, update: Any, context: Any) -> None:
        """Handle /reject command."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        if self._pending_decision is None or self._approval_event.is_set():
            await update.message.reply_text("📭 No pending decision to reject.")
            return

        self._resolve_approval(approved=False)
        await update.message.reply_text("❌ <b>Trade rejected.</b> No orders will be placed.", parse_mode="HTML")

    async def _cmd_history(self, update: Any, context: Any) -> None:
        """Handle /history command — recent decision history."""
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        if not self._decision_history:
            await update.message.reply_text("📭 No decision history yet.")
            return

        # Show last 10 decisions
        recent = self._decision_history[-10:]
        lines = ["<b>📋 Recent Trade Decisions</b>\n"]
        for i, decision in enumerate(reversed(recent), 1):
            status_emoji = {"approved": "✅", "rejected": "❌", "pending": "⏳", "rejected (timeout)": "⏰"}.get(
                decision.get("status", "unknown"), "❓"
            )
            ts = decision.get("timestamp", "?")[:16].replace("T", " ")
            # Truncate text for history display
            text = decision.get("text", "")
            if len(text) > 150:
                text = text[:147] + "..."
            lines.append(f"{status_emoji} <b>#{i}</b> <i>{ts}</i>\n{text}\n")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _handle_callback(self, update: Any, context: Any) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        chat_id = query.message.chat.id if query.message else None

        if chat_id and not self._ensure_authorized(chat_id):
            await query.answer("⛔ Unauthorized.", show_alert=True)
            return

        await query.answer()

        data = query.data

        if data == "approve":
            if self._pending_decision is None or self._approval_event.is_set():
                await query.edit_message_text(
                    query.message.text + "\n\n<i>⚠️ This decision is no longer pending.</i>",
                    parse_mode="HTML",
                )
                return
            self._resolve_approval(approved=True)
            await query.edit_message_text(
                query.message.text + "\n\n✅ <b>APPROVED</b> — Executing trade...",
                parse_mode="HTML",
            )

        elif data == "reject":
            if self._pending_decision is None or self._approval_event.is_set():
                await query.edit_message_text(
                    query.message.text + "\n\n<i>⚠️ This decision is no longer pending.</i>",
                    parse_mode="HTML",
                )
                return
            self._resolve_approval(approved=False)
            await query.edit_message_text(
                query.message.text + "\n\n❌ <b>REJECTED</b> — No orders will be placed.",
                parse_mode="HTML",
            )

    # ------------------------------------------------------------------
    # Portfolio data helpers
    # ------------------------------------------------------------------

    def _get_strategy_safe(self) -> Any:
        """Get the strategy reference, returning None if not available."""
        return self._strategy

    def _build_status_text(self) -> str:
        """Build a status summary string from strategy data."""
        strategy = self._get_strategy_safe()
        if strategy is None:
            return "⚠️ Strategy not connected yet."

        try:
            cash = strategy.get_cash()
            portfolio_value = strategy.get_portfolio_value()
            positions = strategy.get_positions()

            unrealized_pl = sum(
                getattr(p, "pnl", 0) or 0
                for p in (positions or [])
            )
            pnl_pct = (unrealized_pl / portfolio_value * 100) if portfolio_value and portfolio_value > 0 else 0

            now = strategy.get_datetime()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S %Z") if now else "unknown"

            return (
                f"<b>📊 Portfolio Status</b>\n\n"
                f"<b>Time:</b> {now_str}\n"
                f"<b>Cash:</b> ${cash:,.2f}\n"
                f"<b>Portfolio Value:</b> ${portfolio_value:,.2f}\n"
                f"<b>Unrealized P&amp;L:</b> ${unrealized_pl:,.2f} ({pnl_pct:+.1f}%)\n"
                f"<b>Open Positions:</b> {len(positions) if positions else 0}\n\n"
                f"<i>Use /portfolio for full breakdown or /positions for details.</i>"
            )
        except Exception as e:
            logger.error(f"Error building status: {e}")
            return f"⚠️ Error fetching portfolio data: {e}"

    def _build_portfolio_text(self) -> str:
        """Build a full portfolio breakdown string."""
        strategy = self._get_strategy_safe()
        if strategy is None:
            return "⚠️ Strategy not connected yet."

        try:
            cash = strategy.get_cash()
            portfolio_value = strategy.get_portfolio_value()
            positions = strategy.get_positions() or []

            lines = [
                "<b>💼 Portfolio Breakdown</b>\n",
                f"<b>Cash:</b> ${cash:,.2f}",
                f"<b>Total Value:</b> ${portfolio_value:,.2f}\n",
            ]

            if not positions:
                lines.append("<i>No open positions.</i>")
            else:
                for i, pos in enumerate(positions, 1):
                    symbol = getattr(pos, "symbol", "?")
                    asset_type = getattr(getattr(pos, "asset", None), "asset_type", "stock")
                    qty = getattr(pos, "quantity", 0) or 0
                    market_value = getattr(pos, "market_value", 0) or 0
                    unrealized_pl = getattr(pos, "pnl", 0) or 0
                    pnl_pct = (unrealized_pl / (market_value - unrealized_pl) * 100) if (market_value - unrealized_pl) != 0 else 0

                    type_tag = ""
                    if asset_type and "option" in str(asset_type).lower():
                        right = getattr(getattr(pos, "asset", None), "right", "")
                        strike = getattr(getattr(pos, "asset", None), "strike", "")
                        exp = getattr(getattr(pos, "asset", None), "expiration", "")
                        exp_str = str(exp)[:10] if exp else ""
                        type_tag = f" [{right} {strike} {exp_str}]"

                    lines.append(
                        f"<b>#{i}</b> {symbol}{type_tag} — {qty} shares "
                        f"| Value: ${market_value:,.2f} "
                        f"| P&amp;L: ${unrealized_pl:,.2f} ({pnl_pct:+.1f}%)"
                    )

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error building portfolio: {e}")
            return f"⚠️ Error fetching portfolio data: {e}"

    def _build_positions_text(self) -> str:
        """Build a detailed positions list string."""
        strategy = self._get_strategy_safe()
        if strategy is None:
            return "⚠️ Strategy not connected yet."

        try:
            positions = strategy.get_positions() or []

            if not positions:
                return "📭 <b>No open positions.</b>"

            lines = [f"<b>📋 Open Positions ({len(positions)})</b>\n"]

            for i, pos in enumerate(positions, 1):
                symbol = getattr(pos, "symbol", "?")
                qty = getattr(pos, "quantity", 0) or 0
                avg_price = getattr(pos, "avg_fill_price", None)
                last_price = getattr(pos, "last_price", None)
                market_value = getattr(pos, "market_value", 0) or 0
                unrealized_pl = getattr(pos, "pnl", 0) or 0

                asset = getattr(pos, "asset", None)
                asset_type = getattr(asset, "asset_type", "stock") if asset else "unknown"

                lines.append(f"<b>#{i} {symbol}</b>")
                lines.append(f"   Type: {asset_type} | Qty: {qty}")
                if avg_price:
                    lines.append(f"   Avg Entry: ${avg_price:.2f}")
                if last_price:
                    lines.append(f"   Last: ${last_price:.2f}")
                lines.append(f"   Value: ${market_value:,.2f} | P&amp;L: ${unrealized_pl:+,.2f}")

                # Option-specific details
                if asset and "option" in str(asset_type).lower():
                    right = getattr(asset, "right", "")
                    strike = getattr(asset, "strike", "")
                    exp = getattr(asset, "expiration", "")
                    exp_str = str(exp)[:10] if exp else ""
                    lines.append(f"   Option: {right} {strike} Exp: {exp_str}")

                lines.append("")  # blank line between positions

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error building positions: {e}")
            return f"⚠️ Error fetching positions: {e}"
