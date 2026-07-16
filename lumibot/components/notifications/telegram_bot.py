"""
Telegram bot for LumiBot strategy monitoring and notifications.

Provides:
- Command handlers for portfolio queries (/status, /portfolio, /positions)
- Outbound notifications via send_message() from strategy code
- Thread-safe: bot runs in a background daemon thread

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
        self.telegram_bot.send_message("Trade executed: ...")
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot with command handlers for strategy monitoring.

    Runs in a background thread using python-telegram-bot's polling mechanism.

    Parameters
    ----------
    bot_token : str
        Telegram bot token from @BotFather.
    chat_id : str
        Target chat ID for outgoing messages.
    strategy : Any, optional
        Reference to the LumiBot Strategy instance for portfolio queries.
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

        self._application: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._authorized_chat_ids: set[str] = set()
        if self.chat_id:
            self._authorized_chat_ids.add(str(self.chat_id))

    # ------------------------------------------------------------------
    # Strategy reference
    # ------------------------------------------------------------------

    def set_strategy(self, strategy: Any) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> Any:
        return self._strategy

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
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
        self._running = False
        if self._application is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._application.shutdown(), self._loop)
            except Exception:
                pass
        logger.info("TelegramBot stopped.")

    def _run_bot(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler
        except ImportError:
            logger.error(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )
            self._running = False
            return

        self._application = Application.builder().token(self.bot_token).build()
        self._loop = asyncio.get_event_loop()

        self._application.add_handler(CommandHandler("start", self._cmd_start))
        self._application.add_handler(CommandHandler("help", self._cmd_help))
        self._application.add_handler(CommandHandler("status", self._cmd_status))
        self._application.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        self._application.add_handler(CommandHandler("positions", self._cmd_positions))

        logger.info("TelegramBot polling started.")
        self._application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=[])

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _is_authorized(self, chat_id: str | int) -> bool:
        if not self._authorized_chat_ids:
            return True
        return str(chat_id) in self._authorized_chat_ids

    def _ensure_authorized(self, chat_id: str | int) -> bool:
        if self._is_authorized(chat_id):
            return True
        logger.warning(f"Unauthorized access attempt from chat_id={chat_id}")
        return False

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
    ) -> None:
        """Send a message to the configured chat (thread-safe)."""
        if not self._application or not self._loop:
            logger.warning("TelegramBot not started. Cannot send message.")
            return
        asyncio.run_coroutine_threadsafe(
            self._send_message_async(text, parse_mode),
            self._loop,
        )

    async def _send_message_async(
        self,
        text: str,
        parse_mode: str = "HTML",
    ) -> Any:
        if not self._application or not self.chat_id:
            return None
        try:
            return await self._application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return None

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Any, context: Any) -> None:
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("\u26d4 You are not authorized to use this bot.")
            return

        welcome = (
            "\U0001f44b <b>LumiBot Options Trading System</b>\n\n"
            "I monitor your options trading strategy.\n\n"
            "<b>Available commands:</b>\n"
            "/status — Portfolio summary (cash, value, P&amp;L)\n"
            "/portfolio — Full portfolio breakdown\n"
            "/positions — List all open positions\n"
            "/help — Show this message"
        )
        await update.message.reply_text(welcome, parse_mode="HTML")

    async def _cmd_help(self, update: Any, context: Any) -> None:
        await self._cmd_start(update, context)

    async def _cmd_status(self, update: Any, context: Any) -> None:
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("\u26d4 Unauthorized.")
            return

        status_text = self._build_status_text()
        await update.message.reply_text(status_text, parse_mode="HTML")

    async def _cmd_portfolio(self, update: Any, context: Any) -> None:
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("\u26d4 Unauthorized.")
            return

        portfolio_text = self._build_portfolio_text()
        await update.message.reply_text(portfolio_text, parse_mode="HTML")

    async def _cmd_positions(self, update: Any, context: Any) -> None:
        chat_id = update.effective_chat.id
        if not self._ensure_authorized(chat_id):
            await update.message.reply_text("\u26d4 Unauthorized.")
            return

        positions_text = self._build_positions_text()
        await update.message.reply_text(positions_text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # Portfolio data helpers
    # ------------------------------------------------------------------

    def _build_status_text(self) -> str:
        strategy = self._strategy
        if strategy is None:
            return "\u26a0\ufe0f Strategy not connected yet."

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
                f"<b>\U0001f4ca Portfolio Status</b>\n\n"
                f"<b>Time:</b> {now_str}\n"
                f"<b>Cash:</b> ${cash:,.2f}\n"
                f"<b>Portfolio Value:</b> ${portfolio_value:,.2f}\n"
                f"<b>Unrealized P&amp;L:</b> ${unrealized_pl:,.2f} ({pnl_pct:+.1f}%)\n"
                f"<b>Open Positions:</b> {len(positions) if positions else 0}\n\n"
                f"<i>Use /portfolio for full breakdown or /positions for details.</i>"
            )
        except Exception as e:
            logger.error(f"Error building status: {e}")
            return f"\u26a0\ufe0f Error fetching portfolio data: {e}"

    def _build_portfolio_text(self) -> str:
        strategy = self._strategy
        if strategy is None:
            return "\u26a0\ufe0f Strategy not connected yet."

        try:
            cash = strategy.get_cash()
            portfolio_value = strategy.get_portfolio_value()
            positions = strategy.get_positions() or []

            lines = [
                "<b>\U0001f4bc Portfolio Breakdown</b>\n",
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
            return f"\u26a0\ufe0f Error fetching portfolio data: {e}"

    def _build_positions_text(self) -> str:
        strategy = self._strategy
        if strategy is None:
            return "\u26a0\ufe0f Strategy not connected yet."

        try:
            positions = strategy.get_positions() or []

            if not positions:
                return "\U0001f4ed <b>No open positions.</b>"

            lines = [f"<b>\U0001f4cb Open Positions ({len(positions)})</b>\n"]

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

                if asset and "option" in str(asset_type).lower():
                    right = getattr(asset, "right", "")
                    strike = getattr(asset, "strike", "")
                    exp = getattr(asset, "expiration", "")
                    exp_str = str(exp)[:10] if exp else ""
                    lines.append(f"   Option: {right} {strike} Exp: {exp_str}")

                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error building positions: {e}")
            return f"\u26a0\ufe0f Error fetching positions: {e}"
