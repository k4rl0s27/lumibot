"""
Markdown run logger for LumiBot strategies.

Writes a single .md file per trading cycle to a configurable directory,
capturing all agent outputs and errors in a format easily read by AI assistants.

Usage:
    from lumibot.components.notifications import RunLogger

    def initialize(self):
        self.run_logger = RunLogger(
            log_dir=os.environ.get("RUN_LOG_DIR", "./logs"),
            strategy_name=self.__class__.__name__,
            telegram_bot=self.telegram_bot,
        )

    def on_trading_iteration(self):
        self.run_logger.start_cycle(today, universe, params)
        macro = self.agents["macro_analyst"].run(...)
        self.run_logger.log_agent_output("research", "macro_analyst", macro.summary or macro.text)
        ...
        self.run_logger.finalize(decision_text, is_trade=...)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunLogger:
    """Writes a markdown audit trail for each strategy cycle."""

    def __init__(
        self,
        log_dir: str = "./logs",
        strategy_name: str = "",
        telegram_bot: Any = None,
    ) -> None:
        self._log_dir = Path(log_dir) / strategy_name
        self._telegram_bot = telegram_bot
        self._lines: list[str] = []
        self._errors: list[str] = []
        self._file_path: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_cycle(
        self,
        date: str,
        universe: list[str],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Begin a new cycle log. Call once at the start of on_trading_iteration."""
        self._lines = []
        self._errors = []

        now = datetime.now(timezone.utc)
        cycle_id = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._log_dir / f"{date}_{cycle_id}.md"

        self._lines.append(f"# Options Debate Cycle — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        self._lines.append(f"**Cycle ID:** `{cycle_id}`")
        self._lines.append(f"**Date:** {date}")
        self._lines.append(f"**Universe:** {', '.join(universe) if universe else 'none'}")
        if params:
            self._lines.append(f"**Parameters:** {_format_params(params)}")
        self._lines.append("")

    def log_agent_output(self, phase: str, agent_name: str, output: str | None) -> None:
        """Log an agent's output to the current cycle. Call after each agent.run()."""
        if not output:
            self._lines.append(f"## {phase} — {agent_name}\n\n*(no output)*\n")
            return
        self._lines.append(f"## {phase} — {agent_name}")
        self._lines.append("")
        self._lines.append(output.strip())
        self._lines.append("")

    def log_error(self, phase: str, agent_name: str, error: str) -> None:
        """Log an error that occurred during a phase."""
        self._errors.append(f"- **{phase} / {agent_name}:** {error}")
        self._lines.append(f"## {phase} — {agent_name} \u26a0 ERROR")
        self._lines.append("")
        self._lines.append(f"```\n{error}\n```")
        self._lines.append("")

    def finalize(
        self,
        decision_text: str,
        is_trade: bool = False,
    ) -> Path | None:
        """Write the .md file and optionally send a Telegram notification."""
        if self._errors:
            self._lines.append("## Errors")
            self._lines.append("")
            for err in self._errors:
                self._lines.append(err)
            self._lines.append("")

        if self._file_path is None:
            return None

        self._file_path.write_text("\n".join(self._lines), encoding="utf-8")
        logger.info(f"Run log written to {self._file_path}")

        # Telegram notification
        if self._telegram_bot is not None:
            try:
                if is_trade:
                    self._telegram_bot.send_message(
                        f"<b>\U0001f4c8 Trade Executed</b>\n\n{decision_text[:1500]}"
                    )
                else:
                    self._telegram_bot.send_message(
                        f"<b>\U0001f4ed PASS \u2014 No Trade Today</b>\n\n{decision_text[:1000]}"
                    )
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

        return self._file_path


def _format_params(params: dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, float):
            v = f"{v}%"
        parts.append(f"{k}={v}")
    return ", ".join(parts)
