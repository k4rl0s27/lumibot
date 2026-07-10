"""Minimal backtest runner for dry-run validation."""
import os
import sys

# Fix: pip's truststore hijacks ssl.SSLContext on Windows, breaking async DNS.
# Restore the real SSLContext from the stdlib before any litellm/httpx imports.
import ssl
if "pip._vendor.truststore" in sys.modules:
    ssl.SSLContext = ssl.create_default_context().__class__

os.environ["IS_BACKTESTING"] = "true"

# Ensure the project root is importable (example_strategies lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from example_strategies.ai_trading_team_options_debate import AITradingTeamOptionsDebateStrategy
from lumibot.backtesting import YahooDataBacktesting

print("Starting backtest dry run...")
AITradingTeamOptionsDebateStrategy.backtest(
    YahooDataBacktesting,
    datetime(2026, 7, 1),
    datetime(2026, 7, 3),
    benchmark_asset="SPY",
    parameters={"universe": ["SPY"]},
)
print("Backtest complete.")
