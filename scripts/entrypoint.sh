#!/bin/bash
# =============================================================================
# LumiBot Options Trading System — Docker Entrypoint
# =============================================================================
# Validates required environment variables and starts the strategy.
# =============================================================================

set -euo pipefail

echo "============================================"
echo " LumiBot Options Trading System"
echo " Starting up..."
echo "============================================"

# ------------------------------------------------------------------
# Helper: check if a variable is set and non-empty
# ------------------------------------------------------------------
require_env() {
    local var_name="$1"
    local value="${!var_name:-}"
    if [ -z "$value" ] || [ "$value" = "your_${var_name,,}_here" ]; then
        echo "ERROR: ${var_name} is not set or still has the placeholder value."
        echo "       Copy .env.example to .env and fill in your values."
        exit 1
    fi
}

# ------------------------------------------------------------------
# Validate required environment variables
# ------------------------------------------------------------------
echo ""
echo "[1/4] Validating environment variables..."

require_env TRADIER_ACCESS_TOKEN
require_env TRADIER_ACCOUNT_NUMBER

# Tradier paper mode safety check
if [ "${TRADIER_PAPER:-}" != "true" ]; then
    echo "ERROR: TRADIER_PAPER must be 'true' for paper trading."
    echo "       This system is designed for paper trading validation first."
    exit 1
fi

require_env TELEGRAM_BOT_TOKEN
require_env TELEGRAM_CHAT_ID

# At least one LLM provider key must be set
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: At least one LLM provider key must be set."
    echo "       Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in .env"
    exit 1
fi

echo "       All required environment variables are set."

# ------------------------------------------------------------------
# Set defaults for optional variables
# ------------------------------------------------------------------
echo ""
echo "[2/4] Setting defaults..."

export LUMIBOT_MEMORY_DIR="${LUMIBOT_MEMORY_DIR:-/data/memory}"
export APPROVAL_TIMEOUT_MINUTES="${APPROVAL_TIMEOUT_MINUTES:-30}"
export TRADING_UNIVERSE="${TRADING_UNIVERSE:-SPY,QQQ,IWM}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "$LUMIBOT_MEMORY_DIR" /data/logs

echo "       LUMIBOT_MEMORY_DIR=${LUMIBOT_MEMORY_DIR}"
echo "       APPROVAL_TIMEOUT_MINUTES=${APPROVAL_TIMEOUT_MINUTES}"
echo "       TRADING_UNIVERSE=${TRADING_UNIVERSE}"

# ------------------------------------------------------------------
# Verify connectivity (optional but helpful)
# ------------------------------------------------------------------
echo ""
echo "[3/4] Checking connectivity..."

# Check Tradier API reachability
if curl -s --connect-timeout 5 "https://api.tradier.com/v1/markets/clock" > /dev/null 2>&1; then
    echo "       Tradier API: reachable"
else
    echo "       Tradier API: WARNING — could not reach api.tradier.com"
    echo "       (will retry when strategy starts)"
fi

# Check Telegram API reachability
if curl -s --connect-timeout 5 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" > /dev/null 2>&1; then
    echo "       Telegram API: reachable"
else
    echo "       Telegram API: WARNING — could not reach api.telegram.org"
    echo "       (will retry when bot starts)"
fi

# ------------------------------------------------------------------
# Start the strategy
# ------------------------------------------------------------------
echo ""
echo "[4/4] Starting options trading strategy..."
echo "       Broker: Tradier (paper)"
echo "       Schedule: Daily"
echo "       Universe: ${TRADING_UNIVERSE}"
echo "============================================"
echo ""

# Run the strategy
# The strategy file is mounted at runtime; if not found, show a clear error
STRATEGY_FILE="/app/example_strategies/ai_trading_team_options_debate.py"

if [ ! -f "$STRATEGY_FILE" ]; then
    echo "ERROR: Strategy file not found at ${STRATEGY_FILE}"
    echo "       Make sure the strategy is created (Phase 4) and mounted correctly."
    exit 1
fi

exec python "$STRATEGY_FILE"
