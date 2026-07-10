# LumiBot Options Trading System
# Python 3.11-slim base with all dependencies
FROM python:3.11-slim

LABEL org.opencontainers.image.title="LumiBot Options Trading"
LABEL org.opencontainers.image.description="Multi-agent LLM options paper trading system with Tradier + Telegram"
LABEL org.opencontainers.image.version="0.1.0"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash lumibot && \
    mkdir -p /app /data/memory /data/logs && \
    chown -R lumibot:lumibot /app /data

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=lumibot:lumibot . .

# Install lumibot in development mode so strategy imports work
RUN pip install --no-cache-dir -e .

# Copy entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER lumibot

# Health check — verify the process is running
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f "ai_trading_team_options_debate" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
