#!/usr/bin/env bash
# Runs the live bot and restarts it a few seconds after any exit. Run inside tmux so it survives the Session Manager window.
# The bot itself is stateless about positions (the wallet is the truth) so a restart is always safe.
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs
while true; do
  echo "$(date -u +%FT%TZ) starting bot (commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown))" | tee -a logs/restarts.log
  python -m bot.main --live "$@"
  echo "$(date -u +%FT%TZ) bot exited with code $?; restarting in 10s" | tee -a logs/restarts.log
  sleep 10
done
