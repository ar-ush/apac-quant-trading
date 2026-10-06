#!/usr/bin/env bash
# Runs the live bot and restarts it a few seconds after any exit. Run inside tmux so it survives the Session Manager window.
# The bot itself is stateless about positions (the wallet is the truth) so a restart is always safe.
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs state
# only one bot may ever trade this account: a second copy would double every order
exec 9>state/bot.lock
flock -n 9 || { echo "another bot instance is already running (state/bot.lock); exiting"; exit 1; }
# on the organisers' EC2 the bot must trade the competition account, never the test one by mistake
if [ -z "$ALLOW_TEST_ACCOUNT" ] && ! grep -Eq '^ROOSTOO_ACCOUNT=competition' .env 2>/dev/null; then
  echo "ROOSTOO_ACCOUNT is not 'competition' in .env (set ALLOW_TEST_ACCOUNT=1 to run on the test account); exiting"; exit 1
fi
while true; do
  echo "$(date -u +%FT%TZ) starting bot (commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown))" | tee -a logs/restarts.log
  python -m bot.main --live "$@"
  echo "$(date -u +%FT%TZ) bot exited with code $?; restarting in 10s" | tee -a logs/restarts.log
  sleep 10
done
