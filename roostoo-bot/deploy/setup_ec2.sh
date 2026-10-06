#!/usr/bin/env bash
# One-shot setup for the organisers' EC2 box (Amazon Linux 2023, Session Manager shell, no SSH).
#   git clone <this repo> && cd <repo>/roostoo-bot && bash deploy/setup_ec2.sh
# Then: cp .env.example .env && nano .env  (competition keys)  and start with deploy/run_bot.sh inside tmux.
set -euo pipefail

sudo dnf install -y git tmux python3.11 python3.11-pip 2>/dev/null || sudo dnf install -y git tmux python3 python3-pip
PY=$(command -v python3.11 || command -v python3)
echo "using $($PY --version)"

$PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -m pytest tests -q
mkdir -p logs state
chmod +x deploy/run_bot.sh
echo
echo "Setup OK. Next:"
echo "  1) cp .env.example .env && nano .env      # put the COMPETITION keys, set ROOSTOO_ACCOUNT=competition"
echo "  2) python -m bot.main --check             # read-only: shows the account equity (should be ~100,000 USD)"
echo "  3) tmux new -s bot"
echo "  4) bash deploy/run_bot.sh                 # restarts the bot if it ever exits; detach with Ctrl+B then D"
