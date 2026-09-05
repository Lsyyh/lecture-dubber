#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs sox libsox-dev

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
fi

echo "Base environment ready. Next: install WhisperX, start your local Qwen OpenAI-compatible server, and install CosyVoice per README.md."
