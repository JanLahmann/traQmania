#!/usr/bin/env bash
# One-command start: create/refresh a venv, install traqmania, launch the server.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet -e .

URL="http://127.0.0.1:${TRAQMANIA_PORT:-8000}"
echo "traQmania starting at $URL"
(command -v open >/dev/null && sleep 2 && open "$URL" &) 2>/dev/null || true
(command -v xdg-open >/dev/null && sleep 2 && xdg-open "$URL" &) 2>/dev/null || true

exec python -m traqmania "$@"
