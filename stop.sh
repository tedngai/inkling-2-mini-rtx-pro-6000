#!/usr/bin/env bash
# Stop the server started by start.sh on the given port.
set -euo pipefail
PORT="${PORT:-8000}"
SELF="$$"
PARENT="$PPID"
PIDS="$(pgrep -f "vllm serve thinkingmachines/Inkling-2-Mini.*--port ${PORT}" | grep -vE "^(${SELF}|${PARENT})$" || true)"
if [[ -z "$PIDS" ]]; then
  echo "no Inkling-2-Mini server found on port ${PORT}"
  exit 0
fi
ps -o pid,ppid,etime,cmd -p $(echo "$PIDS" | tr '\n' ',' | sed 's/,$//')
kill $PIDS
echo "signalled server process(es) on port ${PORT}"
