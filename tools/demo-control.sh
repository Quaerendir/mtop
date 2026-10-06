#!/usr/bin/env bash
# Record docs/demo-control.cast and render docs/demo-control.gif: a scripted
# `mtop --control` session against a local Ollama (load from the picker,
# keep-alive, pull with progress, stop).
#
#   tools/demo-control.sh            # needs tmux, uvx (asciinema 2.x), agg
#
# The session runs in a private tmux server sized 136x38; this script types
# into it. Model names below are what the Spark has; edit LOAD_A/LOAD_B (their
# rows in the picker, which sorts /api/tags case-insensitively) and PULL for
# another host. The pulled model is deleted again at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
LOAD_A=${LOAD_A:-2}            # bielik-ocr-64k:latest
LOAD_B=${LOAD_B:-13}           # qwen3.8:27b-mtp-q4_K_M
PULL=${PULL:-smollm:135m}
API=${API:-http://localhost:11434}
AGG=${AGG:-agg}
# The README screenshots' palette (tools/screenshot.py), background + 16 colors.
THEME="1e1e2e,cdd6f4,45475a,f38ba8,a6e3a1,f9e2af,89b4fa,f5c2e7,94e2d5,bac2de,585b70,f38ba8,a6e3a1,f9e2af,89b4fa,f5c2e7,94e2d5,a6adc8"

t() { tmux -L mtopdemo -f /dev/null "$@"; }
key() { t send-keys -t demo "$@"; }
wait_for() {  # until the pane shows $1, at most $2 s
  for _ in $(seq 1 $(($2 * 10))); do
    t capture-pane -p -t demo | grep -qF -- "$1" && return 0
    sleep 0.1
  done
  echo "timeout waiting for: $1" >&2
  return 1
}

t kill-server 2>/dev/null || true
t new-session -d -s demo -x 136 -y 38 \
  "TERM=xterm-256color uvx asciinema rec --overwrite -q -i 2 -t 'mtop --control' \
   -c 'env PYTHONPATH=src python3 -m mtop --control' docs/demo-control.cast; sleep 3"
sleep 4
key L; sleep 1.5
for _ in $(seq 1 "$LOAD_A"); do key Down; sleep 0.35; done
sleep 0.8; key Enter
wait_for "Loaded " 120; sleep 2.5
key L; sleep 1.5
for _ in $(seq 1 "$LOAD_B"); do key Down; sleep 0.18; done
sleep 0.8; key Enter
wait_for "Loaded " 180; sleep 2.5
key t; sleep 1.8; key 2
wait_for "Keeping" 30; sleep 2.5
key P; sleep 0.8
for ((i = 0; i < ${#PULL}; i++)); do key -l "${PULL:i:1}"; sleep 0.12; done
sleep 0.8; key Enter
wait_for "Pulled " 300; sleep 2.5
key Down; sleep 0.8; key s; sleep 1.5; key y
wait_for "Stopped" 60; sleep 3
key q
sleep 2
t kill-server 2>/dev/null || true
curl -s -X DELETE "$API/api/delete" -d "{\"model\":\"$PULL\"}" >/dev/null || true

"$AGG" --theme "$THEME" --font-family "DejaVu Sans Mono" \
  --font-dir /usr/share/fonts/truetype/dejavu --font-size 16 \
  --idle-time-limit 2 --last-frame-duration 3 --fps-cap 15 \
  docs/demo-control.cast docs/demo-control.gif
echo "wrote docs/demo-control.cast docs/demo-control.gif"
