#!/usr/bin/env bash
# Install the voice-daemon systemd --user unit and the CLI wrappers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
UNIT_DIR="$HOME/.config/systemd/user"

# Hermes profile sessions and kanban workers export HOME=~/.hermes/profiles/<p>/home.
# Running the installer under that value silently builds a second, useless
# bin dir and systemd user dir inside the profile instead of the real ones.
if [[ "$HOME" == */.hermes/profiles/* ]]; then
  echo "HOME=$HOME is a Hermes profile home, not a login home." >&2
  echo "Run:  HOME=/home/\$(id -un) $0" >&2
  exit 1
fi

mkdir -p "$BIN_DIR" "$UNIT_DIR"

for wrapper in "$ROOT"/bin/*; do
  [[ -f "$wrapper" ]] || continue
  chmod +x "$wrapper"
  name="$(basename "$wrapper")"
  ln -sfn "$wrapper" "$BIN_DIR/$name"
  echo "  wrapper $BIN_DIR/$name -> $wrapper"
done

# The tracked unit is an example written for ~/repo/voice-daemon-public;
# render it with this checkout's real path instead of symlinking it.
sed "s#%h/repo/voice-daemon-public#$ROOT#g" "$ROOT/systemd/voice-daemon.service" \
  > "$UNIT_DIR/voice-daemon.service"
echo "  unit    $UNIT_DIR/voice-daemon.service (rendered from systemd/voice-daemon.service)"

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "  env     $ROOT/.env (from .env.example)"
fi

systemctl --user daemon-reload
echo
echo "Enable and start:  systemctl --user enable --now voice-daemon"
echo "Health:            curl -s localhost:9877/health | jq"
