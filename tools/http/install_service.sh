#!/usr/bin/env bash
# I3.4: install/uninstall the AInfer systemd user service.
# Usage: ./tools/http/install_service.sh [install|uninstall|status]
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="ainfer.service"

cmd="${1:-install}"
case "$cmd" in
  install)
    mkdir -p "$UNIT_DIR"
    cp "$ROOT/tools/http/$UNIT" "$UNIT_DIR/$UNIT"
    systemctl --user daemon-reload
    systemctl --user enable "$UNIT"
    echo "installed $UNIT_DIR/$UNIT (enabled, not started)"
    echo "start now: systemctl --user start $UNIT"
    echo "logs:      journalctl --user -u $UNIT -f"
    ;;
  uninstall)
    systemctl --user disable "$UNIT" 2>/dev/null || true
    rm -f "$UNIT_DIR/$UNIT"
    systemctl --user daemon-reload
    echo "uninstalled $UNIT"
    ;;
  status)
    systemctl --user status "$UNIT" --no-pager || true
    ;;
  *)
    echo "Usage: $0 [install|uninstall|status]" >&2
    exit 2
    ;;
esac
