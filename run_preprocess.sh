#!/usr/bin/env bash
# Preprocess input/<method> MP4s into generate/<method>_<batch_ts>_<run_ts>
set -Eeuo pipefail

restore_terminal() {
  stty sane 2>/dev/null || true
  printf '\r\033[K\033[0m\033[?25h\n'
}
trap restore_terminal EXIT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-python3}"
METHOD="${1:-ennerverse}"

"$PYTHON" "$ROOT/scripts/preprocess.py" "$METHOD" --config "$ROOT/config/config.yaml" "${@:2}"
