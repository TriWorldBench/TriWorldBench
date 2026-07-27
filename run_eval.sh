#!/usr/bin/env bash
# Run Triworld benchmark evaluation from config/config.yaml
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG="${CONFIG:-$ROOT/config/config.yaml}"

# Read python from config if yq/python available, else default
PYTHON="${PYTHON:-python3}"
if command -v python3 >/dev/null 2>&1; then
  PYTHON="$(python3 -c "
import yaml, sys
from pathlib import Path
cfg = yaml.safe_load(Path('$CONFIG').read_text())
print(cfg.get('python', 'python3'))
" 2>/dev/null || echo python3)"
fi

# Catch missing/broken imports before long pipeline runs
"$PYTHON" "$ROOT/scripts/smoke_imports.py"

# Banner + rich UI rendered by scripts/lib/pipeline_ui.py
exec "$PYTHON" "$ROOT/scripts/run_eval.py" --config "$CONFIG" "$@"
