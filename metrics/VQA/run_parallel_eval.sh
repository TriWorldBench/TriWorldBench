#!/usr/bin/env bash
set -Eeuo pipefail

STARTED_AT_EPOCH="$(date +%s)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd -P)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT:-$BENCHMARK_ROOT/generate}}"
GPU_LIST="${GPU:-${VLM_EVAL_GPU:-}}"
OUTPUT_ROOT=""
RESUME=0
FORWARD_ARGS=()

while (($#)); do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--gpus 0,1,...] [--output-root DIR|--resume DIR] [evaluator options]"
      echo "Runs one evaluation shard per GPU and merges all shard reports."
      exit 0
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --gpus=*)
      GPU_LIST="${1#*=}"
      shift
      ;;
    --output-root|--output_root)
      ((RESUME == 0)) || { echo "--output-root and --resume cannot be used together" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-root=*|--output_root=*)
      ((RESUME == 0)) || { echo "--output-root and --resume cannot be used together" >&2; exit 2; }
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --resume)
      [[ -z "$OUTPUT_ROOT" ]] || { echo "--output-root and --resume cannot be used together" >&2; exit 2; }
      RESUME=1
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --resume=*)
      [[ -z "$OUTPUT_ROOT" ]] || { echo "--output-root and --resume cannot be used together" >&2; exit 2; }
      RESUME=1
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --input-dir|--input_dir|--data-root|--data_root)
      INPUT_DIR="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --input-dir=*|--input_dir=*|--data-root=*|--data_root=*)
      INPUT_DIR="${1#*=}"
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --num-shards|--shard-id|--gpu|--cot-log-root|--cot-log-root=*)
      echo "$1 is managed by run_parallel_eval.sh" >&2
      exit 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$GPU_LIST" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
fi
if [[ -z "$GPU_LIST" ]]; then
  echo "No GPUs detected. Pass --gpus with a comma-separated device list." >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if ((${#GPUS[@]} == 0)); then
  echo "No GPU IDs supplied" >&2
  exit 1
fi

if ((RESUME)); then
  [[ -d "$OUTPUT_ROOT" ]] || { echo "Resume directory not found: $OUTPUT_ROOT" >&2; exit 1; }
elif [[ -z "$OUTPUT_ROOT" ]]; then
  DATA_LABEL="$(basename "${INPUT_DIR%/}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/run_parallel_${DATA_LABEL}_${TIMESTAMP}"
fi
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
SHARD_ROOT="$OUTPUT_ROOT/shards"
LOG_ROOT="$OUTPUT_ROOT/logs"
NUM_SHARDS="${#GPUS[@]}"
if ((RESUME)) && [[ -d "$SHARD_ROOT" ]]; then
  shopt -s nullglob
  EXISTING_SHARDS=("$SHARD_ROOT"/shard_*)
  shopt -u nullglob
  if ((${#EXISTING_SHARDS[@]} > 0 && ${#EXISTING_SHARDS[@]} != NUM_SHARDS)); then
    echo "Resume requires the original shard count: found ${#EXISTING_SHARDS[@]}, requested $NUM_SHARDS GPUs." >&2
    exit 1
  fi
  for ((shard_id = 0; shard_id < ${#EXISTING_SHARDS[@]}; shard_id++)); do
    [[ -d "$SHARD_ROOT/shard_$shard_id" ]] || {
      echo "Resume shard layout is incomplete: missing $SHARD_ROOT/shard_$shard_id" >&2
      exit 1
    }
  done
fi
mkdir -p "$SHARD_ROOT" "$LOG_ROOT"

PIDS=()
MONITOR_PID=""
DONE_FILE="$OUTPUT_ROOT/.evaluation_done"
rm -f "$DONE_FILE"
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

if ((RESUME)); then
  echo "[resume] workers=$NUM_SHARDS gpus=$GPU_LIST output=$OUTPUT_ROOT"
else
  echo "[parallel] workers=$NUM_SHARDS gpus=$GPU_LIST output=$OUTPUT_ROOT"
fi
for ((shard_id = 0; shard_id < NUM_SHARDS; shard_id++)); do
  gpu="${GPUS[$shard_id]}"
  shard_output="$SHARD_ROOT/shard_$shard_id"
  log_path="$LOG_ROOT/shard_$shard_id.log"
  echo "[launch] shard=$shard_id gpu=$gpu log=$log_path"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PROJECT_ROOT/run_yes_no_vlm_eval.py" \
    "${FORWARD_ARGS[@]}" \
    --gpu "$gpu" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$shard_id" \
    --output-root "$shard_output" \
    --cot-log-root "$OUTPUT_ROOT/cot_logs" \
    >>"$log_path" 2>&1 &
  PIDS+=("$!")
done

"$PYTHON" "$PROJECT_ROOT/live_merge_eval_reports.py" \
  --shard-root "$SHARD_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --done-file "$DONE_FILE" \
  --num-shards "$NUM_SHARDS" \
  >>"$LOG_ROOT/live_report.log" 2>&1 &
MONITOR_PID="$!"
echo "[live-report] global CSV updates after every episode: $OUTPUT_ROOT/report.csv"

FAILED=0
for ((shard_id = 0; shard_id < NUM_SHARDS; shard_id++)); do
  if wait "${PIDS[$shard_id]}"; then
    echo "[complete] shard=$shard_id"
  else
    echo "[failed] shard=$shard_id log=$LOG_ROOT/shard_$shard_id.log" >&2
    FAILED=1
  fi
done
touch "$DONE_FILE"
wait "$MONITOR_PID"
MONITOR_PID=""
trap - INT TERM
if ((FAILED)); then
  echo "One or more shards failed; reports were not merged." >&2
  exit 1
fi

"$PYTHON" "$PROJECT_ROOT/merge_shard_reports.py" \
  --shard-root "$SHARD_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --started-at-epoch "$STARTED_AT_EPOCH"

echo "Report JSON: $OUTPUT_ROOT/report.json"
echo "Report CSV:  $OUTPUT_ROOT/report.csv"
