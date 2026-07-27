#!/usr/bin/env python3
"""Persistent per-GPU CLIP text-similarity worker for semantic_alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.semantic_alignment import load_clip_text_encoder, run_clip_text_similarity  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load_clip_text_encoder(args.model_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "STOP":
            break
        job = json.loads(line)
        gen_id = str(job.get("gen_id") or "")
        try:
            score = run_clip_text_similarity(
                model,
                tokenizer,
                str(job["gen_text"]),
                str(job["gt_text"]),
            )
            print(json.dumps({"status": "success", "gen_id": gen_id, "score": score}), flush=True)
        except Exception as exc:
            print(json.dumps({
                "status": "failed",
                "gen_id": gen_id,
                "error": str(exc),
            }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
