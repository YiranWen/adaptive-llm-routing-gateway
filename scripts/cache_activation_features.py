"""Cache Qwen activation features for a prepared routing table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.config import ModelConfig
from llm_router.dataset import load_routing_table
from llm_router.feature_extractor import MechanisticFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "routing_table_train_512.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "activation_features_train_512.npz",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layer-offset", type=int, default=-2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--raw-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = load_routing_table(args.input)
    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.row_limit is not None and args.row_limit <= 0:
        raise ValueError("--row-limit must be positive")

    row_end = None if args.row_limit is None else args.row_start + args.row_limit
    table = table.iloc[args.row_start : row_end].reset_index(drop=True)
    prompts = table["prompt"].astype(str).tolist()
    if not prompts:
        raise ValueError(f"no prompts found in {args.input}")

    config = ModelConfig(
        model_name=args.model,
        max_length=args.max_length,
        device=args.device,
        layer_offset=args.layer_offset,
        use_chat_template=not args.raw_prompt,
    )
    extractor = MechanisticFeatureExtractor(config)

    features: list[np.ndarray] = []
    batch_latencies: list[float] = []
    batch_token_counts: list[int] = []

    started = time.perf_counter()
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="caching features"):
        batch = prompts[start : start + args.batch_size]
        result = extractor.extract_batch(batch)
        features.append(result.features.astype(np.float32))
        batch_latencies.append(result.elapsed_ms)
        batch_token_counts.append(result.num_tokens)

    feature_matrix = np.concatenate(features, axis=0)
    elapsed_s = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=feature_matrix,
        score_small=table["score_small"].to_numpy(dtype=np.float32),
        score_large=table["score_large"].to_numpy(dtype=np.float32),
        latency_small=table["latency_small"].to_numpy(dtype=np.float32),
        latency_large=table["latency_large"].to_numpy(dtype=np.float32),
        small_failure=table["small_failure"].to_numpy(dtype=np.int64),
        prompt=np.array(prompts, dtype=object),
    )

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "model": config.model_name,
        "device": extractor.device,
        "max_length": config.max_length,
        "layer_index": extractor.layer_index,
        "feature_dim": int(feature_matrix.shape[1]),
        "num_rows": int(feature_matrix.shape[0]),
        "batch_size": args.batch_size,
        "row_start": args.row_start,
        "row_limit": args.row_limit,
        "row_end": args.row_start + int(feature_matrix.shape[0]),
        "elapsed_s": elapsed_s,
        "avg_batch_latency_ms": float(np.mean(batch_latencies)),
        "avg_tokens_per_batch": float(np.mean(batch_token_counts)),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_path}")
    print(f"feature_shape: {feature_matrix.shape}")
    print(f"feature_dtype: {feature_matrix.dtype}")
    print(f"row_start: {args.row_start}")
    print(f"row_end: {args.row_start + feature_matrix.shape[0]}")
    print(f"device: {extractor.device}")
    print(f"layer_index: {extractor.layer_index}")
    print(f"elapsed_s: {elapsed_s:.2f}")
    print(f"avg_batch_latency_ms: {np.mean(batch_latencies):.2f}")


if __name__ == "__main__":
    main()
