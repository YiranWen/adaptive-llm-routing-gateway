"""Download and normalize a RouteLLM dataset split for offline routing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import load_dataset

from llm_router.dataset import (
    DEFAULT_DATASET_NAME,
    DEFAULT_LARGE_MODEL,
    DEFAULT_SMALL_MODEL,
    PreparedDatasetMetadata,
    prepare_gpt4_dataset_table,
    save_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=184)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--small-failure-threshold", type=float, default=0.7)
    parser.add_argument("--latency-small", type=float, default=0.2)
    parser.add_argument("--latency-large", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "routing_table_train_512.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split)

    shuffled = not args.no_shuffle
    if shuffled:
        ds = ds.shuffle(seed=args.seed)
    if args.limit is not None and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    table = prepare_gpt4_dataset_table(
        ds,
        small_failure_threshold=args.small_failure_threshold,
        latency_small=args.latency_small,
        latency_large=args.latency_large,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    metadata = PreparedDatasetMetadata(
        dataset_name=args.dataset,
        split=args.split,
        num_rows=len(table),
        small_model=DEFAULT_SMALL_MODEL,
        large_model=DEFAULT_LARGE_MODEL,
        small_failure_threshold=args.small_failure_threshold,
        latency_small=args.latency_small,
        latency_large=args.latency_large,
        seed=args.seed,
        shuffled=shuffled,
    )
    metadata_path = args.output.with_suffix(".metadata.json")
    save_metadata(metadata_path, metadata)

    print(f"dataset: {args.dataset}")
    print(f"split: {args.split}")
    print(f"rows: {len(table)}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_path}")
    print(f"score_small_mean: {table['score_small'].mean():.4f}")
    print(f"small_failure_rate: {table['small_failure'].mean():.4f}")


if __name__ == "__main__":
    main()

