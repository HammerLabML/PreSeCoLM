#!/usr/bin/env python3
"""Split an experiment config into n sub-configs, distributing models across jobs.

Each model list (huggingface / openai / sentence_transformer) is split separately.
Models are assigned round-robin with a running counter that carries over between
lists, so each job ends up with roughly the same total number of models.

Usage:
    python split_config.py base_config.json -n 4
    python split_config.py base_config.json -n 4 -o experiments/configs/split/
"""
import argparse
import copy
import json
from pathlib import Path

MODEL_KEYS = ["huggingface_models", "openai_models", "sentence_transformer_models"]


def split_config(config: dict, n: int, sep: str = "_") -> list[dict]:
    subconfigs = [copy.deepcopy(config) for _ in range(n)]
    for sub in subconfigs:
        for key in MODEL_KEYS:
            if key in sub:
                sub[key] = []

    counter = 0
    for key in MODEL_KEYS:
        for model in config.get(key, []):
            subconfigs[counter % n][key].append(model)
            counter += 1

    # drop jobs that ended up with no models (when n > total number of models)
    subconfigs = [s for s in subconfigs if any(s.get(k) for k in MODEL_KEYS)]

    base_path = config["results_path"].rstrip("/")
    for i, sub in enumerate(subconfigs, start=1):
        sub["results_path"] = f"{base_path}{sep}{i}"
    return subconfigs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="base config with all models")
    parser.add_argument("-n", "--num-splits", type=int, required=True, help="number of sub-configs")
    parser.add_argument("-r", "--results-path", help="override results_path from the base config")
    parser.add_argument("-o", "--out-dir", type=Path, help="output directory (default: next to base config)")
    parser.add_argument("--sep", default="_", help="separator between results path and index (default: _)")
    args = parser.parse_args()

    if args.num_splits < 1:
        parser.error("-n must be >= 1")

    with open(args.config) as f:
        config = json.load(f)
    if args.results_path:
        config["results_path"] = args.results_path

    subconfigs = split_config(config, args.num_splits, args.sep)
    if len(subconfigs) < args.num_splits:
        print(f"Warning: only {len(subconfigs)} non-empty sub-configs (fewer models than n)")

    out_dir = args.out_dir or args.config.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sub in enumerate(subconfigs, start=1):
        out_file = out_dir / f"{args.config.stem}{args.sep}{i}.json"
        with open(out_file, "w") as f:
            json.dump(sub, f, indent=2)
        counts = ", ".join(f"{k.split('_')[0]}={len(sub.get(k, []))}" for k in MODEL_KEYS)
        print(f"{out_file}  ({counts})  -> {sub['results_path']}")


if __name__ == "__main__":
    main()
