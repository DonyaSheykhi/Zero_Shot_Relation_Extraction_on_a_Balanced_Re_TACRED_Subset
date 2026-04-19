"""Prepare balanced Re-TACRED subsets for the zero-shot prompting study.

This script recreates the balanced label subsets used in the paper:
- development: 30 examples per label
- test: 15 examples per label

It does not ship the benchmark data itself. Use one of the following modes:

1) Hugging Face:
   python src/prepare_retacred_public.py --source huggingface

2) Local JSON files:
   python src/prepare_retacred_public.py --source local --data-dir path/to/re_tacred_json

Expected local files:
- dev.json
- test.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
TARGET_LABELS = [
    "per:employee_of",
    "per:schools_attended",
    "org:top_members/employees",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:origin",
    "no_relation",
]

RELATION_CATEGORY = {
    "per:employee_of": "easy_lexical",
    "per:schools_attended": "semantically_subtle",
    "org:top_members/employees": "near_neighbor_confusable",
    "per:cities_of_residence": "semantically_subtle",
    "per:city_of_birth": "near_neighbor_confusable",
    "per:origin": "ambiguous_or_confusable",
    "no_relation": "negative_control",
}

DEV_PER_LABEL = 30
TEST_PER_LABEL = 15


def ptb_to_text(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = text.replace(" -LRB- ", " (").replace("-LRB-", "(")
    text = text.replace(" -RRB- ", ") ").replace("-RRB-", ")")
    text = text.replace(" -LSB- ", " [").replace("-LSB-", "[")
    text = text.replace(" -RSB- ", "] ").replace("-RSB-", "]")
    text = text.replace(" -LCB- ", " {").replace("-LCB-", "{")
    text = text.replace(" -RCB- ", "} ").replace("-RCB-", "}")
    for punct in [".", ",", ":", ";", "?", "!", "%", "'s"]:
        text = text.replace(f" {punct}", punct)
    text = text.replace(" n't", "n't")
    text = text.replace(" 'm", "'m").replace(" 're", "'re").replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll").replace(" 'd", "'d")
    return " ".join(text.split()).strip()


def span_text(tokens: list[str], start: int, end: int) -> str:
    # Re-TACRED end indices are inclusive in the local JSON files.
    return " ".join(tokens[start : end + 1])


def to_record(example: dict, split_name: str) -> dict:
    tokens = example["token"]
    return {
        "example_id": f"{split_name}__{example['id']}",
        "split": split_name,
        "sentence": ptb_to_text(tokens),
        "subj_text": span_text(tokens, example["subj_start"], example["subj_end"]),
        "subj_type": example["subj_type"],
        "obj_text": span_text(tokens, example["obj_start"], example["obj_end"]),
        "obj_type": example["obj_type"],
        "gold_relation": example["relation"],
        "relation_category": RELATION_CATEGORY[example["relation"]],
        "source_instance_id": example["id"],
    }


def balanced_sample(records: list[dict], per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["gold_relation"]].append(record)

    for label in TARGET_LABELS:
        available = len(grouped[label])
        if available < per_label:
            raise ValueError(f"Not enough examples for {label}: need {per_label}, found {available}")

    sampled: list[dict] = []
    for label in TARGET_LABELS:
        pool = grouped[label][:]
        rng.shuffle(pool)
        sampled.extend(pool[:per_label])

    rng.shuffle(sampled)
    return sampled


def load_from_local_json(data_dir: Path) -> tuple[list[dict], list[dict]]:
    dev_path = data_dir / "dev.json"
    test_path = data_dir / "test.json"
    if not dev_path.exists() or not test_path.exists():
        raise FileNotFoundError("Local mode expects dev.json and test.json inside --data-dir")

    with dev_path.open("r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with test_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    dev_records = [to_record(ex, "dev") for ex in dev_data if ex["relation"] in TARGET_LABELS]
    test_records = [to_record(ex, "test") for ex in test_data if ex["relation"] in TARGET_LABELS]
    return dev_records, test_records


def load_from_huggingface() -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset("DFKI-SLT/tacred", name="re-tacred", trust_remote_code=True)

    dev_records = [to_record(ex, "dev") for ex in dataset["validation"] if ex["relation"] in TARGET_LABELS]
    test_records = [to_record(ex, "test") for ex in dataset["test"] if ex["relation"] in TARGET_LABELS]
    return dev_records, test_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["huggingface", "local"], default="huggingface")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing dev.json and test.json")
    parser.add_argument("--outdir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.source == "local":
        if args.data_dir is None:
            raise ValueError("--data-dir is required when --source local")
        dev_records, test_records = load_from_local_json(args.data_dir)
    else:
        dev_records, test_records = load_from_huggingface()

    args.outdir.mkdir(parents=True, exist_ok=True)

    dev_subset = balanced_sample(dev_records, DEV_PER_LABEL, seed=args.seed)
    test_subset = balanced_sample(test_records, TEST_PER_LABEL, seed=args.seed)

    dev_df = pd.DataFrame(dev_subset)
    test_df = pd.DataFrame(test_subset)
    all_df = pd.concat([dev_df, test_df], ignore_index=True)

    dev_df.to_json(args.outdir / "dev_subset.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json(args.outdir / "test_subset.jsonl", orient="records", lines=True, force_ascii=False)
    all_df.to_csv(args.outdir / "prompt_ready_records.csv", index=False)

    print("\nSaved:")
    print(args.outdir / "dev_subset.jsonl")
    print(args.outdir / "test_subset.jsonl")
    print(args.outdir / "prompt_ready_records.csv")

    print("\nBalanced development counts:")
    print(dev_df["gold_relation"].value_counts().sort_index())

    print("\nBalanced test counts:")
    print(test_df["gold_relation"].value_counts().sort_index())


if __name__ == "__main__":
    main()
