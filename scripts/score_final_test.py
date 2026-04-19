"""Score held-out test predictions overall, by relation, and by category."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
IN_FILE = PROJECT_DIR / "results" / "test" / "final_test_results.jsonl"
OUT_TEST = PROJECT_DIR / "results" / "test_results.csv"
OUT_REL = PROJECT_DIR / "results" / "relation_level_results.csv"
OUT_CAT = PROJECT_DIR / "results" / "category_level_results.csv"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    rows = load_jsonl(IN_FILE)
    df = pd.DataFrame(rows)
    df = df[df["predicted_relation"] != "ERROR"].copy()
    df["correct_prediction"] = (df["gold_relation"] == df["predicted_relation"]).astype(int)

    overall = (
        df.groupby("prompt_condition")["correct_prediction"]
        .agg(count="count", correct="sum", accuracy="mean")
        .reset_index()
        .sort_values(["accuracy", "prompt_condition"], ascending=[False, True])
    )

    by_relation = (
        df.groupby(["prompt_condition", "gold_relation"])["correct_prediction"]
        .agg(count="count", correct="sum", accuracy="mean")
        .reset_index()
    )

    by_category = (
        df.groupby(["prompt_condition", "relation_category"])["correct_prediction"]
        .agg(count="count", correct="sum", accuracy="mean")
        .reset_index()
    )

    overall.to_csv(OUT_TEST, index=False)
    by_relation.to_csv(OUT_REL, index=False)
    by_category.to_csv(OUT_CAT, index=False)

    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nSaved:")
    print(OUT_TEST)
    print(OUT_REL)
    print(OUT_CAT)


if __name__ == "__main__":
    main()
