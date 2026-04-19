"""Score development-set predictions by prompt condition."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
IN_FILE = PROJECT_DIR / "results" / "dev" / "full_dev_results.jsonl"
OUT_CSV = PROJECT_DIR / "results" / "dev_results.csv"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    rows = load_jsonl(IN_FILE)
    df = pd.DataFrame(rows)
    df = df[df["predicted_relation"] != "ERROR"].copy()
    df["correct_prediction"] = (df["gold_relation"] == df["predicted_relation"]).astype(int)

    summary = (
        df.groupby("prompt_condition")["correct_prediction"]
        .agg(count="count", correct="sum", accuracy="mean")
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    summary.to_csv(OUT_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
