"""Run the held-out finalist prompt conditions on the balanced test subset.

Environment variables:
- OPENAI_API_KEY
- OPENAI_MODEL (optional)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_FILE = PROJECT_DIR / "prepared_data" / "test_subset.jsonl"
PROMPT_FILE = PROJECT_DIR / "prompts" / "prompt_pack.json"
OUT_FILE = PROJECT_DIR / "results" / "test" / "final_test_results.jsonl"

MODEL = os.getenv("OPENAI_MODEL", "REPLACE_WITH_MODEL_NAME")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

FINAL_CONDITIONS = [
    "label_only__no_evidence",
    "label_only__with_evidence",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_relation": {
            "type": "string",
            "enum": [
                "per:employee_of",
                "per:schools_attended",
                "org:top_members/employees",
                "per:cities_of_residence",
                "per:city_of_birth",
                "per:origin",
                "no_relation",
            ],
        },
        "evidence_span": {"type": "string"},
        "abstain": {"type": "boolean"},
        "confidence_short_reason": {"type": "string"},
    },
    "required": [
        "predicted_relation",
        "evidence_span",
        "abstain",
        "confidence_short_reason",
    ],
    "additionalProperties": False,
}


def safe_parse_response(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(content)
        return obj


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_relation_block(pack: dict) -> str:
    return "Allowed relation labels:\n" + "\n".join(f"- {x}" for x in pack["labels"])


def build_prompt(example: dict, prompt_condition: str, pack: dict) -> str:
    _, evidence_mode = prompt_condition.split("__")
    with_evidence = evidence_mode == "with_evidence"

    instructions = [
        "Choose exactly one label.",
        "Use only the information in the sentence.",
        "Do not use outside knowledge.",
    ]
    if with_evidence:
        instructions.append("Provide the exact supporting text span from the sentence.")
        instructions.append("If the correct output is no_relation, set evidence_span to NONE.")
    else:
        instructions.append("If no target relation is clearly supported, choose no_relation.")
        instructions.append("Set evidence_span to NONE.")

    return f"""
You are performing zero-shot relation extraction.

Task:
Given a sentence and a marked subject-object entity pair, predict the single best relation between the subject and the object.

{build_relation_block(pack)}

Instructions:
- {' '.join(instructions)}

Sentence: "{example['sentence']}"
Subject: "{example['subj_text']}" ({example['subj_type']})
Object: "{example['obj_text']}" ({example['obj_type']})
""".strip()


def main() -> None:
    if MODEL == "REPLACE_WITH_MODEL_NAME":
        raise ValueError("Set OPENAI_MODEL before running this script.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pack = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    test_examples = load_jsonl(TEST_FILE)

    results = []
    for ex in test_examples:
        for cond in FINAL_CONDITIONS:
            prompt = build_prompt(ex, cond, pack)
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    messages=[
                        {"role": "developer", "content": "You are a careful relation extraction system."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "relation_extraction_result",
                            "strict": True,
                            "schema": SCHEMA,
                        },
                    },
                )
                raw_content = response.choices[0].message.content
                parsed = safe_parse_response(raw_content)
                results.append(
                    {
                        "example_id": ex["example_id"],
                        "split": ex["split"],
                        "gold_relation": ex["gold_relation"],
                        "relation_category": ex["relation_category"],
                        "prompt_condition": cond,
                        "sentence": ex["sentence"],
                        "subj_text": ex["subj_text"],
                        "subj_type": ex["subj_type"],
                        "obj_text": ex["obj_text"],
                        "obj_type": ex["obj_type"],
                        "predicted_relation": parsed["predicted_relation"],
                        "evidence_span": parsed["evidence_span"],
                        "abstain": parsed["abstain"],
                        "confidence_short_reason": parsed["confidence_short_reason"],
                        "raw_model_result": raw_content,
                    }
                )
                print(f"Done: {ex['example_id']} | {cond}")
            except Exception as exc:
                print(f"FAILED: {ex['example_id']} | {cond} | {exc}")
                results.append(
                    {
                        "example_id": ex["example_id"],
                        "split": ex["split"],
                        "gold_relation": ex["gold_relation"],
                        "relation_category": ex["relation_category"],
                        "prompt_condition": cond,
                        "sentence": ex["sentence"],
                        "subj_text": ex["subj_text"],
                        "subj_type": ex["subj_type"],
                        "obj_text": ex["obj_text"],
                        "obj_type": ex["obj_type"],
                        "predicted_relation": "ERROR",
                        "evidence_span": "ERROR",
                        "abstain": False,
                        "confidence_short_reason": str(exc),
                        "raw_model_result": "",
                    }
                )

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
