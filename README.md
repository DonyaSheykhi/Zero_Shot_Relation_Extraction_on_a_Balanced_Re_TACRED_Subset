# Evidence-Grounded Prompting in Zero-Shot Relation Extraction

This repository contains a cleaned public release of a controlled prompting study on zero-shot relation extraction using a balanced subset derived from Re-TACRED.

## Research question

How does evidence-grounded prompting interact with the semantic richness of relation descriptions in zero-shot relation extraction, especially for semantically subtle and confusable relations?

## Experimental setup

- 7 relation labels:
  - `per:employee_of`
  - `per:schools_attended`
  - `org:top_members/employees`
  - `per:cities_of_residence`
  - `per:city_of_birth`
  - `per:origin`
  - `no_relation`
- Balanced development subset: 210 examples total, 30 per label
- Balanced held-out test subset: 105 examples total, 15 per label
- 6 prompt conditions:
  - label only / short definition / rich definition
  - each with and without evidence

## Main findings

- Best development result: `label_only__with_evidence` = `0.852381`
- Held-out finalists:
  - `label_only__no_evidence` = `0.742857`
  - `label_only__with_evidence` = `0.742857`
- Evidence changed the distribution of errors but did not improve held-out accuracy overall
- Richer relation descriptions did not outperform simpler label-only prompting
- The hardest remaining relation was `per:city_of_birth`

## Public release notes

This repo is intentionally cleaned for public sharing.

- Personal identifiers and course-submission metadata are not included
- Local machine paths were removed
- The benchmark data is not redistributed here
- The public release preserves the prompt design, summary results, and cleaned reproduction scripts

## Repository layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── README.md
├── prompts/
│   ├── prompt_pack.json
│   └── *.txt
├── src/
│   └── prepare_retacred_public.py
├── scripts/
│   ├── run_dev_eval.py
│   ├── run_final_test.py
│   ├── score_dev_results.py
│   └── score_final_test.py
├── results/
│   ├── dev_results.csv
│   ├── test_results.csv
│   ├── relation_level_results.csv
│   └── category_level_results.csv
├── figures/
│   └── main_results.png
└── paper/
    └── public_summary.pdf
```

## Quickstart

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the balanced subsets:

```bash
python src/prepare_retacred_public.py --source huggingface
```

Run the development comparison:

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_MODEL=YOUR_MODEL_NAME
python scripts/run_dev_eval.py
python scripts/score_dev_results.py
```

Run the held-out finalists:

```bash
python scripts/run_final_test.py
python scripts/score_final_test.py
```

## Reproducibility note

This was a single-run, API-based prompt comparison. The archived project record preserves the prompt intervention, label inventory, output schema, and evaluation logic, but does not preserve the provider-specific model snapshot and decoding configuration with enough detail for exact external replication. The results should therefore be interpreted descriptively rather than as definitive general performance claims.

## Citation

If you reference this project, please cite or link to the accompanying public summary for this repository.
