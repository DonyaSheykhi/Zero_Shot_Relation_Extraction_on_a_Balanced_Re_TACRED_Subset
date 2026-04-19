# Data note

This public repository does not redistribute Re-TACRED examples.

To recreate the balanced subsets used in the study, download or access Re-TACRED locally and run:

```bash
python src/prepare_retacred_public.py --source huggingface
```

or, if you already have local JSON files:

```bash
python src/prepare_retacred_public.py --source local --data-dir path/to/re_tacred_json
```

Expected output files:
- `prepared_data/dev_subset.jsonl`
- `prepared_data/test_subset.jsonl`
- `prepared_data/prompt_ready_records.csv`
