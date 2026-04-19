# Public Summary

## Title
Evidence-Grounded Prompting in Zero-Shot Relation Extraction on a Balanced Re-TACRED Subset

## Study goal
This project tested whether two prompt-design choices improve zero-shot relation extraction:

1. requiring textual evidence
2. enriching relation descriptions with short or rich natural-language definitions

## Setup
The study used a balanced subset derived from Re-TACRED with 7 relation labels. The development subset contained 210 examples total, with 30 examples per label. The held-out test subset contained 105 examples total, with 15 examples per label.

Six prompt conditions were compared on development data:
- label only / short definition / rich definition
- each with and without evidence

The two strongest development conditions were then evaluated on the held-out test set.

## Main results
Development:
- label_only__with_evidence: 0.852381
- label_only__no_evidence: 0.842857
- short_definition__no_evidence: 0.842857
- rich_definition__with_evidence: 0.838095
- short_definition__with_evidence: 0.828571
- rich_definition__no_evidence: 0.819048

Held-out test:
- label_only__no_evidence: 0.742857
- label_only__with_evidence: 0.742857

## Interpretation
Evidence grounding changed where the model succeeded and failed, but it did not improve held-out accuracy in aggregate. It improved `per:schools_attended` from 0.800000 to 0.866667, but reduced `no_relation` from 1.000000 to 0.933333. The hardest relation under both finalists was `per:city_of_birth`, which remained at 0.200000.

Richer relation descriptions did not outperform simpler label-only prompting in this setup. The strongest remaining problem was near-neighbor label confusion rather than broad ambiguity alone.

## Limitations
This was a single-run, API-based comparison on a small balanced subset. The archived record preserves the prompt intervention and evaluation logic, but not the exact provider-specific model snapshot and decoding configuration needed for exact external replication. The results should therefore be read as descriptive findings.
