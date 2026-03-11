---
name: bq
description: Run BigQuery CLI read-only analyses (SELECT-only) and summarize insights for experiment decisions.
disable-model-invocation: true
---

Assumptions:
- The user has `bq` configured and authenticated on this machine.

Rules:
- Default to **read-only** queries (SELECT). Do not run CREATE/UPDATE/DELETE unless explicitly asked.
- Always echo the query, dataset/table names, and filters before execution.

Workflow:
1) Restate the analysis question and define the metric(s).
2) Draft the SQL with explicit date ranges / partitions.
3) Run via `bq query --use_legacy_sql=false ...` (or the project’s standard wrapper).
4) Save results under `notes/` (csv + a short markdown summary).
