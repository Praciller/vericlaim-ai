# Evaluation

The repository contains a typed SciFact adapter, hash-validated dataset manifest, deterministic closed-corpus BM25 retrieval, structured-response cache, and bounded primary plus agent-isolation runners. The official snapshot lives under the ignored `evals/data/scifact/data/` directory after `scripts/prepare_scifact.py`; the small checked-in fixture under `evals/fixtures/scifact/` is for offline CI only.

Run the fixture dry-run without network or provider calls:

```powershell
python scripts/eval_scifact.py --dataset-dir evals/fixtures/scifact/data --manifest evals/fixtures/scifact/manifest.json --architecture all --profile smoke --offline --dry-run
```

Run the official bounded evaluation only after validating the manifest and reviewing the dry-run budget. See [../docs/evaluation.md](../docs/evaluation.md) for the protocol and artifact contract. No benchmark number is promoted here without a validated run.

The intended primary ablation boundary is:

1. single LLM
2. retrieval + judge
3. support + counter + judge
4. full multi-agent workflow

The follow-up isolation boundary is `C_SUPPORT_COUNTER`, `D1_AUDITOR`, `D2_CRITIC`, `D3_AUDITOR_CRITIC`, and `D4_CONDITIONAL_CRITIC`. Use `--architecture isolation` to run the same sample across that family. Auditor and critic are assurance mechanisms, not independent verdict producers.

For a budget-gated live isolation pass, review `python scripts/eval_agent_isolation.py --profile live-5 --dry-run` first, then run the exact same provider selection. Unknown provider quota is bounded by a hard maximum of 100 calls and a derived token ceiling; provider failures or incomplete paired architecture coverage produce `BLOCKED`, not a recommendation. No live-10 run is automatic.

External datasets and real providers are opt-in and must not run in the normal test suite.
