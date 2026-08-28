# Evaluation protocol

## Research question

The evaluation asks which bounded component adds value under one reproducible SciFact configuration: does closed-corpus retrieval improve over a single LLM, does explicit support/counter analysis improve over retrieval plus judgment, and do targeted assurance mechanisms justify their calls, tokens, and latency? Results are claims about the recorded sample and configuration only.

## Dataset and versioning

The adapter uses the official AllenAI SciFact release. `scripts/prepare_scifact.py` records the AllenAI repository revision, immutable archive SHA-256, per-file SHA-256, row counts, split names, and claim-level label distribution in `evals/data/scifact/manifest.json`. The labeled `dev` split is the default evaluation population; the public test split has no labels and is not evaluated. A changed file fails manifest validation instead of being silently re-downloaded.

SciFact’s source labels are `SUPPORT`, `CONTRADICT`, and empty evidence. They map explicitly to `SUPPORTED`, `REFUTED`, and `INSUFFICIENT_EVIDENCE`. Mixed document annotations map to `MIXED`. `MIXED`, `INSUFFICIENT_EVIDENCE`, and `NON_VERIFIABLE` are explicit abstentions for coverage reporting.

## Architectures and fair controls

| Architecture | Components | Evidence available to the LLM |
| --- | --- | --- |
| A_SINGLE_LLM | one fixed judge | claim only |
| B_RETRIEVAL_JUDGE | deterministic BM25 retrieval + one judge | top-k corpus sentences |
| C_SUPPORT_COUNTER | retrieval + support/counter classifier + judge | top-k sentences plus classifier stances |
| D_FULL_VERICLAIM | retrieval + classifier + auditor + judge + critic | bounded intermediate outputs |

The follow-up isolation family keeps the same claims and retrieval boundary while splitting the combined C-to-D effect:

| Architecture | Components |
| --- | --- |
| C_SUPPORT_COUNTER | retrieval + support/counter classifier + judge |
| D1_AUDITOR | C + advisory evidence auditor |
| D2_CRITIC | C + always-on advisory critic |
| D3_AUDITOR_CRITIC | C + auditor + critic |
| D4_CONDITIONAL_CRITIC | C + critic only when deterministic risk signals fire |

The auditor assesses relevance, directness, scope, quantifier, temporal compatibility, usability, and bounded issues. It does not select the final evidence set or produce a verdict. The critic returns only `PASS` or `CHALLENGE` with a bounded reason and evidence IDs. A challenge triggers at most one judge recheck; a failed recheck abstains. Hard provenance/schema failures remain deterministic validator concerns.

All architectures use the same labeled split, deterministic seed, selected claim IDs, retrieval configuration for B/C/D, temperature 0, fixed prompt versions, and fixed provider/model profile. Gold labels, gold evidence, and annotation rationale are evaluation-only and never appear in inference prompts. Deterministic validation is not counted as an LLM call. OpenRouter, paid models, dynamic model routing, and fallback providers are excluded. The isolated live benchmark requires one explicitly selected provider/model for every stage; a model substitution or provider failure invalidates the paired comparison.

Retrieval is lexical BM25 (`bm25_lexical_v1`) over the local SciFact corpus. No vector database, hosted embedding, OpenAlex, Crossref, or arbitrary URL fetch is used by this benchmark. Document Recall@K, sentence Recall@K, and MRR measure retrieval; final evidence precision/recall/F1 measure selected sentence IDs separately.

## Metrics

Claim metrics include accuracy, macro precision/recall/F1, per-class metrics, and a confusion matrix. Macro F1 is primary because the dev labels are imbalanced and the benchmark must expose each class. Abstention metrics include coverage, abstention rate, selective accuracy, and selective error rate; primary accuracy penalizes abstention when it is wrong.

Calibration reports Brier score, ten-bin ECE, and bin counts. Confidence means confidence in the run verdict given the evidence, not objective truth probability. Fewer than 20 examples produces `CALIBRATION_SAMPLE_TOO_SMALL`. Production’s current `MIXED` value of 0.65 is documented as a heuristic and is not treated as calibrated.

The deterministic `unsupported_verdict_rate` flags a non-abstaining output with no selected evidence from the same run’s retrieved sentence set. Error taxonomy entries include retrieval miss, evidence selection miss, stance classification error, quantifier mismatch, excessive abstention, failed-to-abstain, schema parse failure, provider failure, and unknown. High-confidence wrong and low-confidence correct rows are exported without LLM-generated explanations.

`evals/fixtures/scope_quantifier.json` is a separate deterministic fixture for always/never/all/percent/multiplier/conditional scope handling. It does not claim SciFact performance.

## Efficiency and free-tier constraints

Every result records provider, configured/actual model, prompt version, stage invocations, actual provider calls, cache hits/misses, input/output/total tokens charged to the current run, cached source usage, latency, failures, retries, and fallbacks. The client maintains one authoritative telemetry record for every provider inference attempt, including failed responses; budget-stop and provider-block events are also retained with zero provider calls. For a budgeted live run, the budget gate’s actual attempt/failure counters are cross-checked against that ledger, so an attempt cannot disappear when a stop occurs between stages. `avg_llm_calls` is the logical stage-invocation count and is distinct from `avg_provider_calls`; cache replay contributes to the former but not the latter. Cache-hit latency and tokens are not counted as current-run provider cost. `API_COST_USD=N/A` is used because the run does not infer a dollar price. The usage class is `FREE_TIER_USAGE`; paid API usage is forbidden. The smoke profile selects 10 claims. The pilot profile selects 50 claims only after its dry-run estimate is reviewed. Extended evaluation requires an explicit flag and is never automatic.

Structured response caching is keyed by cache version, semantic stage scope, provider, configured model, prompt version, generation parameters, and an input hash that includes the dataset revision, retrieval configuration, stage input, and prompt. Semantically identical classifier, auditor, and judge stages may share a cache entry across isolation architectures; their original architecture remains in per-prediction telemetry. Only parsed response objects and usage metadata are cached; prompts, headers, keys, raw provider payloads, and hidden reasoning are never written. Changing any key input invalidates the entry.

## Reproduction commands

```powershell
python scripts/prepare_scifact.py --root evals/data/scifact --download
python scripts/eval_scifact.py --architecture all --profile smoke --dry-run
python scripts/eval_scifact.py --architecture all --profile smoke --offline
python scripts/eval_scifact.py --architecture all --profile pilot --dry-run
python scripts/eval_scifact.py --architecture isolation --profile smoke --offline
```

Live runs require an explicit enabled, authorized key for the selected fixed provider in the ignored `.env`, and the exact fixed model. Query availability and review the dry-run before invoking a live profile. Live provider failures are recorded; the runner never silently substitutes OpenRouter or another model.

The agent-isolation live pass is separately budget-gated:

```powershell
python scripts/eval_agent_isolation.py --profile live-5 --dry-run
python scripts/eval_agent_isolation.py --profile live-5
```

The default is Gemini for every stage. A provider can be selected explicitly for every stage, for example `--core-provider groq --classifier-provider groq --critic-provider groq`; all three selections must match for the fixed benchmark. OKMD is also supported as an explicit fixed provider with model `deepseek-v4-flash`; it is never a fallback. An OKMD replay first runs two identical bounded probes and requires the same configured and actual model plus known quota metadata. The preflight then probes the existing structured cache, recalculates the historical average from successful non-offline records, reserves the bounded critic/recheck branches, and requires at least 10% call and token headroom before live inference. The gate denies before inference when a selected provider is unavailable or disabled, when known quota use would exceed 50% of remaining requests/tokens, or when the unknown-quota conservative ceilings would be exceeded. The unknown-quota ceilings are at most 100 provider calls and a token ceiling derived from the recorded historical average tokens/call multiplied by 100 and a safety factor below 1. A provider rate-limit/quota or incomplete-response error stops the run; an incomplete, substituted, or failed paired run is `BLOCKED` and is not used for a component recommendation. `live-10` is not automatic.

### Resumable windowed live-5

When a fixed free provider passes a short probe but is unreliable under benchmark payloads, use `scripts/eval_agent_isolation_resumable.py`. The reliability calibration is transport-only: it selects three claims outside the locked sample `13, 208, 268, 314, 549`, chooses payloads near the target size, and records input size, configured output limit, timeout, finish reason, completeness, latency, actual model, and safe quota metadata. It does not tune prompts or measure SciFact quality.

```powershell
python scripts/eval_agent_isolation_resumable.py --provider okmd --calibrate-only --reliability-profile evals/results/scifact-reliability-live-5-okmd.json
python scripts/eval_agent_isolation_resumable.py --provider okmd --reliability-profile evals/results/scifact-reliability-live-5-okmd.json --window-max-calls 8
python scripts/eval_agent_isolation_resumable.py --provider okmd --resume --checkpoint evals/results/scifact-live-5-resumable-<run>/checkpoint.json --reliability-profile evals/results/scifact-reliability-live-5-okmd.json --window-max-calls 8
```

The live run is one immutable run over the five locked claims and five isolation architectures. Each window is sequential, has no immediate retry, applies configurable pacing, stops on provider failure, and atomically checkpoints successful stages. The cache namespace includes dataset/retrieval, provider/model, prompt/generation, timeout/pacing, and reliability-profile configuration hashes. A changed envelope requires a new reliability profile and new run. The checkpoint records window ID/times, provider and configured/actual model, configuration hashes, attempts, successes, tokens, cache hits, failure categories, and a first-missing-pair cursor. If any later response changes actual model, the run is `BENCHMARK_INVALID_MODEL_DRIFT` and old/new results are not combined. After three bounded low-progress failures, it stops as `PROVIDER_UNSUITABLE_FOR_BENCHMARK`. Only a complete `25/25` run is eligible for C/D1/D2/D3/D4 comparison; prior or incomplete artifacts are diagnostic only.

## Result artifacts

Each run is stored under `evals/results/<benchmark_run_id>/`: `manifest.json`, `metrics.json`, `predictions.jsonl`, `provider_usage.json`, `critic_effects.jsonl`, `auditor_effects.jsonl`, `errors.jsonl`, `error_analysis.json`, CSV exports, and `summary.md`. Predictions retain claim IDs, mapped labels, confidence/correctness, retrieved document IDs, selected sentence IDs, gold IDs for evaluation-only analysis, usage, and bounded errors. The causal effect exports retain before/after evidence IDs, auditor effect categories, critic PASS/CHALLENGE decisions, bounded rechecks, and wrong-to-right/right-to-wrong counts. These IDs are sufficient for a later Evidence Graph without building a graph database now.

## Limitations

SciFact claims are already atomic, so claim analysis and decomposition are not isolated by this ablation. A small free-tier pilot cannot establish general superiority, and smoke calibration is intentionally insufficient. BM25 and fixed prompts measure this closed-corpus protocol, not open-web retrieval. The isolation family measures auditor/critic contribution under separate prompts, but it does not prove general production benefit. Optional confidence intervals or McNemar tests are not reported unless the run explicitly computes them.
