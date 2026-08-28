# SciFact offline smoke reference

This compact reference is derived from the local deterministic smoke artifact
`scifact-smoke-20260825T163113Z-2c42b8a2`. It is retained as project evidence;
the raw run directory remains ignored.

- Dataset: SciFact `dev`, 10 examples
- Revision: `allenai/scifact@68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`
- Release archive SHA-256: `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`
- Mode: `OFFLINE_FIXTURE`; model: `deterministic-eval-fixture-v1`
- Retrieval: `bm25_lexical_v1`, top-k 5

| Architecture | Accuracy | Macro F1 | Evidence F1 | Avg calls | Avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_SINGLE_LLM | 0.5000 | 0.2222 | N/A | 1.0000 | 40.6000 |
| B_RETRIEVAL_JUDGE | 0.5000 | 0.2222 | 0.3750 | 1.0000 | 535.4000 |
| C_SUPPORT_COUNTER | 0.5000 | 0.2222 | 0.3750 | 2.0000 | 1094.8000 |
| D_FULL_VERICLAIM | 0.5000 | 0.2222 | 0.3750 | 4.0000 | 1294.4000 |

Calibration status is `CALIBRATION_SAMPLE_TOO_SMALL` at population 10. This
fixture run does not establish live paired architecture superiority, a
production promotion, or calibrated confidence.
