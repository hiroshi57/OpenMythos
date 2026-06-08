# OpenMythos — Sprint Plans
> 最終更新: 2026-06-08 (Sprint 45 完了 / 次回: Sprint 46) | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 45 完了: HuggingFace Hub 統合 → v0.48.0
> アーカイブ: Sprint 1〜9 → `docs/archive/sprint-plans-1-9.md`
>            Sprint 10〜19 → `docs/archive/sprint-plans-10-19.md`
>            Sprint 20〜25 → `docs/archive/sprint-plans-20-25.md`
>            Sprint 26〜35 → `docs/archive/sprint-plans-26-35.md`
>            Sprint 36〜43 → `docs/archive/sprint-plans-36-42.md`

---

## Sprint 全サマリー

| Sprint | テーマ | コアモジュール | テスト | Ver |
|--------|--------|--------------|--------|-----|
| 1〜9 | 基盤構築 / Serving / Marketing eval | `main.py` `serve/api.py` | 508 | v0.13 |
| 10〜19 | LLMO / Tool Use / RAG / ReAct / SEO | `llmo.py` `tools.py` `rag.py` | 1079 | v0.22 |
| 20 | **P1** 討議型集合知 | `debate.py` | 1138 | v0.23 |
| 21 | **P2** KPI 駆動自己改善 | `kpi_agent.py` | 1204 | v0.24 |
| 22 | **P3** ボトルネック発見・解消 | `profiler.py` | 1265 | v0.25 |
| 23 | **P4** 外部要因適応 | `external_signal.py` | 1325 | v0.26 |
| 24 | **P5** ミスから学習 | `error_memory.py` | 1365 | v0.27 |
| 25 | **P6** 継続的自己蒸留 | `self_distill.py` | 1408 | v0.28 |
| 26 | **P7** 長期記憶統合 | `long_term_memory.py` | 1450 | v0.29 |
| 27 | **P8** アンサンブル品質評価 | `ensemble_scorer.py` | 1490 | v0.30 |
| 28 | **P9** 適応型プロンプト進化 | `prompt_evolution.py` | 1530 | v0.31 |
| 29 | **P10** 自律タスク計画 | `task_planner.py` | 1570 | v0.32 |
| 30 | P1〜P10 統合オーケストレーター | `growing_ai_orchestrator.py` | 1617 | v0.33 |
| 31 | GPU LoRA SFT 統合 | `lora_trainer.py` | 1657 | v0.34 |
| 32 | エラーメモリ SQLite 永続化 | `error_memory.py` | 1697 | v0.35 |
| 33 | LongTermMemory FAISS ANN | `long_term_memory.py` | 1737 | v0.36 |
| 34 | MistakeGuardMiddleware | `error_memory.py` | 1777 | v0.37 |
| 35 | ベンチマーク強化 (10パターン KPI 計測) | `benchmark/growing_ai_bench.py` | 1822 | v0.38 |
| 36 | API ドキュメント整備 + CI ベンチ自動化 | `serve/api.py` `.github/workflows/bench.yml` | 1861 | v0.39 |
| 37 | **ベンチマーク結果可視化 + E2E 疎通テスト** | `benchmark/report.py` | 1922 | v0.40 |
| 38 | **GPU LoRA CosineScheduler 統合 + 実機検証基盤** | `open_mythos/lora_trainer.py` | 1963 | v0.41 |
| 39 | **ショーケースダッシュボード + Prometheus メトリクス + GitHub Pages** | `serve/dashboard.py` `serve/api.py` | 2012 | v0.42 |
| 40 | **OpenAI 互換 Streaming 強化 + /v1/completions** | `serve/api.py` | 2077 | v0.43 |
| 41 | **Function Calling 統合 (tools / tool_calls / tool ロール)** | `serve/api.py` | 2120 | v0.44 |
| 42 | **/v1/embeddings + セマンティック検索** | `open_mythos/main.py` `serve/api.py` | 2166 | v0.45 |
| 43 | **HermesOrchestrator: Layer 2 Ultracode Mode** | `open_mythos/hermes_orchestrator.py` `serve/api.py` | 2260 | v0.46 |
| 44 | **Vector DB 統合 + Instructor 構造化出力** | `open_mythos/skills/vector_store.py` `open_mythos/skills/instructor_extract.py` `serve/api.py` | 2322 | v0.47 |
| 45 | **HuggingFace Hub 統合** | `open_mythos/skills/hf_hub.py` `serve/api.py` | 2384 | v0.48 |

> **累計テスト数**: 2384 PASS (Sprint 45: +62)

---

## 「育つAI」10パターン (P1〜P10)

| # | パターン | Sprint | モジュール | API |
|---|---------|--------|-----------|-----|
| P1 | 討議型集合知 | 20 | `debate.py` | `/v1/debate/run` |
| P2 | KPI駆動自己改善 | 21 | `kpi_agent.py` | `/v1/kpi/*` |
| P3 | ボトルネック発見・解消 | 22 | `profiler.py` | `/v1/profile/*` |
| P4 | 外部要因適応 | 23 | `external_signal.py` | `/v1/signal/*` |
| P5 | ミスから学習 | 24 | `error_memory.py` | `/v1/mistakes/*` |
| P6 | 継続的自己蒸留 | 25 | `self_distill.py` | `/v1/distill/*` |
| P7 | 長期記憶統合 | 26 | `long_term_memory.py` | `/v1/memory/*` |
| P8 | アンサンブル品質評価 | 27 | `ensemble_scorer.py` | `/v1/ensemble/*` |
| P9 | 適応型プロンプト進化 | 28 | `prompt_evolution.py` | `/v1/evolve/*` |
| P10 | 自律タスク計画 | 29 | `task_planner.py` | `/v1/plan/*` |

---

## Sprint 45 詳細 (最新)

### Sprint 45: HuggingFace Hub 統合 — v0.48.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 45.1 | `open_mythos/skills/hf_hub.py` — HFModelInfo / HFDatasetInfo / HFHubClient | cc:完了 |
| 45.2 | `open_mythos/skills/hf_hub.py` — FastTokenizer / TokenizerResult | cc:完了 |
| 45.3 | `open_mythos/skills/hf_hub.py` — LoRAConfig / PEFTAdapter / PEFTTrainResult | cc:完了 |
| 45.4 | `open_mythos/skills/hf_hub.py` — EvalTask / EvalResult / LMEvaluator | cc:完了 |
| 45.5 | `serve/api.py` — `/v1/hf/search/models` `/v1/hf/search/datasets` `/v1/hf/model/{id}` `/v1/tokenize` `/v1/peft/estimate` `/v1/lm-eval` | cc:完了 |
| 45.T | `tests/test_sprint45.py` — 62 tests PASS (累計 2384) | cc:完了 |
| 45.V | PyPI v0.48.0 | cc:完了 |

---

## Sprint 44 詳細

### Sprint 44: Vector DB 統合 + Instructor 構造化出力 — v0.47.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 44.1 | `open_mythos/skills/vector_store.py` — VectorDocument / VectorStoreConfig / ChromaStore / QdrantStore / PineconeStore / FaissStore / VectorStoreFactory | cc:完了 |
| 44.2 | `open_mythos/skills/instructor_extract.py` — ExtractionSchema / ExtractionResult / InstructorExtractor | cc:完了 |
| 44.3 | `serve/api.py` — `/v1/vector-store/*` + `/v1/extract` + `/v1/extract/prompt` エンドポイント追加 | cc:完了 |
| 44.4 | `open_mythos/skills/__init__.py` — Vector DB / Instructor クラス全エクスポート追加 | cc:完了 |
| 44.T | `tests/test_sprint44.py` — 62 tests PASS (累計 2322) | cc:完了 |
| 44.V | PyPI v0.47.0 | cc:完了 |

---

## Sprint 43 詳細

### Sprint 43: HermesOrchestrator — Layer 2 Ultracode Mode — v0.46.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 43.1 | `open_mythos/hermes_orchestrator.py` — SubTask / SubAgentSpec / HermesAgentResult / VerificationResult / HermesReport | cc:完了 |
| 43.2 | `open_mythos/hermes_orchestrator.py` — TaskDecomposer / AgentSpawner / ParallelExecutor / ResultVerifier / ReportBuilder | cc:完了 |
| 43.3 | `open_mythos/hermes_orchestrator.py` — HermesOrchestrator (plan / spawn / verify / report / run / run_async) | cc:完了 |
| 43.4 | `serve/api.py` — `/v1/hermes/run` + `/v1/hermes/plan` エンドポイント追加 | cc:完了 |
| 43.5 | `open_mythos/__init__.py` — HermesOrchestrator 関連クラス全エクスポート追加 | cc:完了 |
| 43.T | `tests/test_sprint43.py` — 94 tests PASS (累計 2260) | cc:完了 |
| 43.V | PyPI v0.46.0 | cc:完了 |

---

## 技術的知見メモ

- `freqs_cis` は必ず `[:T]` スライス (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` に `.clamp(min=1e-6)` 必要 (float32 飽和防止)
- `store or Store()` は空ストア (len=0→falsy) を別インスタンスに差し替える → `is not None` チェックを使う
- `ConsensusEngine.score(texts)` が正しい API (build_consensus/compute_agreement は存在しない)
- テスト間レート制限干渉: `tests/conftest.py` の `reset_rate_limiter` fixture で全スイート実行時に自動リセット
