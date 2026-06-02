# Sprint 26〜35 アーカイブ — 「育つAI」P7〜P10 + 品質強化
> アーカイブ日: 2026-06-02

---

## Sprint 26: P7 長期記憶統合 — v0.29.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 26.1 | `long_term_memory.py` — `MemoryEntry` / `EpisodicStore` / `SemanticStore` / `MemoryRetrieval` | cc:完了 |
| 26.2 | `long_term_memory.py` — `LongTermMemoryAgent` (store/retrieve/consolidate) | cc:完了 |
| 26.3 | `serve/api.py` — `/v1/memory/store` / `/retrieve` / `/consolidate` | cc:完了 |
| 26.T | `tests/test_sprint26.py` — 42 tests PASS | cc:完了 |
| 26.V | PyPI v0.29.0 | cc:完了 |

## Sprint 27: P8 アンサンブル品質評価 — v0.30.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 27.1 | `ensemble_scorer.py` — `ScorerWeight` / `ScorerBreakdown` / `EnsembleScore` | cc:完了 |
| 27.2 | `ensemble_scorer.py` — `EnsembleScorer` (score/batch/rank/custom/adaptive) | cc:完了 |
| 27.3 | `serve/api.py` — `/v1/ensemble/score` / `/rank` / `/feedback` | cc:完了 |
| 27.T | `tests/test_sprint27.py` — 40 tests PASS | cc:完了 |
| 27.V | PyPI v0.30.0 | cc:完了 |

## Sprint 28: P9 適応型プロンプト進化 — v0.31.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 28.1 | `prompt_evolution.py` — `PromptGene` / `EvolutionConfig` / `EvolutionRound` / `EvolutionResult` | cc:完了 |
| 28.2 | `prompt_evolution.py` — `PromptEvolution.evolve()` (GA: 選択/交叉/変異/エリート/早期終了) | cc:完了 |
| 28.3 | `serve/api.py` — `/v1/evolve/run` | cc:完了 |
| 28.T | `tests/test_sprint28.py` — 40 tests PASS | cc:完了 |
| 28.V | PyPI v0.31.0 | cc:完了 |

## Sprint 29: P10 自律タスク計画 — v0.32.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 29.1 | `task_planner.py` — `Task` / `TaskGraph` / `TaskExecutionResult` / `TaskPlan` / `TaskPlanResult` | cc:完了 |
| 29.2 | `task_planner.py` — `TaskPlanner.decompose()` (DAG + wave 並列) | cc:完了 |
| 29.3 | `task_planner.py` — `TaskPlanner.execute()` (前段出力の自動受け渡し + KPI 判定) | cc:完了 |
| 29.4 | `serve/api.py` — `/v1/plan/decompose` / `/execute` | cc:完了 |
| 29.T | `tests/test_sprint29.py` — 40 tests PASS | cc:完了 |
| 29.V | PyPI v0.32.0 | cc:完了 |

## Sprint 30: 統合 GrowingAIOrchestrator — v0.33.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 30.1 | `growing_ai_orchestrator.py` — `PatternType` / `GrowthContext` / `PatternResult` / `OrchestratorResult` | cc:完了 |
| 30.2 | `growing_ai_orchestrator.py` — `PatternSelector` (キーワードベース選択) | cc:完了 |
| 30.3 | `growing_ai_orchestrator.py` — `GrowingAIOrchestrator.run()` (選択・実行・統合) | cc:完了 |
| 30.4 | `serve/api.py` — `POST /v1/grow/run` | cc:完了 |
| 30.T | `tests/test_sprint30.py` — 47 tests PASS | cc:完了 |
| 30.V | PyPI v0.33.0 | cc:完了 |

## Sprint 31: GPU LoRA SFT 統合 — v0.34.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 31.1 | `lora_trainer.py` — `LoraTrainerConfig` / `DistillInMemoryDataset` / `collate_distill` | cc:完了 |
| 31.2 | `lora_trainer.py` — `LoraTrainer.train()` (GPU: 実訓練 / CPU: シミュレーション自動選択) | cc:完了 |
| 31.3 | `self_distill.py` — `SelfDistillConfig.sft_backend` / `SelfDistillResult.best_output` / `run()` 強化 | cc:完了 |
| 31.4 | `growing_ai_orchestrator.py` — DISTILL パターンバグ修正 | cc:完了 |
| 31.T | `tests/test_sprint31.py` — 40 tests PASS | cc:完了 |
| 31.V | PyPI v0.34.0 | cc:完了 |

## Sprint 32: エラーメモリ永続化 — v0.35.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 32.1 | `error_memory.py` — `ErrorMemoryStore(backend="sqlite", db_path=...)` SQLite バックエンド | cc:完了 |
| 32.2 | `error_memory.py` — `export_jsonl()` / `export_records()` / `save_jsonl()` / `import_jsonl()` | cc:完了 |
| 32.3 | `error_memory.py` — `clear()` / `close()` / `total` (SQLite COUNT) | cc:完了 |
| 32.4 | `serve/api.py` — `GET /v1/mistakes/export` + `DELETE /v1/mistakes/clear` | cc:完了 |
| 32.5 | `serve/api.py` — `MISTAKES_BACKEND` / `MISTAKES_DB_PATH` 環境変数対応 | cc:完了 |
| 32.T | `tests/test_sprint32.py` — 40 tests PASS | cc:完了 |
| 32.V | PyPI v0.35.0 | cc:完了 |

## Sprint 33: LongTermMemory FAISS ANN インデックス — v0.36.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 33.1 | `long_term_memory.py` — `_text_to_vector()` ハッシュ TF-IDF L2 正規化 | cc:完了 |
| 33.2 | `long_term_memory.py` — `ANNIndex` (FAISS IndexFlatIP + linear fallback) | cc:完了 |
| 33.3 | `long_term_memory.py` — ハイブリッド検索 (FAISS × TF-IDF) | cc:完了 |
| 33.4 | `long_term_memory.py` — `_rebuild_ann()` / evict / consolidate 後の再構築 | cc:完了 |
| 33.5 | `long_term_memory.py` — `LongTermMemoryAgent(ann_backend=...)` | cc:完了 |
| 33.T | `tests/test_sprint33.py` — 40 tests PASS | cc:完了 |
| 33.V | PyPI v0.36.0 | cc:完了 |

## Sprint 34: MistakeGuardMiddleware — v0.37.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 34.1 | `error_memory.py` — `GuardMiddlewareConfig` | cc:完了 |
| 34.2 | `error_memory.py` — `MistakeGuardMiddleware` + `is not None` バグ修正 | cc:完了 |
| 34.3 | `serve/api.py` — `_MistakeGuardHTTPMiddleware` 全 POST/PUT 透過チェック | cc:完了 |
| 34.4 | `serve/api.py` — `GET /v1/guard/stats` / `POST /v1/guard/refresh` | cc:完了 |
| 34.T | `tests/test_sprint34.py` — 40 tests PASS | cc:完了 |
| 34.V | PyPI v0.37.0 | cc:完了 |

## Sprint 35: ベンチマーク強化 — v0.38.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 35.1 | `benchmark/growing_ai_bench.py` — `PatternBenchResult` / P1〜P10 bench 関数 | cc:完了 |
| 35.2 | `benchmark/growing_ai_bench.py` — `GrowingAIBenchmark` / `BenchmarkReport` | cc:完了 |
| 35.3 | `benchmark/growing_ai_bench.py` — CLI (`--patterns` / `--verbose` / `--output`) | cc:完了 |
| 35.T | `tests/test_sprint35.py` — 45 tests PASS | cc:完了 |
| 35.V | PyPI v0.38.0 | cc:完了 |
