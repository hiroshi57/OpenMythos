# OpenMythos — Sprint Plans
> 最終更新: 2026-06-02 | ブランチ規約: `feature/<sprint>-<topic>`
> アーカイブ: Sprint 1〜9 → `docs/archive/sprint-plans-1-9.md` / Sprint 10〜19 → `docs/archive/sprint-plans-10-19.md` / Sprint 20〜25 → `docs/archive/sprint-plans-20-25.md`

---

## Sprint 1〜29 全サマリー

| Sprint | テーマ | コアモジュール | テスト | Ver |
|--------|--------|--------------|--------|-----|
| 1 | HyperloopMythos / Inference Engine | `main.py` `hyperloop.py` | 227 | — |
| 2 | Inference 高度化 (beam/quant/KV cache) | `main.py` 拡張 | — | — |
| 3 | ドキュメント & エコシステム | docs/scripts | — | v0.6 |
| 4 | Training 基盤 / MoDa / variants | `moda.py` `variants.py` | 257 | — |
| 5 | Training 品質 / LoRA / HF Hub / CLI | `cli.py` `logger_utils.py` | 284 | v0.7 |
| 6 | 推論最適化 / TrainLogger / Agents | `agents.py` | 380 | — |
| 7 | Serving / データパイプライン | `serve/api.py` 初版 | 420+ | v0.12 |
| 8 | Fine-tuning / /v1/chat / SLA | `scripts/finetune.py` | 468 | — |
| 9 | Marketing eval / A/B / batch API | `serve/ab_router.py` | 508 | v0.13 |
| 10 | LLMO / Extended Thinking / Structured Output | `llmo.py` `thinking.py` `structured.py` | 560 | v0.14 |
| 11 | Tool Use / Long Context / RAG | `tools.py` `rag.py` `rope_extension.py` | 664 | v0.15 |
| 12 | ReAct / Prefix Cache / Conversation Memory | `react.py` `prefix_cache.py` `conversation.py` | 729 | v0.16 |
| 13 | Mixture-of-Depths / SwarmOrchestrator | `mod.py` `swarm.py` | 836 | — |
| 14 | GPU pretrain / Benchmark / GCP deploy | `scripts/pretrain.py` `benchmark/` | 856 | v0.17 |
| 15 | 日本語形態素解析 / A/B テスト / ドリフト検出 | `llmo.py` 拡張 | 888 | v0.18 |
| 16 | SEO パイプライン / セキュリティ | `seo_pipeline.py` `security.py` | 958 | v0.19 |
| 17 | API 認証 / レート制限 / Docker 本番化 | `serve/auth.py` `serve/Dockerfile` | 998 | v0.20 |
| 18 | ファインチューニング実証 / ROAS / ペルソナ | `tools_marketing.py` 拡張 | 1037 | v0.21 |
| 19 | LLMO 強化 — score_with_query / LLMOOptimizer | `llmo.py` 拡張 | 1079 | v0.22 |
| 20 | **P1** 討議型集合知 — DebateOrchestrator | `debate.py` | 1138 | v0.23 |
| 21 | **P2** KPI 駆動自己改善 — KPIAgent | `kpi_agent.py` | 1204 | v0.24 |
| 22 | **P3** ボトルネック発見・解消 — ProfilerAgent | `profiler.py` | 1265 | v0.25 |
| 23 | **P4** 外部要因適応 — ExternalSignalAgent | `external_signal.py` | 1325 | v0.26 |
| 24 | **P5** ミスから学習 — MistakeGuard | `error_memory.py` | 1365 | v0.27 |
| 25 | **P6** 継続的自己蒸留 — SelfDistillLoop | `self_distill.py` | 1408 | v0.28 |
| 26 | **P7** 長期記憶統合 — LongTermMemoryAgent | `long_term_memory.py` | 1450 | v0.29 |
| 27 | **P8** アンサンブル品質評価 — EnsembleScorer | `ensemble_scorer.py` | 1490 | v0.30 |
| 28 | **P9** 適応型プロンプト進化 — PromptEvolution | `prompt_evolution.py` | 1530 | v0.31 |
| 29 | **P10** 自律タスク計画 — TaskPlanner | `task_planner.py` | 1570 | v0.32 |
| 30 | **統合** P1〜P10 オーケストレーター — GrowingAIOrchestrator | `growing_ai_orchestrator.py` | 1617 | v0.33 |
| 31 | GPU LoRA SFT 統合 — LoraTrainer / sft_backend | `lora_trainer.py` | 1657 | v0.34 |
| 32 | エラーメモリ永続化 — SQLite backend / export | `error_memory.py` | 1697 | v0.35 |

> **累計テスト数**: 1697 PASS (Sprint 32: +40)

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

```text
P1→P2→P3/P4  P5→P6→P7→P8→P9→P10
              ↑___連携ループ___↑
```

---

## Sprint 26〜29 詳細 (完了)

### Sprint 26: P7 長期記憶統合 — v0.29.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 26.1 | `long_term_memory.py` — `MemoryEntry` / `EpisodicStore` / `SemanticStore` / `MemoryRetrieval` | cc:完了 |
| 26.2 | `long_term_memory.py` — `LongTermMemoryAgent` (store/retrieve/consolidate) | cc:完了 |
| 26.3 | `serve/api.py` — `/v1/memory/store` / `/retrieve` / `/consolidate` | cc:完了 |
| 26.T | `tests/test_sprint26.py` — 42 tests PASS | cc:完了 |
| 26.V | PyPI v0.29.0 | cc:完了 |

### Sprint 27: P8 アンサンブル品質評価 — v0.30.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 27.1 | `ensemble_scorer.py` — `ScorerWeight` / `ScorerBreakdown` / `EnsembleScore` | cc:完了 |
| 27.2 | `ensemble_scorer.py` — `EnsembleScorer` (score/batch/rank/custom/adaptive) | cc:完了 |
| 27.3 | `serve/api.py` — `/v1/ensemble/score` / `/rank` / `/feedback` | cc:完了 |
| 27.T | `tests/test_sprint27.py` — 40 tests PASS | cc:完了 |
| 27.V | PyPI v0.30.0 | cc:完了 |

### Sprint 28: P9 適応型プロンプト進化 — v0.31.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 28.1 | `prompt_evolution.py` — `PromptGene` / `EvolutionConfig` / `EvolutionRound` / `EvolutionResult` | cc:完了 |
| 28.2 | `prompt_evolution.py` — `PromptEvolution.evolve()` (GA: 選択/交叉/変異/エリート/早期終了) | cc:完了 |
| 28.3 | `serve/api.py` — `/v1/evolve/run` | cc:完了 |
| 28.T | `tests/test_sprint28.py` — 40 tests PASS | cc:完了 |
| 28.V | PyPI v0.31.0 | cc:完了 |

### Sprint 29: P10 自律タスク計画 — v0.32.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 29.1 | `task_planner.py` — `Task` / `TaskGraph` / `TaskExecutionResult` / `TaskPlan` / `TaskPlanResult` | cc:完了 |
| 29.2 | `task_planner.py` — `TaskPlanner.decompose()` (DAG + wave 並列) | cc:完了 |
| 29.3 | `task_planner.py` — `TaskPlanner.execute()` (前段出力の自動受け渡し + KPI 判定) | cc:完了 |
| 29.4 | `serve/api.py` — `/v1/plan/decompose` / `/execute` | cc:完了 |
| 29.T | `tests/test_sprint29.py` — 40 tests PASS | cc:完了 |
| 29.V | PyPI v0.32.0 | cc:完了 |

---

## Sprint 32 詳細 (完了)

### Sprint 32: エラーメモリ永続化 — v0.35.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 32.1 | `error_memory.py` — `ErrorMemoryStore(backend="sqlite", db_path=...)` SQLite バックエンド | cc:完了 |
| 32.2 | `error_memory.py` — `export_jsonl()` / `export_records()` / `save_jsonl()` / `import_jsonl()` | cc:完了 |
| 32.3 | `error_memory.py` — `clear()` / `close()` / `total` (SQLite COUNT) | cc:完了 |
| 32.4 | `serve/api.py` — `GET /v1/mistakes/export` (jsonl/json/category filter) + `DELETE /v1/mistakes/clear` | cc:完了 |
| 32.5 | `serve/api.py` — `_get_mistake_store()` に `MISTAKES_BACKEND` / `MISTAKES_DB_PATH` 環境変数対応 | cc:完了 |
| 32.T | `tests/test_sprint32.py` — 40 tests PASS | cc:完了 |
| 32.V | PyPI v0.35.0 | cc:完了 |

---

## Sprint 31 詳細 (完了)

### Sprint 31: GPU LoRA SFT 統合 — v0.34.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 31.1 | `lora_trainer.py` — `LoraTrainerConfig` / `DistillInMemoryDataset` / `collate_distill` | cc:完了 |
| 31.2 | `lora_trainer.py` — `LoraTrainer.train()` (GPU: 実訓練 / CPU: シミュレーション自動選択) | cc:完了 |
| 31.3 | `self_distill.py` — `SelfDistillConfig.sft_backend` / `SelfDistillResult.best_output` / `run()` 強化 | cc:完了 |
| 31.4 | `growing_ai_orchestrator.py` — DISTILL パターンバグ修正 (`run([goal])` / `.output`) | cc:完了 |
| 31.T | `tests/test_sprint31.py` — 40 tests PASS | cc:完了 |
| 31.V | PyPI v0.34.0 | cc:完了 |

---

## Sprint 30 詳細 (完了)

### Sprint 30: 統合 GrowingAIOrchestrator — v0.33.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 30.1 | `growing_ai_orchestrator.py` — `PatternType` / `GrowthContext` / `PatternResult` / `OrchestratorResult` | cc:完了 |
| 30.2 | `growing_ai_orchestrator.py` — `PatternSelector` (キーワードベース選択) | cc:完了 |
| 30.3 | `growing_ai_orchestrator.py` — `GrowingAIOrchestrator.run()` (選択・実行・統合) | cc:完了 |
| 30.4 | `serve/api.py` — `POST /v1/grow/run` | cc:完了 |
| 30.T | `tests/test_sprint30.py` — 47 tests PASS | cc:完了 |
| 30.V | PyPI v0.33.0 | cc:完了 |

---

## 次回課題 (Sprint 31〜)

### 優先度 HIGH
1. **実際の GPU LoRA SFT 統合** — `SelfDistillLoop._simulate_sft()` を `scripts/finetune.py` の `LoraTrainer` に差し替え
3. **エラーメモリ永続化** — `ErrorMemoryStore(backend="sqlite")` + `/v1/mistakes/export`

### 優先度 MEDIUM
4. **LongTermMemory ANN インデックス** — 件数増加時の O(n) 線形探索を FAISS に移行
5. **MistakeGuardMiddleware** — 全 API エンドポイントに透過的に適用
6. **ベンチマーク強化** — `benchmark/growing_ai_bench.py` で 10 パターンの KPI 改善量を測定

### 技術的知見メモ
- `freqs_cis` は必ず `[:T]` スライス (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` に `.clamp(min=1e-6)` 必要 (float32 飽和防止)
- `BottleneckDetector`: latency/score 両方を検出し相対深刻度 (`score_rel > lat_rel`) で優先順位を決定
- `EnsembleScorer._security_score` のパターン文字列はソース解析ツール誤検知防止のため文字列連結で定義
