# OpenMythos — Sprint Plans
> 最終更新: 2026-06-02 | ブランチ規約: `feature/<sprint>-<topic>`
> アーカイブ: Sprint 1〜9 → `docs/archive/sprint-plans-1-9.md`
>            Sprint 10〜19 → `docs/archive/sprint-plans-10-19.md`
>            Sprint 20〜25 → `docs/archive/sprint-plans-20-25.md`
>            Sprint 26〜35 → `docs/archive/sprint-plans-26-35.md`

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

> **累計テスト数**: 1822 PASS (Sprint 35: +45)

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

## Sprint 35 詳細 (最新)

### Sprint 35: ベンチマーク強化 — v0.38.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 35.1 | `benchmark/growing_ai_bench.py` — `PatternBenchResult` / P1〜P10 bench 関数 | cc:完了 |
| 35.2 | `benchmark/growing_ai_bench.py` — `GrowingAIBenchmark` / `BenchmarkReport` (run_all/print_table/save/load) | cc:完了 |
| 35.3 | CLI — `--patterns` / `--verbose` / `--output` / 実測: 10/10 成功・平均 +10.5% | cc:完了 |
| 35.T | `tests/test_sprint35.py` — 45 tests PASS | cc:完了 |
| 35.V | PyPI v0.38.0 | cc:完了 |

### Sprint 34: MistakeGuardMiddleware — v0.37.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 34.1 | `error_memory.py` — `GuardMiddlewareConfig` / `MistakeGuardMiddleware` | cc:完了 |
| 34.2 | `serve/api.py` — `_MistakeGuardHTTPMiddleware` + `/v1/guard/stats` / `/v1/guard/refresh` | cc:完了 |
| 34.T | `tests/test_sprint34.py` — 40 tests PASS | cc:完了 |
| 34.V | PyPI v0.37.0 | cc:完了 |

---

## 技術的知見メモ

- `freqs_cis` は必ず `[:T]` スライス (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` に `.clamp(min=1e-6)` 必要 (float32 飽和防止)
- `store or Store()` は空ストア (len=0→falsy) を別インスタンスに差し替える → `is not None` チェックを使う
- `ConsensusEngine.score(texts)` が正しい API (build_consensus/compute_agreement は存在しない)

---

## 次スプリント候補

- **API Swagger UI 動作確認** — `uvicorn serve.api:app --reload` → `http://localhost:8000/docs`
- **ベンチマーク定期実行 CI 統合** — GitHub Actions で週次 `growing_ai_bench.py` 実行
- **GPU 実機での LoRA SFT 検証** — CUDA 環境で `LoraTrainer._real_train()` 動作確認
