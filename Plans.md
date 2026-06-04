# OpenMythos — Sprint Plans
> 最終更新: 2026-06-04 (Sprint 40 完了 / 次回: Sprint 41) | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 40 完了: OpenAI 互換 Streaming 強化 + /v1/completions → v0.43.0
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
| 36 | API ドキュメント整備 + CI ベンチ自動化 | `serve/api.py` `.github/workflows/bench.yml` | 1861 | v0.39 |
| 37 | **ベンチマーク結果可視化 + E2E 疎通テスト** | `benchmark/report.py` | 1922 | v0.40 |
| 38 | **GPU LoRA CosineScheduler 統合 + 実機検証基盤** | `open_mythos/lora_trainer.py` | 1963 | v0.41 |
| 39 | **ショーケースダッシュボード + Prometheus メトリクス + GitHub Pages** | `serve/dashboard.py` `serve/api.py` | 2012 | v0.42 |
| 40 | **OpenAI 互換 Streaming 強化 + /v1/completions** | `serve/api.py` | 2077 | v0.43 |

> **累計テスト数**: 2077 PASS (Sprint 40: +65)

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

## Sprint 40 詳細 (最新)

### Sprint 40: OpenAI 互換 Streaming 強化 + /v1/completions — v0.43.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 40.1 | `serve/api.py` — サンプリングヘルパー (`_apply_top_p` / `_apply_sampling_penalties` / `_collect_logprobs` / `_check_stop` / `_truncate_at_stop`) 追加 | cc:完了 |
| 40.2 | `serve/api.py` — `ChatRequest` 拡張: `stop` / `n` / `logprobs` / `top_logprobs` / `presence_penalty` / `frequency_penalty` | cc:完了 |
| 40.3 | `serve/api.py` — SSE ストリーミング強化: `stop` 対応 / `finish_reason:"length"` / 最終 usage chunk / `model` フィールド | cc:完了 |
| 40.4 | `serve/api.py` — `/v1/completions` 新規追加 (`echo` / `stop` / `n` / `logprobs` / `stream` 対応) | cc:完了 |
| 40.T | `tests/test_sprint40.py` — 65 tests PASS (累計 2077) | cc:完了 |
| 40.V | PyPI v0.43.0 | cc:完了 |

## Sprint 39 詳細 (完了)

### Sprint 39: ショーケースダッシュボード + Prometheus メトリクス + GitHub Pages — v0.42.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 39.1 | `serve/dashboard.py` — `build_showcase_dashboard()` / `save_dashboard()` / FastAPI router (`/dashboard`) | cc:完了 |
| 39.2 | `serve/api.py` — `/metrics` Prometheus エンドポイント + `prometheus_client` 統合 | cc:完了 |
| 39.3 | `serve/monitor.py` — `build_monitor_dashboard_html()` HTML 化 + `/monitor/dashboard` HTMLResponse | cc:完了 |
| 39.4 | `.github/workflows/pages.yml` — GitHub Pages 自動デプロイ (master push + workflow_dispatch) | cc:完了 |
| 39.T | `tests/test_sprint39.py` — 49 tests PASS (累計 2012) | cc:完了 |
| 39.V | PyPI v0.42.0 | cc:完了 |

## Sprint 38 詳細 (完了)

### Sprint 38: GPU LoRA SFT — CosineScheduler 統合 + 実機検証基盤 — v0.41.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 38.1 | `lora_trainer.py` — `LoraTrainerConfig` に `warmup_steps` / `min_lr_ratio` / `use_scheduler` 追加 | cc:完了 |
| 38.2 | `lora_trainer.py` — `_real_train()` に `CosineAnnealingLR` + 線形 warmup 統合 | cc:完了 |
| 38.3 | `lora_trainer.py` — `cosine_t_max()` / `get_current_lr()` ヘルパー追加 | cc:完了 |
| 38.T | `tests/test_sprint38.py` — 41 tests PASS / 3 GPU-skip (累計 1963) | cc:完了 |
| 38.V | PyPI v0.41.0 | cc:完了 |

## Sprint 37 詳細 (完了)

### Sprint 37: ベンチマーク結果可視化 + E2E 疎通テスト — v0.40.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 37.1 | `benchmark/report.py` — `ReportGenerator.to_markdown()` / `to_html()` / `save_*()` | cc:完了 |
| 37.2 | `benchmark/report.py` — `load_reports()` / `trend_table(json_paths, n)` トレンド表生成 | cc:完了 |
| 37.3 | `tests/test_sprint37.py` — P2〜P10 + guard + grow TestClient 疎通テスト (20 tests) | cc:完了 |
| 37.4 | `.github/workflows/bench.yml` — HTML レポート生成ステップ + artifact upload (retention 90日) | cc:完了 |
| 37.T | `tests/test_sprint37.py` — 61 tests PASS (累計 1922) | cc:完了 |
| 37.V | PyPI v0.40.0 | cc:完了 |

## Sprint 36 詳細 (完了)

### Sprint 36: API ドキュメント整備 + CI ベンチマーク自動化 — v0.39.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 36.1a | `serve/api.py` — `version="0.38.0"` 確認・旧バージョン除去 | cc:完了 |
| 36.1b | `serve/api.py` — `openapi_tags` 全 23 タグ整備 (P1〜P10 + grow + guard) | cc:完了 |
| 36.2a | `.github/workflows/bench.yml` — ファイル存在・YAML 構文 | cc:完了 |
| 36.2b | `.github/workflows/bench.yml` — 週次 cron (月曜 09:00 JST) + workflow_dispatch (patterns/verbose 入力) | cc:完了 |
| 36.2c | `.github/workflows/bench.yml` — jobs: checkout / setup-python / bench 実行 / artifact 保存 / GITHUB_STEP_SUMMARY | cc:完了 |
| 36.T | `tests/test_sprint36.py` — 39 tests PASS | cc:完了 |
| 36.V | PyPI v0.39.0 | cc:完了 |

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

## Sprint 37 計画 (次回)

### Sprint 37: ベンチマーク結果可視化 + E2E 疎通テスト — v0.40.0

**テーマ**: Sprint 35/36 で整備したベンチマーク基盤の「見える化」と、P1〜P10 全 API エンドポイントの結合テスト

| task-id | 説明 | 優先 |
|---------|------|------|
| 37.1 | `benchmark/report.py` — `BenchmarkReport` を Markdown / HTML に出力する `ReportGenerator` クラス | 高 |
| 37.2 | `benchmark/report.py` — `trend_table()`: 過去 N 回分の JSON を読み込んで改善率トレンド表を生成 | 中 |
| 37.3 | `serve/api.py` — P1〜P10 全エンドポイント (`/v1/debate/*` 〜 `/v1/plan/*`) の `TestClient` 疎通テスト | 高 |
| 37.4 | `.github/workflows/bench.yml` — `report.py --html` 出力を artifact に追加 (HTML レポートのアップロード) | 中 |
| 37.T | `tests/test_sprint37.py` — 40 tests PASS (目標: 累計 1901) | 高 |
| 37.V | PyPI v0.40.0 | — |

**作業前チェックリスト**:
- [ ] `feature/sprint37-bench-report-e2e` ブランチを切る
- [ ] `benchmark/results/` に過去 JSON が 1 件以上あることを確認 (なければ `growing_ai_bench.py --output` でダミー生成)
- [ ] `serve/api.py` の `TestClient` が `lifespan` を正しく mock できるか確認 (`pytest-asyncio` + `anyio` 設定)

**依存関係**:
- `benchmark/growing_ai_bench.py` (Sprint 35) — `BenchmarkReport.load()` を `report.py` が呼ぶ
- `.github/workflows/bench.yml` (Sprint 36) — HTML artifact upload step を追記

---

## 将来スプリント候補

- **Sprint 38**: GPU 実機 LoRA SFT 検証 — CUDA 環境で `LoraTrainer._real_train()` 動作確認 + `lora_trainer.py` に `CosineScheduler` 統合
- **Sprint 39**: Prometheus / Grafana メトリクス統合 — `serve/monitor.py` に `/metrics` エンドポイント追加
- **Sprint 40**: OpenAI 互換 Streaming 強化 — `/v1/chat/completions` の SSE テスト + `stream=true` オプション拡充
