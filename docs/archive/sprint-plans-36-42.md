# Sprint 34〜43 アーカイブ — API強化・OpenAI互換・埋め込み・Hermes Orchestrator
> アーカイブ日: 2026-06-04

---

## Sprint 37 詳細

### Sprint 37: ベンチマーク結果可視化 + E2E 疎通テスト — v0.40.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 37.1 | `benchmark/report.py` — `ReportGenerator.to_markdown()` / `to_html()` / `save_*()` | cc:完了 |
| 37.2 | `benchmark/report.py` — `load_reports()` / `trend_table(json_paths, n)` トレンド表生成 | cc:完了 |
| 37.3 | `tests/test_sprint37.py` — P2〜P10 + guard + grow TestClient 疎通テスト (20 tests) | cc:完了 |
| 37.4 | `.github/workflows/bench.yml` — HTML レポート生成ステップ + artifact upload (retention 90日) | cc:完了 |
| 37.T | `tests/test_sprint37.py` — 61 tests PASS (累計 1922) | cc:完了 |
| 37.V | PyPI v0.40.0 | cc:完了 |

## Sprint 36 詳細

### Sprint 36: API ドキュメント整備 + CI ベンチマーク自動化 — v0.39.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 36.1a | `serve/api.py` — `version="0.38.0"` 確認・旧バージョン除去 | cc:完了 |
| 36.1b | `serve/api.py` — `openapi_tags` 全 23 タグ整備 (P1〜P10 + grow + guard) | cc:完了 |
| 36.2a | `.github/workflows/bench.yml` — ファイル存在・YAML 構文 | cc:完了 |
| 36.2b | `.github/workflows/bench.yml` — 週次 cron (月曜 09:00 JST) + workflow_dispatch | cc:完了 |
| 36.2c | `.github/workflows/bench.yml` — jobs: checkout / setup-python / bench 実行 / artifact 保存 | cc:完了 |
| 36.T | `tests/test_sprint36.py` — 39 tests PASS | cc:完了 |
| 36.V | PyPI v0.39.0 | cc:完了 |

## Sprint 35 詳細

### Sprint 35: ベンチマーク強化 — v0.38.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 35.1 | `benchmark/growing_ai_bench.py` — `PatternBenchResult` / P1〜P10 bench 関数 | cc:完了 |
| 35.2 | `benchmark/growing_ai_bench.py` — `GrowingAIBenchmark` / `BenchmarkReport` | cc:完了 |
| 35.3 | CLI — `--patterns` / `--verbose` / `--output` / 実測: 10/10 成功・平均 +10.5% | cc:完了 |
| 35.T | `tests/test_sprint35.py` — 45 tests PASS | cc:完了 |
| 35.V | PyPI v0.38.0 | cc:完了 |

## Sprint 34 詳細

### Sprint 34: MistakeGuardMiddleware — v0.37.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 34.1 | `error_memory.py` — `GuardMiddlewareConfig` / `MistakeGuardMiddleware` | cc:完了 |
| 34.2 | `serve/api.py` — `_MistakeGuardHTTPMiddleware` + `/v1/guard/stats` / `/v1/guard/refresh` | cc:完了 |
| 34.T | `tests/test_sprint34.py` — 40 tests PASS | cc:完了 |
| 34.V | PyPI v0.37.0 | cc:完了 |

## Sprint 38 詳細

### Sprint 38: GPU LoRA SFT — CosineScheduler 統合 + 実機検証基盤 — v0.41.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 38.1 | `lora_trainer.py` — `LoraTrainerConfig` に `warmup_steps` / `min_lr_ratio` / `use_scheduler` 追加 | cc:完了 |
| 38.2 | `lora_trainer.py` — `_real_train()` に `CosineAnnealingLR` + 線形 warmup 統合 | cc:完了 |
| 38.3 | `lora_trainer.py` — `cosine_t_max()` / `get_current_lr()` ヘルパー追加 | cc:完了 |
| 38.T | `tests/test_sprint38.py` — 41 tests PASS / 3 GPU-skip (累計 1963) | cc:完了 |
| 38.V | PyPI v0.41.0 | cc:完了 |

## Sprint 39 詳細

### Sprint 39: ショーケースダッシュボード + Prometheus メトリクス + GitHub Pages — v0.42.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 39.1 | `serve/dashboard.py` — `build_showcase_dashboard()` / `save_dashboard()` / FastAPI router | cc:完了 |
| 39.2 | `serve/api.py` — `/metrics` Prometheus エンドポイント + `prometheus_client` 統合 | cc:完了 |
| 39.3 | `serve/monitor.py` — `build_monitor_dashboard_html()` HTML 化 + `/monitor/dashboard` | cc:完了 |
| 39.4 | `.github/workflows/pages.yml` — GitHub Pages 自動デプロイ | cc:完了 |
| 39.T | `tests/test_sprint39.py` — 49 tests PASS (累計 2012) | cc:完了 |
| 39.V | PyPI v0.42.0 | cc:完了 |

## Sprint 40 詳細

### Sprint 40: OpenAI 互換 Streaming 強化 + /v1/completions — v0.43.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 40.1 | `serve/api.py` — サンプリングヘルパー 5 関数追加 | cc:完了 |
| 40.2 | `serve/api.py` — `ChatRequest` 拡張: stop / n / logprobs / penalties | cc:完了 |
| 40.3 | `serve/api.py` — SSE 強化: stop対応 / finish_reason:"length" / usage chunk | cc:完了 |
| 40.4 | `serve/api.py` — `/v1/completions` 新規追加 | cc:完了 |
| 40.T | `tests/test_sprint40.py` — 65 tests PASS (累計 2077) | cc:完了 |
| 40.V | PyPI v0.43.0 | cc:完了 |

## Sprint 41 詳細

### Sprint 41: /v1/chat/completions Function Calling 統合 — v0.44.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 41.1 | `serve/api.py` — `ChatMessage` に tool ロール / tool_call_id / tool_calls 追加 | cc:完了 |
| 41.2 | `serve/api.py` — `ChatRequest` に tools / tool_choice、`ChatChoice` に tool_calls 追加 | cc:完了 |
| 41.3 | `serve/api.py` — ヘルパー 3 関数追加 (inject / parse / build) | cc:完了 |
| 41.4 | `serve/api.py` — tool_calls フロー統合・SSE delta 対応 | cc:完了 |
| 41.T | `tests/test_sprint41.py` — 43 tests PASS (累計 2120) | cc:完了 |
| 41.V | PyPI v0.44.0 | cc:完了 |

## Sprint 42 詳細

### Sprint 42: /v1/embeddings + セマンティック検索 — v0.45.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 42.1 | `open_mythos/main.py` — `OpenMythos.encode()` 追加 | cc:完了 |
| 42.2 | `serve/api.py` — `/v1/embeddings` (OpenAI 互換・float/base64・dimensions・バッチ) | cc:完了 |
| 42.3 | `serve/api.py` — `/v1/semantic-search` (コサイン類似度・top_k) | cc:完了 |
| 42.T | `tests/test_sprint42.py` — 46 tests PASS (累計 2166) | cc:完了 |
| 42.V | PyPI v0.45.0 | cc:完了 |

## Sprint 43 詳細

### Sprint 43: HermesOrchestrator — Layer 2 Ultracode Mode — v0.46.0
| task-id | 説明 | 状態 |
|---------|------|------|
| 43.1 | `open_mythos/hermes_orchestrator.py` — SubTask / SubAgentSpec / HermesAgentResult / VerificationResult / HermesReport | cc:完了 |
| 43.2 | `open_mythos/hermes_orchestrator.py` — TaskDecomposer / AgentSpawner / ParallelExecutor / ResultVerifier / ReportBuilder | cc:完了 |
| 43.3 | `open_mythos/hermes_orchestrator.py` — HermesOrchestrator (plan / spawn / verify / report / run / run_async) | cc:完了 |
| 43.4 | `serve/api.py` — `/v1/hermes/run` (非同期並列実行) + `/v1/hermes/plan` (サブタスク計画) | cc:完了 |
| 43.5 | `open_mythos/__init__.py` — HermesOrchestrator 関連クラス全エクスポート追加 | cc:完了 |
| 43.T | `tests/test_sprint43.py` — 94 tests PASS (累計 2260) | cc:完了 |
| 43.V | PyPI v0.46.0 | cc:完了 |
