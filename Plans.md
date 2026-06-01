# OpenMythos — Sprint Plans
> 最終更新: 2026-06-01 | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 1〜9 のアーカイブ: `docs/archive/sprint-plans-1-9.md`

---

## 完了済み Sprint サマリー

| Sprint | 内容 | テスト | バージョン | commit |
|--------|------|--------|-----------|--------|
| 1 | HyperloopMythos / Inference Engine | 227 PASS | — | 6a64810 |
| 2 | Inference 高度化 (beam/quant/KV cache) | — | — | 9946ef1 |
| 3 | ドキュメント & エコシステム | — | v0.6.0 | — |
| 4 | Training 基盤 / MoDa / variants | 257 PASS | — | — |
| 5 | Training 品質 / LoRA / HF Hub / CLI | 284 PASS | v0.7.0 | — |
| 6 | 推論最適化 / TrainLogger / Agents / Benchmark | 380 PASS | — | — |
| 7 | Serving テスト / データパイプライン | 420+ PASS | v0.12.0 | — |
| 8 | Fine-tuning / /v1/chat / SLA ultra | 468 PASS | — | — |
| 9 | Marketing eval / A/B 検定 / batch API | 508 PASS | v0.13.0 | — |

---

## Sprint 10: LLMO生成 & Extended Thinking & Structured Output & v0.14.0 (完了)

> ブランチ: `feature/sprint10-llmo-thinking` → master merge 済み (PR #8)

| task-id | 説明 | 状態 |
|---------|------|------|
| 10.1.1 | `open_mythos/llmo.py` — LLMOScorer (entity_density / answer_directness / citability) | cc:完了 [ae264dd] |
| 10.1.2 | `scripts/generate_seo.py` — SEO/LLMO最適化コンテンツ生成 (3スタイル) | cc:完了 [ae264dd] |
| 10.1.3 | `serve/api.py` — `/v1/seo/score` & `/v1/seo/generate` | cc:完了 [ae264dd] |
| 10.2.1 | `open_mythos/thinking.py` — Extended Thinking (per-loop 内部状態エクスポート) | cc:完了 [ae264dd] |
| 10.2.2 | `serve/api.py` — `/v1/thinking` + ChatRequest `thinking` フラグ | cc:完了 [ae264dd] |
| 10.3.1 | `open_mythos/structured.py` — JSON mode / Structured Output | cc:完了 [ae264dd] |
| 10.3.2 | `scripts/train_dpo.py` — DPO fine-tuning | cc:完了 [ae264dd] |
| 10.4.1 | PyPI v0.14.0 | cc:完了 [ae264dd] |
| 10.5.1 | test_sprint10.py 52 tests — 560 PASS | cc:完了 [ae264dd] |

---

## Sprint 11: Tool Use / Long Context / RAG & v0.15.0 (完了)

> ブランチ: `feature/sprint11-tools-longctx-rag` → master merge 済み (PR #9)

| task-id | 説明 | 状態 |
|---------|------|------|
| 11.1.1 | `open_mythos/tools.py` — ToolRegistry / @tool / ToolCall / ToolResult | cc:完了 [d557cd5] |
| 11.1.2 | `open_mythos/tools_marketing.py` — search_competitor / calculate_roi / fetch_trend / score_content | cc:完了 [d557cd5] |
| 11.1.3 | `serve/api.py` — `/v1/tools`, `/v1/tools/call`, `/v1/tools/batch` | cc:完了 [d557cd5] |
| 11.2.1 | `open_mythos/rope_extension.py` — YaRN Dynamic NTK-aware RoPE (32K対応) | cc:完了 [d557cd5] |
| 11.2.2 | Long Context 推論テスト — extend_model_context() | cc:完了 [d557cd5] |
| 11.3.1 | `open_mythos/rag.py` — VectorStore / RAGPipeline (numpy + FAISS opt) | cc:完了 [d557cd5] |
| 11.3.2 | `serve/api.py` — `/v1/rag/index`, `/v1/rag` | cc:完了 [d557cd5] |
| 11.4.1 | PyPI v0.15.0 | cc:完了 [d557cd5] |
| 11.5.1 | test_sprint11.py 104 tests — 664 PASS | cc:完了 [d557cd5] |

---

## Sprint 12: ReAct エージェントループ & プロンプトキャッシュ & 会話メモリ & v0.16.0 (完了)

> ブランチ: `feature/sprint12-react-cache-memory`
> 戦略: A) ReAct Agent Loop B) Prompt Prefix Cache C) Conversation Memory / Session API

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 12.1.1 | `open_mythos/react.py` — ReActAgent (Think→Act→Observe ループ + format_agent_trace) | Worker | cc:完了 [292fd88] | (a) AgentStep/AgentResult (b) ループ実装 (c) テスト PASS |
| 12.1.2 | `serve/api.py` に `/v1/agent/run` エンドポイント追加 | Worker | cc:完了 [292fd88] | (a) AgentRunRequest/Response (b) max_iterations 制御 (c) テスト PASS |
| 12.2.1 | `open_mythos/prefix_cache.py` — PromptPrefixCache (LRU prefill キャッシュ) | Worker | cc:完了 [292fd88] | (a) cache_prefix() (b) generate_with_cache() (c) hit_rate 統計 (d) テスト PASS |
| 12.3.1 | `open_mythos/conversation.py` — ConversationMemory + SessionStore | Worker | cc:完了 [292fd88] | (a) add_turn() / to_context_string() (b) 自動圧縮 (c) セッション管理 (d) テスト PASS |
| 12.3.2 | `serve/api.py` に `/v1/sessions/*` エンドポイント追加 | Worker | cc:完了 [292fd88] | (a) POST/GET/DELETE /v1/sessions (b) POST turns (c) GET context (d) テスト PASS |
| 12.4.1 | PyPI v0.16.0 — pyproject.toml 0.15.0→0.16.0 + CHANGELOG Sprint 12 追加 | Worker | cc:完了 [292fd88] | (a) version bump (b) CHANGELOG 追加 |
| 12.5.1 | Sprint 12 テスト追加 + commit + push | Worker | cc:完了 [292fd88] | (a) test_sprint12.py 65 tests (b) 729 PASS (c) git push |

---

## Sprint 13: Mixture-of-Depths (MoD) & SwarmOrchestrator (完了)

> ブランチ: `harness-work/13.1.2`
> 戦略: A) MoD Transformer (routing_entropy / entropy tracking) B) SwarmOrchestrator 並列マルチエージェント

| task-id | 説明 | 状態 |
|---------|------|------|
| 13.1.1 | `open_mythos/mod.py` — MoDConfig / TokenRouter / MixtureOfDepthsBlock / MoDTransformer / MoDAnalytics | cc:完了 [040261b] |
| 13.1.2 | `open_mythos/mod.py` — routing_entropy / MoDAnalytics entropy tracking / MoDTransformer.compute_loss | cc:完了 [d8d9f1e] |
| 13.2.1 | `open_mythos/swarm.py` — SwarmOrchestrator (map / broadcast / pipeline / vote) + 44 tests | cc:完了 [6d4c487] |

> テスト: test_sprint13.py 63 tests + test_sprint13_swarm.py 44 tests = **107 tests PASS**

---

## Sprint 14: GPU pretrain & Benchmark & GCP deploy & v0.17.0 (完了)

> ブランチ: `harness-work/13.1.2`

| task-id | 説明 | 状態 |
|---------|------|------|
| 14.1.1 | `scripts/pretrain.py` — StreamingTokenDataset / warmup_stable_decay / gradient_checkpointing | cc:完了 [8cc9c5f] |
| 14.1.2 | `scripts/pretrain_gcp.sh` — GCP T4 tmux/nohup 実行スクリプト | cc:完了 [8cc9c5f] |
| 14.2.1 | `benchmark/run_eval.py` — PPL / HellaSwag / ARC / WinoGrande 一括評価 + README 自動更新 | cc:完了 [8cc9c5f] |
| 14.3.1 | `serve/deploy_cloudrun.sh` + `cloudrun.env.example` — Cloud Run デプロイ自動化 | cc:完了 [8cc9c5f] |
| 14.4.1 | `examples/demo_seo_llmo.ipynb` — Colab ゼロから動くデモノートブック | cc:完了 [8cc9c5f] |
| 14.5.1 | バグ修正 2件 (mod / prefix_cache / rag / react / rope_extension / thinking / api / monitor) | cc:完了 [8cc9c5f] |
| 14.5.2 | テスト品質強化 9件 (test_sprint7_serve / 10 / 11 / 12 / 13) | cc:完了 [8cc9c5f] |
| 14.6.1 | `tests/test_sprint8_pretrain.py` — 20 tests | cc:完了 [8cc9c5f] |
| 14.7.1 | PyPI v0.17.0 — README 外販版 + CHANGELOG | cc:完了 [8cc9c5f] |

---

## Sprint 15: 日本語形態素解析 & A/Bテスト & ドリフト検出 & v0.18.0 (完了)

> ブランチ: `harness-work/13.1.2`

| task-id | 説明 | 状態 |
|---------|------|------|
| 15.1.1 | `open_mythos/llmo.py` — `_tokenize_ja()` / `_is_japanese()` / `score_with_keywords()` / `ab_test()` | cc:完了 [05d8526] |
| 15.1.2 | `open_mythos/conversation.py` — `ConversationMemory.drift_score()` コンテキストドリフト検出 | cc:完了 [05d8526] |
| 15.2.1 | `benchmark/llmo_bench.py` — ルールベース vs Claude API LLMO スコア比較 | cc:完了 [05d8526] |
| 15.3.1 | `tests/test_sprint15.py` — 32 tests (JaTokenizer / A/B / drift / bench) | cc:完了 [05d8526] |

---

## Sprint 16: SEOパイプライン & QS予測 & 広告バリアント & インジェクション耐性 & v0.19.0 (完了)

> ブランチ: `harness-work/13.1.2`

| task-id | 説明 | 状態 |
|---------|------|------|
| 16.1.1 | `open_mythos/seo_pipeline.py` — SEOPipeline (4ステージ SwarmOrchestrator pipeline) | cc:完了 [0c98c3f] |
| 16.2.1 | `open_mythos/security.py` — InputGuard / OutputGuard / SecurityCheckResult | cc:完了 [0c98c3f] |
| 16.3.1 | `open_mythos/tools_marketing.py` — `quality_score()` / `generate_ad_variants()` | cc:完了 [0c98c3f] |
| 16.4.1 | `tests/test_sprint16.py` — 70 tests (SEOPipeline / QS / AdVariants / Security) | cc:完了 [0c98c3f] |
| 16.5.1 | PyPI v0.19.0 — requirements.txt (janome/fugashi/anthropic) + CHANGELOG | cc:完了 [c0fbfe8] |

---

## Sprint 17: APIキー認証 & レート制限 & Docker本番化 & v0.20.0 (完了)

> ブランチ: `feature/sprint17-auth-docker`

| task-id | 説明 | 状態 |
|---------|------|------|
| 17.1 | `docs/mythos_vs_openmythos.md` — アーキテクチャ差分・ベンチマーク・移行ガイド | cc:完了 [87b669c] |
| 17.2 | `serve/auth.py` — Bearer Token 認証 (`verify_api_key`) + FastAPI global dependency 適用 | cc:完了 [87b669c] |
| 17.3 | `serve/Dockerfile` — Gunicorn + UvicornWorker 本番構成 / 非 root ユーザー | cc:完了 [87b669c] |
| 17.3 | `docker-compose.yml` — RATE_LIMIT_RPM / API_KEY / WORKERS 設定追加 | cc:完了 [87b669c] |
| 17.4 | `serve/auth.py` — `_SlidingWindow` + `RateLimitMiddleware` (60 rpm / `/health` スキップ) | cc:完了 [87b669c] |
| 17.5 | `serve/api.py` — 全エンドポイントに tags / summary / description 追加 (11カテゴリ) | cc:完了 [87b669c] |
| 17.6 | PyPI v0.20.0 — pyproject.toml + CHANGELOG | cc:完了 [87b669c] |
| 17.T | `tests/test_sprint17.py` — 40 tests (auth / rate-limit / Docker / OpenAPI / doc) | cc:完了 [87b669c] |

---

## Sprint 18: ファインチューニング実証 & マーケティング分析強化 & v0.21.0

> ブランチ: `feature/sprint18-finetuning`

| task-id | 説明 | 状態 |
|---------|------|------|
| 18.1 | `scripts/csv_to_jsonl.py` — CSV → JSONL 変換スクリプト (SFT データ前処理) | cc:完了 [89506bc] |
| 18.2 | `scripts/finetune.py` — LoRA SFT 実行スクリプト (Trainer 統合) | cc:完了 [89506bc] |
| 18.3 | `benchmark/compare_opus.py` — OpenMythos LLMOScorer vs ルールベースライン比較 | cc:完了 [89506bc] |
| 18.4 | `serve/api.py` — `/v1/ab/infer` + `/v1/ab/stats` A/B テストエンドポイント | cc:完了 [89506bc] |
| 18.5 | `open_mythos/tools_marketing.py` — `roas_simulate()` Monte Carlo ROAS シミュレーター | cc:完了 [89506bc] |
| 18.6 | `open_mythos/tools_marketing.py` — `persona_ad_match()` TF-IDF ペルソナ×広告マッチング | cc:完了 [89506bc] |
| 18.T | `tests/test_sprint18.py` — 39 tests (roas_simulate / persona_ad_match / compare_opus / A/B) | cc:完了 [89506bc] |
| 18.V | PyPI v0.21.0 — pyproject.toml + CHANGELOG | cc:完了 [89506bc] |

---

## Sprint 19: LLMO 強化 — クエリ関連性 / 意図分類 / LLMOOptimizer & v0.22.0

> ブランチ: `feature/sprint19-llmo-enhance`

| task-id | 説明 | 状態 |
|---------|------|------|
| 19.1.1 | `open_mythos/llmo.py` — `query_relevance` / `intent_type` フィールドを `LLMOScore` に追加 | cc:完了 |
| 19.1.2 | `open_mythos/llmo.py` — `score_with_query()` — TF-IDF コサイン類似度 + 意図分類 | cc:完了 |
| 19.2.1 | `open_mythos/llmo.py` — `Improvement` dataclass + `suggest_improvements()` 優先度付き提案エンジン | cc:完了 |
| 19.3.1 | `open_mythos/llmo.py` — `LLMOOptimizer` + `OptimizedResult` — ルールベース反復最適化 | cc:完了 |
| 19.4.1 | `serve/api.py` — `/v1/llmo/suggest` / `/v1/llmo/optimize` / `/v1/llmo/score` 3エンドポイント追加 | cc:完了 |
| 19.T | `tests/test_sprint19.py` — 42 tests (ScoreWithQuery / SuggestImprovements / LLMOOptimizer / API / Integration) | cc:完了 |
| 19.V | PyPI v0.22.0 — pyproject.toml + CHANGELOG Sprint 19 追加 | cc:完了 |

---

## Sprint 20〜25: 「育つAI」— Self-Improving Agent Framework & v0.23.0〜v0.28.0

> 設計思想: AI が自律的に経験を蓄積し、KPI に近づき、外部要因に適応し、ミスから学習する。
> 「育つパターン」を6つに体系化し、1 Sprint = 1パターン として実装する。

---

### パターン定義

| # | パターン名 | 一言説明 | コアメカニズム |
| --- | --------- | ------- | ------------ |
| P1 | **討議型集合知** | エージェント間で討論し最善策を収束 | Debate → Critique → Consensus |
| P2 | **KPI駆動自己改善** | KPIギャップを検出し行動を自動生成 | Gap Analysis → Action Plan → Execute → Measure |
| P3 | **ボトルネック発見・解消** | 分析で詰まりを検出し自動改善 | Profiling → Root Cause → Patch → Verify |
| P4 | **外部要因適応** | 季節・トレンド等の外部変化を内部行動に変換 | Signal Detect → Impact Estimate → Counter-Action |
| P5 | **ミスから学習** | 失敗パターンをDBに蓄積し再発防止 | Error Capture → Classification → Rule Extraction → Guard |
| P6 | **継続的自己蒸留** | 自分の良い出力を教師データ化し継続ファインチューン | Good Output Filter → JSONL → LoRA SFT → Eval Loop |

---

## Sprint 20: 討議型集合知 — DebateOrchestrator & v0.23.0

> ブランチ: `feature/sprint20-debate`
> パターン P1: 複数エージェントが Propose → Critique → Refine → Consensus の4フェーズで討議し
> 単独エージェントより高品質な意思決定を行う。既存 `SwarmOrchestrator` を基盤に拡張。

| task-id | 説明 | 状態 |
|---------|------|------|
| 20.1 | `open_mythos/debate.py` — `DebateConfig` / `DebateRound` / `DebateResult` dataclass | cc:完了 |
| 20.2 | `open_mythos/debate.py` — `DebateOrchestrator.run()` — Propose→Critique→Refine→Consensus 4フェーズループ | cc:完了 |
| 20.3 | `open_mythos/debate.py` — `ConsensusEngine` — Jaccard類似度合意収束 (agreement_score / confidence / 日本語bi-gram) | cc:完了 |
| 20.4 | `serve/api.py` — `/v1/debate/run` エンドポイント (n_rounds / n_agents / topic / consensus_threshold) | cc:完了 |
| 20.T | `tests/test_sprint20.py` — 40 tests (DebateRound / Consensus / API / multi-agent agreement) | cc:完了 |
| 20.V | PyPI v0.23.0 + CHANGELOG | cc:完了 |

**DoD**: 3エージェント討議で単独エージェント比 agreement_score +15% 以上

---

## Sprint 21: KPI駆動自己改善 — KPIAgent & v0.24.0

> ブランチ: `feature/sprint21-kpi-agent`
> パターン P2: KPI定義 → Gap検出 → ActionPlan生成 → 実行 → 再計測 のサイクルを自律実行。
> LLMO スコア / ROAS / conversion_rate など任意の KPI に適用可能。

| task-id | 説明 | 状態 |
|---------|------|------|
| 21.1 | `open_mythos/kpi_agent.py` — `KPIDefinition` / `KPISnapshot` / `GapReport` / `Action` / `ActionPlan` / `KPIImproveResult` | cc:完了 |
| 21.2 | `open_mythos/kpi_agent.py` — `KPIAgent.measure()` — KPI値計測・スナップショット保存 | cc:完了 |
| 21.3 | `open_mythos/kpi_agent.py` — `KPIAgent.analyze()` / `plan()` — Gap分析 → ActionPlan生成 (estimated_impact順) | cc:完了 |
| 21.4 | `open_mythos/kpi_agent.py` — `KPIAgent.execute()` — アクション変換関数を順次適用 | cc:完了 |
| 21.5 | `open_mythos/kpi_agent.py` — `KPIAgent.improve_loop()` — measure→analyze→plan→execute を n_cycles 自律実行 | cc:完了 |
| 21.6 | `serve/api.py` — `/v1/kpi/measure` / `/v1/kpi/improve` エンドポイント | cc:完了 |
| 21.T | `tests/test_sprint21.py` — 40 tests (KPIDefinition / GapReport / ActionPlan / improve_loop / API) | cc:完了 |
| 21.V | PyPI v0.24.0 + CHANGELOG | cc:完了 |

**DoD**: LLMO KPI を target に対して 2サイクル以内に +10% 改善できること

---

## Sprint 22: ボトルネック発見・解消 — ProfilerAgent & v0.25.0

> ブランチ: `feature/sprint22-profiler`
> パターン P3: パイプライン各ステージの実行時間・スコア・エラー率を計測し
> ボトルネックを自動特定、改善パッチを生成して適用・検証するサイクル。

| task-id | 説明 | 状態 |
|---------|------|------|
| 22.1 | `open_mythos/profiler.py` — `StageMetrics` / `ProfileResult` / `BottleneckReport` / `AutoFixResult` dataclass | cc:完了 |
| 22.2 | `open_mythos/profiler.py` — `PipelineProfiler.run()` — 全ステージを順次実行・計測 | cc:完了 |
| 22.3 | `open_mythos/profiler.py` — `BottleneckDetector.detect()` — IQR法で latency / score / error 外れ値検出 | cc:完了 |
| 22.4 | `open_mythos/profiler.py` — `ProfilerAgent.auto_fix()` — latency/score/error 別パッチ適用・再計測 | cc:完了 |
| 22.5 | `serve/api.py` — `/v1/profile/run` / `/v1/profile/fix` / `/v1/profile/report` エンドポイント | cc:完了 |
| 22.T | `tests/test_sprint22.py` — 35 tests (StageMetrics / BottleneckDetector / ProfilerAgent / API) | cc:完了 |
| 22.V | PyPI v0.25.0 + CHANGELOG | cc:完了 |

**DoD**: 意図的に遅くしたステージを正しく検出し latency -20% 改善できること

---

## Sprint 23: 外部要因適応 — ExternalSignalAgent & v0.26.0

> ブランチ: `feature/sprint23-external-signal`
> パターン P4: 季節変化・トレンド急上昇・競合動向などの外部シグナルを検出し
> 内部コンテンツ戦略・広告パラメータを自動調整する。

| task-id | 説明 | 状態 |
|---------|------|------|
| 23.1 | `open_mythos/external_signal.py` — `ExternalSignal` / `SignalType` / `ImpactEstimate` dataclass | 未着手 |
| 23.2 | `open_mythos/external_signal.py` — `SignalDetector.detect()` — 季節スコア / トレンドスパイク / 競合変化を数値化 | 未着手 |
| 23.3 | `open_mythos/external_signal.py` — `ImpactEstimator.estimate()` — シグナル強度 → KPI影響量マッピング | 未着手 |
| 23.4 | `open_mythos/external_signal.py` — `ExternalSignalAgent.counter_action()` — 影響を打ち消す内部アクション生成 | 未着手 |
| 23.5 | `serve/api.py` — `/v1/signal/detect` / `/v1/signal/counter` エンドポイント | 未着手 |
| 23.T | `tests/test_sprint23.py` — 35 tests (SignalDetector / ImpactEstimator / counter_action / seasonal) | 未着手 |
| 23.V | PyPI v0.26.0 + CHANGELOG | 未着手 |

**DoD**: 季節シグナル強度 0.8 のとき counter_action を生成し LLMO score が維持されること

---

## Sprint 24: ミスから学習 — ErrorMemory & MistakeGuard & v0.27.0

> ブランチ: `feature/sprint24-error-memory`
> パターン P5: エラー・低品質出力を自動分類・蓄積し、同パターンのミスを事前にブロックする
> ガードレールと、ルール抽出による継続的な品質向上ループ。

| task-id | 説明 | 状態 |
|---------|------|------|
| 24.1 | `open_mythos/error_memory.py` — `MistakeRecord` / `MistakeCategory` dataclass | 未着手 |
| 24.2 | `open_mythos/error_memory.py` — `ErrorMemoryStore` — append / query_similar (TF-IDF) / stats | 未着手 |
| 24.3 | `open_mythos/error_memory.py` — `MistakeClassifier.classify()` — エラータイプ自動分類 (8カテゴリ) | 未着手 |
| 24.4 | `open_mythos/error_memory.py` — `RuleExtractor.extract()` — 蓄積ミスから防止ルールを自動生成 | 未着手 |
| 24.5 | `open_mythos/error_memory.py` — `MistakeGuard.check()` — 入力/出力をルールDB照合し事前ブロック | 未着手 |
| 24.6 | `serve/api.py` — `/v1/mistakes/record` / `/v1/mistakes/rules` / `/v1/mistakes/check` エンドポイント | 未着手 |
| 24.T | `tests/test_sprint24.py` — 40 tests (ErrorMemoryStore / RuleExtractor / MistakeGuard / API) | 未着手 |
| 24.V | PyPI v0.27.0 + CHANGELOG | 未着手 |

**DoD**: 同カテゴリのミスを10件蓄積後、MistakeGuard が類似入力を 80% 以上ブロックできること

---

## Sprint 25: 継続的自己蒸留 — SelfDistillLoop & v0.28.0

> ブランチ: `feature/sprint25-self-distill`
> パターン P6: 自分が生成した出力のうち高スコアのものを教師データとしてフィルタリングし
> LoRA SFT で継続的にファインチューンするセルフプレイ型成長ループ。

| task-id | 説明 | 状態 |
|---------|------|------|
| 25.1 | `open_mythos/self_distill.py` — `DistillSample` / `DistillDataset` dataclass | 未着手 |
| 25.2 | `open_mythos/self_distill.py` — `OutputFilter.filter()` — LLMO スコア閾値フィルタ + 多様性保証 | 未着手 |
| 25.3 | `open_mythos/self_distill.py` — `SelfDistillCollector` — 推論実行 → スコア → 保存 パイプライン | 未着手 |
| 25.4 | `open_mythos/self_distill.py` — `SelfDistillLoop.run()` — Collect→Filter→SFT→Eval を n_rounds 自律実行 | 未着手 |
| 25.5 | `serve/api.py` — `/v1/distill/collect` / `/v1/distill/train` / `/v1/distill/status` エンドポイント | 未着手 |
| 25.T | `tests/test_sprint25.py` — 40 tests (OutputFilter / SelfDistillCollector / SelfDistillLoop / API) | 未着手 |
| 25.V | PyPI v0.28.0 + CHANGELOG | 未着手 |

**DoD**: 3ラウンド後に LLMO スコア平均 +5% 以上改善、訓練データ品質 (mean_score > 0.7) を維持

---

## 「育つAI」パターン依存関係

```text
P1 討議型集合知 (Sprint 20)
    └─ P2 KPI駆動自己改善 (Sprint 21)  ← P1 の Consensus を KPI評価に活用
        ├─ P3 ボトルネック発見 (Sprint 22)  ← P2 の measure() を流用
        └─ P4 外部要因適応 (Sprint 23)  ← P2 の ActionPlan 生成を流用
P5 ミスから学習 (Sprint 24)  ← P3/P4 の実行ログをエラーDBに投入
    └─ P6 継続的自己蒸留 (Sprint 25)  ← P5 のフィルタ済みデータで SFT
```

---

## 進行中の作業メモ

### 現在のブランチ状態 (2026-06-01 更新)
- `master`: `3d6ffac` — Sprint 1〜12 全完了 (729 PASS / v0.16.0)
- `harness-work/13.1.2` → PR #11: Sprint 13〜16 全完了 (933 tests 収集 / v0.19.0) — master merge 待ち
- `feature/sprint17-auth-docker`: Sprint 17 全完了 (v0.20.0 / 40 new tests)

### 重要な技術的知見
- `freqs_cis` は必ず `[:T]` スライスして渡すこと (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` は `.clamp(min=1e-6)` が必要 (float32 飽和防止)
- decode_loops 2-phase: prefill=4 / decode=1 が最速 (2.54x)
- stash pop 時に Plans.md でコンフリクト発生しやすい → `git checkout stash -- Plans.md` で解決
