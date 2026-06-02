# OpenMythos — Sprint Plans Archive (Sprint 10〜19)

> アーカイブ日: 2026-06-01 | 全タスク cc:完了 済み

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

> ブランチ: `feature/sprint12-react-cache-memory` → master merge 済み (PR #10)

| task-id | 説明 | 状態 |
|---------|------|------|
| 12.1.1 | `open_mythos/react.py` — ReActAgent (Think→Act→Observe ループ + format_agent_trace) | cc:完了 [292fd88] |
| 12.1.2 | `serve/api.py` — `/v1/agent/run` エンドポイント追加 | cc:完了 [292fd88] |
| 12.2.1 | `open_mythos/prefix_cache.py` — PromptPrefixCache (LRU prefill キャッシュ) | cc:完了 [292fd88] |
| 12.3.1 | `open_mythos/conversation.py` — ConversationMemory + SessionStore | cc:完了 [292fd88] |
| 12.3.2 | `serve/api.py` — `/v1/sessions/*` エンドポイント追加 | cc:完了 [292fd88] |
| 12.4.1 | PyPI v0.16.0 — pyproject.toml + CHANGELOG | cc:完了 [292fd88] |
| 12.5.1 | test_sprint12.py 65 tests — 729 PASS | cc:完了 [292fd88] |

---

## Sprint 13: Mixture-of-Depths (MoD) & SwarmOrchestrator (完了)

> ブランチ: `harness-work/13.1.2` → master merge 済み (PR #11)

| task-id | 説明 | 状態 |
|---------|------|------|
| 13.1.1 | `open_mythos/mod.py` — MoDConfig / TokenRouter / MixtureOfDepthsBlock / MoDTransformer / MoDAnalytics | cc:完了 [040261b] |
| 13.1.2 | `open_mythos/mod.py` — routing_entropy / MoDAnalytics entropy tracking / MoDTransformer.compute_loss | cc:完了 [d8d9f1e] |
| 13.2.1 | `open_mythos/swarm.py` — SwarmOrchestrator (map / broadcast / pipeline / vote) + 44 tests | cc:完了 [6d4c487] |

> テスト: test_sprint13.py 63 tests + test_sprint13_swarm.py 44 tests = **107 tests PASS**

---

## Sprint 14: GPU pretrain & Benchmark & GCP deploy & v0.17.0 (完了)

> ブランチ: `harness-work/13.1.2` → master merge 済み (PR #11)

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

> ブランチ: `harness-work/13.1.2` → master merge 済み (PR #11)

| task-id | 説明 | 状態 |
|---------|------|------|
| 15.1.1 | `open_mythos/llmo.py` — `_tokenize_ja()` / `_is_japanese()` / `score_with_keywords()` / `ab_test()` | cc:完了 [05d8526] |
| 15.1.2 | `open_mythos/conversation.py` — `ConversationMemory.drift_score()` コンテキストドリフト検出 | cc:完了 [05d8526] |
| 15.2.1 | `benchmark/llmo_bench.py` — ルールベース vs Claude API LLMO スコア比較 | cc:完了 [05d8526] |
| 15.3.1 | `tests/test_sprint15.py` — 32 tests (JaTokenizer / A/B / drift / bench) | cc:完了 [05d8526] |

---

## Sprint 16: SEOパイプライン & QS予測 & 広告バリアント & インジェクション耐性 & v0.19.0 (完了)

> ブランチ: `harness-work/13.1.2` → master merge 済み (PR #11)

| task-id | 説明 | 状態 |
|---------|------|------|
| 16.1.1 | `open_mythos/seo_pipeline.py` — SEOPipeline (4ステージ SwarmOrchestrator pipeline) | cc:完了 [0c98c3f] |
| 16.2.1 | `open_mythos/security.py` — InputGuard / OutputGuard / SecurityCheckResult | cc:完了 [0c98c3f] |
| 16.3.1 | `open_mythos/tools_marketing.py` — `quality_score()` / `generate_ad_variants()` | cc:完了 [0c98c3f] |
| 16.4.1 | `tests/test_sprint16.py` — 70 tests (SEOPipeline / QS / AdVariants / Security) | cc:完了 [0c98c3f] |
| 16.5.1 | PyPI v0.19.0 — requirements.txt (janome/fugashi/anthropic) + CHANGELOG | cc:完了 [c0fbfe8] |

---

## Sprint 17: APIキー認証 & レート制限 & Docker本番化 & v0.20.0 (完了)

> ブランチ: `feature/sprint17-auth-docker` → master merge 済み (PR #14)

| task-id | 説明 | 状態 |
|---------|------|------|
| 17.1 | `docs/mythos_vs_openmythos.md` — アーキテクチャ差分・ベンチマーク・移行ガイド | cc:完了 [87b669c] |
| 17.2 | `serve/auth.py` — Bearer Token 認証 (`verify_api_key`) + FastAPI global dependency 適用 | cc:完了 [87b669c] |
| 17.3 | `serve/Dockerfile` — Gunicorn + UvicornWorker 本番構成 / 非 root ユーザー | cc:完了 [87b669c] |
| 17.4 | `serve/auth.py` — `_SlidingWindow` + `RateLimitMiddleware` (60 rpm / `/health` スキップ) | cc:完了 [87b669c] |
| 17.5 | `serve/api.py` — 全エンドポイントに tags / summary / description 追加 (11カテゴリ) | cc:完了 [87b669c] |
| 17.6 | PyPI v0.20.0 — pyproject.toml + CHANGELOG | cc:完了 [87b669c] |
| 17.T | `tests/test_sprint17.py` — 40 tests (auth / rate-limit / Docker / OpenAPI / doc) | cc:完了 [87b669c] |

---

## Sprint 18: ファインチューニング実証 & マーケティング分析強化 & v0.21.0 (完了)

> ブランチ: `feature/sprint18-finetuning` → master merge 済み (PR #15)

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

## Sprint 19: LLMO 強化 — クエリ関連性 / 意図分類 / LLMOOptimizer & v0.22.0 (完了)

> ブランチ: `feature/sprint19-llmo-enhance` → master merge 済み (PR #16)

| task-id | 説明 | 状態 |
|---------|------|------|
| 19.1.1 | `open_mythos/llmo.py` — `query_relevance` / `intent_type` フィールドを `LLMOScore` に追加 | cc:完了 [fca7e84] |
| 19.1.2 | `open_mythos/llmo.py` — `score_with_query()` — TF-IDF コサイン類似度 + 意図分類 | cc:完了 [fca7e84] |
| 19.2.1 | `open_mythos/llmo.py` — `Improvement` dataclass + `suggest_improvements()` 優先度付き提案エンジン | cc:完了 [fca7e84] |
| 19.3.1 | `open_mythos/llmo.py` — `LLMOOptimizer` + `OptimizedResult` — ルールベース反復最適化 | cc:完了 [fca7e84] |
| 19.4.1 | `serve/api.py` — `/v1/llmo/suggest` / `/v1/llmo/optimize` / `/v1/llmo/score` 3エンドポイント追加 | cc:完了 [fca7e84] |
| 19.T | `tests/test_sprint19.py` — 42 tests (ScoreWithQuery / SuggestImprovements / LLMOOptimizer / API / Integration) | cc:完了 [fca7e84] |
| 19.V | PyPI v0.22.0 — pyproject.toml + CHANGELOG Sprint 19 追加 | cc:完了 [fca7e84] |
