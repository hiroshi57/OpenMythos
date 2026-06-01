# Changelog

All notable changes to OpenMythos are documented here.

---

## [0.23.0] — 2026-06-01

### Sprint 20: 討議型集合知 — DebateOrchestrator (P1パターン)

#### DebateOrchestrator (`open_mythos/debate.py`)

- `DebateConfig` — n_agents / n_rounds / consensus_threshold / max_workers 設定
- `DebateRound` — 1ラウンド (proposals / critiques / refinements / agreement_score)
- `DebateResult` — 討議全体の結果 (consensus / agreement_score / confidence / early_stopped)
- `ConsensusEngine.score()` — Jaccard類似度ベース合意スコア + 代表テキスト選出 (日本語bi-gram対応)
- `ConsensusEngine.confidence()` — ラウンドスコア収束信頼度計算
- `DebateOrchestrator.run()` — Propose → Critique → Refine → Consensus 4フェーズ討議
- 早期終了: agreement_score が consensus_threshold を超えた時点でループ終了
- スレッドプール並列実行 (各フェーズで全エージェントを同時実行)

#### API (`serve/api.py`)

- `POST /v1/debate/run` — topic / n_agents / n_rounds / consensus_threshold 指定で討議実行
  - レスポンス: consensus / agreement_score / confidence / rounds / early_stopped / improved_over_solo

#### テスト (`tests/test_sprint20.py`)

- `tests/test_sprint20.py` — 40 tests (DebateConfig / ConsensusEngine / DebateRound / DebateResult / DebateOrchestrator / API)

---

## [0.22.0] — 2026-06-01

### Sprint 19: LLMO 強化 — クエリ関連性 / 意図分類 / 自動最適化 (LLMOOptimizer)

#### クエリ関連性スコアリング (`open_mythos/llmo.py`)

- `LLMOScore.query_relevance: float` — クエリとテキストの TF-IDF コサイン類似度 (0〜1)
- `LLMOScore.intent_type: str` — 検索意図分類 (`informational` / `navigational` / `transactional` / `commercial`)
- `LLMOScorer.score_with_query(text, query)` — 既存 3軸スコア + クエリ関連性 + 意図分類を一括計算
- `LLMOScorer._calc_query_relevance()` — TF-IDF コサイン類似度 (cosine×0.7 + hit_rate×0.3、外部依存なし)
- `LLMOScorer._classify_intent()` — transactional → commercial → navigational → informational の優先順位で判定

#### 改善提案エンジン (`open_mythos/llmo.py`)

- `Improvement` dataclass — `axis / priority / suggestion / expected_delta`
- `LLMOScorer.suggest_improvements(text, query, *, max_suggestions)` — 優先度高→中→低 + expected_delta 降順でソートされた改善提案リスト

#### LLMOOptimizer — ルールベース自動最適化 (`open_mythos/llmo.py`)

- `OptimizedResult` dataclass — `original_score / optimized_score / iterations / transformations_applied / final_text`
- `LLMOOptimizer(scorer, *, entity_weight, directness_weight, citability_weight)` — 重み付きスコアで反復最適化
- `optimize(text, *, query, target_score, max_iterations)` — 目標スコア到達まで最大 N 回反復
- `rewrite_for_answer_first(text)` — 「なぜなら」センテンスを文頭に移動 (answer-first 変換)
- 変換種別: `boost_entity_density` / `add_structure` (## Overview/Details/Summary) / `add_citation_cues` / `expand_content` (FAQ + Use Cases) / `inject_query_keyword`

#### API エンドポイント追加 (`serve/api.py`)

- `POST /v1/llmo/suggest` — テキスト + クエリから優先順位付き改善提案リストを返す
- `POST /v1/llmo/optimize` — LLMOOptimizer による自動最適化テキストと変換履歴を返す
- `POST /v1/llmo/score` — クエリ考慮スコアリング (query_relevance + intent_type 付き)

#### テスト (`tests/test_sprint19.py`)

- 42 tests — TestScoreWithQuery / TestSuggestImprovements / TestLLMOOptimizer / TestLLMOAPILogic / TestLLMOSprint19Integration

---

## [0.21.0] — 2026-06-01

### Sprint 18: ファインチューニング実証 & マーケティング分析強化

#### ROAS Monte Carlo シミュレーター (`open_mythos/tools_marketing.py`)

- `roas_simulate()` — モンテカルロ法による ROAS 予測 (信頼区間付き)
  - 各パラメータ (ctr / cvr / aov) に ±20% 一様ノイズを加えて n 回シミュレーション
  - 返却値: mean_roas / p5 / p25 / p50 / p75 / p95 / std_dev / profitable_probability / expected_revenue_usd
  - シード固定で再現可能、noise=0 で決定論的動作

#### ペルソナ × 広告マッチング (`open_mythos/tools_marketing.py`)

- `persona_ad_match()` — TF-IDF コサイン類似度ベースのペルソナ×広告スコアリング
  - 外部依存なし (pure Python + math)
  - 返却値: ranked (top_k件) / best_match / best_score / persona_keywords

#### LLMO スコア比較ベンチマーク (`benchmark/compare_opus.py`)

- OpenMythos LLMOScorer vs ルールベースライン (Opus 4.8 代替) の自動比較ツール
- 6 組み込みテストケース (日英 SEO / 広告コピー / 技術文書)
- オプション: `--claude` で実 Claude API と比較 (ANTHROPIC_API_KEY 必要)
- CLI: `python benchmark/compare_opus.py --input data/seo_train.jsonl --n 50`
- 結果保存: `benchmark/results/opus_comparison_YYYYMMDD_HHMMSS.json`

#### A/B テストエンドポイント (`serve/api.py`)

- `POST /v1/ab/infer` — hash(user_id) % 100 でトラフィックを振り分け
  - `AB_OPENMYTHOS_PCT` 環境変数で OpenMythos 比率を変更 (デフォルト 20%)
  - openmythos グループ: モデル直接推論
  - existing_ml グループ: 決定論的スタブ (既存 ML 代替)
- `GET /v1/ab/stats` — グループ別リクエスト数 / 平均レイテンシ / 平均スコア + Welch t 検定

#### テスト (`tests/test_sprint18.py`)

- 39 tests — roas_simulate / persona_ad_match / compare_opus / A/B router

---

## [0.20.0] — 2026-06-01

### Sprint 17: Mythos 公開対応 & 外販 API 完成

#### APIキー認証 (`serve/auth.py`)

- `verify_api_key` — FastAPI dependency。Bearer Token 検証 (env: `API_KEY` / `API_KEYS`)
- 環境変数未設定時は開発モード (認証スキップ) として動作
- `app = FastAPI(dependencies=[Depends(verify_api_key)])` でグローバル適用

#### レート制限 (`serve/auth.py`)

- `_SlidingWindow` — スレッドセーフなスライディングウィンドウ方式 (外部依存なし)
- `RateLimitMiddleware` — Starlette ミドルウェア。`/health` エンドポイントはスキップ
- デフォルト 60 rpm (env: `RATE_LIMIT_RPM` で変更可)
- レスポンスヘッダ: `X-RateLimit-Limit` / `X-RateLimit-Remaining` / `Retry-After`

#### Docker 本番イメージ (`serve/Dockerfile`)

- Gunicorn + UvicornWorker による本番グレードのマルチワーカー構成
- 非 root ユーザー (`mythos`) でプロセスを実行 (セキュリティ強化)
- `WORKERS` 環境変数でワーカー数を実行時に変更可能

#### docker-compose.yml 更新

- `RATE_LIMIT_RPM` / `API_KEY` / `WORKERS` 設定を追記
- `start_period: 60s` のヘルスチェック (モデルロード時間を考慮)
- JSON logging 設定追加

#### OpenAPI ドキュメント整備

- 全エンドポイントに `tags` / `summary` / `description` を追加
- `openapi_tags` 一覧: health / infer / generate / agent / chat / seo / thinking / tools / rag / sessions / batch
- `serve/api.py` バージョンを `0.20.0` に更新

#### Mythos 公開対応ドキュメント

- `docs/mythos_vs_openmythos.md` — アーキテクチャ差分・ベンチマーク比較・移行ガイド
- セキュリティ / コンテキストドリフト / API 互換性 / SEO 特化機能の比較表

Tests: 40 tests PASS (`tests/test_sprint17.py`)

---

## [0.17.0] — 2026-05-30

### Sprint 8: GPU訓練 & ベンチマーク一括実行 & Cloud Runデプロイ

#### 事前学習スクリプト (`scripts/pretrain.py`)

- `StreamingTokenDataset` — FineWeb-Edu streaming (sample-10BT) からの逐次チャンク読み込み
- `warmup_stable_decay_schedule()` — 3-phase LR (linear warmup → stable → cosine decay)
- bf16 autocast (CUDA) / float32 fallback (CPU)
- `TransformerBlock.gradient_checkpointing` による GPU メモリ削減
- チェックポイント形式: `{"model": state_dict, "cfg": cfg}` (perplexity.py / lm_eval_harness.py 互換)
- seq_len を `model.cfg.max_seq_len` に自動クリップ (RoPE クラッシュ防止)
- CLI: `--variant`, `--max-tokens`, `--grad-accum`, `--save-every`, `--eval-every`, `--resume`

#### GCP T4 実行スクリプト (`scripts/pretrain_gcp.sh`)

- tmux + nohup でバックグラウンド実行
- VM作成・依存インストール手順のコメント付き
- `VARIANT`, `BATCH`, `GRAD_ACCUM` 等を環境変数で上書き可能

#### ベンチマーク一括実行 (`benchmark/run_eval.py`)

- `run_perplexity_eval()` — WikiText-2 test-set PPL 評価
- `run_lm_eval()` — HellaSwag / ARC-Easy / WinoGrande (lm-eval 未インストール時は LAMBADA fallback)
- `save_results()` — `benchmark/results/{variant}_{timestamp}.json` に保存
- `update_readme()` — README.md の `<!-- BENCHMARK_TABLE_START/END -->` タグを自動更新
- `benchmark/results/` ディレクトリ追加

#### GCP Cloud Run デプロイ (`serve/deploy_cloudrun.sh`, `serve/cloudrun.env.example`)

- Artifact Registry へのイメージプッシュ
- `gcloud run deploy` (--memory 4Gi, --cpu 2, --concurrency 10, --min-instances 0, --max-instances 3)
- デプロイ後にサービス URL とサンプル curl コマンドを表示

Tests: 440+ PASS (up from 420)

---

## [0.16.0] — 2026-05-29

### Sprint 12: ReAct エージェントループ & プロンプトキャッシュ & 会話メモリ

#### 戦略
A) ReAct Agent Loop — Tool Use を活用した自律エージェント
B) Prompt KV Prefix Cache — 共通プレフィックスの prefill 省略で TTFT 削減
C) Conversation Memory — rolling window + 要約圧縮 + セッション管理

#### ReAct エージェントループ (`open_mythos/react.py`) [新規]

- `AgentStep` — 1ステップの思考/行動/観察コンテナ (thought/action/observation/answer)
- `AgentResult` — エージェント実行結果 (steps/iterations_used/stopped_reason)
- `ReActAgent` — Think→Act→Observe ループエンジン
  - `parse_tool_calls()` でツール呼び出しを検出して実行
  - `max_iterations` で暴走防止 (デフォルト: 6)
  - 停止条件: "Final Answer:" パターン / ツール不要 / max_iterations
- `format_agent_trace()` — エージェント実行ログをデバッグ用テキストに整形

#### プロンプト KV プレフィックスキャッシュ (`open_mythos/prefix_cache.py`) [新規]

- `PrefixCacheEntry` — prefill 済み KV ロジットキャッシュエントリ
- `PromptPrefixCache` — LRU キャッシュ管理 (max_entries, hit_rate 統計)
  - `cache_prefix()` — プレフィックスを prefill してキャッシュ保存
  - `generate_with_cache()` — キャッシュヒット時は prefill をスキップして継続生成
  - `evict()` / `clear()` — キャッシュ操作
- `CachedGenResult` — cache_hit / prefill_skipped_tokens などの統計付き生成結果

#### 会話メモリ (`open_mythos/conversation.py`) [新規]

- `Turn` — 1ターン (role / content / turn_id / metadata)
- `MemorySummary` — 圧縮されたサマリー (compression_ratio 付き)
- `ConversationMemory` — rolling window + 自動要約圧縮
  - `add_turn()` / `add_user()` / `add_assistant()` — ターン追加
  - `to_context_string()` — プロンプト用コンテキスト文字列生成
  - `to_messages()` — OpenAI messages 形式変換
  - `compress_now()` — 手動圧縮
- `SessionStore` — セッション ID ベースのメモリ管理 (LRU eviction / TTL)

#### Serve API 拡張 (`serve/api.py`)

- `POST /v1/agent/run` — ReAct エージェントループでタスクを解決
- `POST /v1/sessions` — 新規会話セッション作成
- `GET /v1/sessions/{id}` — セッション統計取得
- `DELETE /v1/sessions/{id}` — セッション削除
- `POST /v1/sessions/{id}/turns` — ターン追加
- `GET /v1/sessions/{id}/context` — コンテキスト文字列取得

---

## [0.15.0] — 2026-05-29

### Sprint 11: Tool Use / Long Context (YaRN RoPE) / RAG Pipeline

#### 戦略
A) Tool Use/Function Calling — OpenAI互換ツール実行エンジン
B) Long Context 32K — YaRN Dynamic NTK-aware RoPE スケーリング
C) RAG Pipeline — numpy cosine similarity + FAISS オプション

#### Tool Use エンジン (`open_mythos/tools.py`) [新規]

- `ToolDefinition` — OpenAI 互換 function definition (to_openai_schema() 実装)
- `ToolCall` / `ToolResult` — ツール呼び出し・結果スキーマ (OpenAI tool message 互換)
- `ToolRegistry` — ツール登録・検索・実行レジストリ (global/local 両対応)
- `@tool` デコレータ — 関数を自動登録 (シグネチャからパラメータ自動推論)
- `execute_tool_call()` / `execute_tool_calls()` — エラーハンドリング付き実行エンジン
- `parse_tool_calls()` — `<tool_call>` / ` ```json ` 両フォーマット対応パーサ
- `build_tool_prompt()` — ツール一覧をシステムプロンプトに整形

#### マーケ特化ツール (`open_mythos/tools_marketing.py`) [新規]

- `search_competitor(company, metric, period)` — 競合広告費・CTR・SEOスコア取得
- `calculate_roi(ad_spend, revenue, cogs, clicks, impressions)` — ROI/ROAS/CTR/CPA計算
- `fetch_trend(keyword, region, category)` — トレンドスコア・LLMO人気度・関連KW
- `score_content(text, target_keyword, style)` — SEO/LLMOスコア + 改善推奨生成
- `register_marketing_tools(registry)` — 4ツールをレジストリに一括登録

#### Long Context YaRN RoPE (`open_mythos/rope_extension.py`) [新規]

- `yarn_rope_freqs()` — YaRN (Yet another RoPE extensioN) 周波数生成 (Peng et al., 2023)
  - 高周波成分保護 / 低周波成分延伸 / 中間は beta_fast/beta_slow で滑らか補間
- `get_rope_freqs()` — config に応じて none/linear/ntk/yarn を自動選択
- `RopeScalingConfig` — スケーリング設定 (for_32k() / for_8k() ファクトリ)
- `extend_model_context()` — 学習済みモデルの freqs_cis を in-place で 32K 対応に差し替え
- `MythosConfig.rope_scaling_factor` — config からスケーリング係数を参照可能

#### RAG Pipeline (`open_mythos/rag.py`) [新規]

- `Document` — テキスト・埋め込みベクトル・メタデータのコンテナ
- `VectorStore` — numpy cosine similarity + FAISS オプション (graceful fallback)
- `_BagOfCharsEncoder` — 外部モデル不要の軽量 Bag-of-Chars 埋め込みエンコーダ
- `RAGPipeline.add_documents()` — テキストをエンコードしてインデックスに追加
- `RAGPipeline.retrieve()` — クエリに類似したドキュメント検索 (top_k)
- `RAGPipeline.generate_with_context()` — 検索結果をコンテキストに組み込んで生成

#### Serve API 拡張 (`serve/api.py`)

- `GET /v1/tools` — 利用可能ツール一覧 (OpenAI 互換 schema)
- `POST /v1/tools/call` — 単一ツール呼び出し
- `POST /v1/tools/batch` — 最大16件の一括ツール呼び出し
- `POST /v1/rag/index` — ドキュメントをRAGインデックスに追加
- `POST /v1/rag` — RAG検索 + 生成 (generate=False で検索のみ)

---

## [0.14.0] — 2026-05-27

### Sprint 10: LLMO生成 & Extended Thinking & Structured Output & DPO

#### 戦略
競合 (Jasper AI・MarketMuse・ClaudeMythos) に対して3軸で差別化:
① LLMO生成パイプライン（独自差別化）② Extended Thinking（ClaudeMythos追随）③ Structured Output/DPO（競合パリティ）

#### LLMO スコアリングモジュール (`open_mythos/llmo.py`) [新規]

- `LLMOScorer` — entity_density / answer_directness / citability の 3 スコア計算エンジン
- `LLMOScore` — スコア集約データクラス (entities リスト・word_count・sentence_count 含む)
- `LLMOScorer.batch_score()` — 複数テキストの一括スコアリング
- `LLMOScorer.rank()` — LLMO スコア降順ランキング
- `LLMOScorer.compare()` — 2テキストの差分比較 (improvement_pct 付き)
- entity_density: エンティティ密度 sigmoid 正規化。15個/100語 ≈ 0.75
- answer_directness: 冒頭 1 文の answer-first パターン + 短文スコア
- citability: 引用誘発パターン + 構造マーカー + 文書長 bell curve + 平均文長

#### SEO/LLMO コンテンツ生成パイプライン (`scripts/generate_seo.py`) [新規]

- `generate_seo_content()` — スタイル別 SEO コンテンツ生成 + LLMO スコア付き出力
- 3 スタイル: `answer_first` / `faq` / `entity_rich`
- `generate_all_styles()` — 3 スタイル全比較・LLMO スコア降順ソート
- モデルの `n_loops` を活用した精度/速度トレードオフ制御
- max_prompt_len で freqs_cis ブロードキャストエラーを防止

#### SEO/LLMO API エンドポイント (`serve/api.py`) [追加]

- `POST /v1/seo/score` — テキスト → {entity_density, answer_directness, citability, llmo_total}
- `POST /v1/seo/generate` — prompt+style → LLMO スコア付き生成テキスト

#### Extended Thinking (`open_mythos/thinking.py`) [新規]

- `ThinkingEngine` — ClaudeMythos Extended Thinking に対応するオープン実装
- `generate_with_thinking(prompt, think_loops, answer_loops)` — 思考/回答フェーズ分離生成
- `_LoopCaptureBlock` — 各ループの隠れ状態ノルムを non-intrusive にキャプチャ
- `ThinkingResult` — `{thinking: str, answer: str, loop_states: list, ...}` 集約
- `_classify_loop_phase()` — ノルム変化を Exploring/Refining/Converging/Stable に分類
- Thinking フェーズは ACT 早期終了なし（全ループ走破）で内部状態を完全記録

#### Extended Thinking API エンドポイント (`serve/api.py`) [追加]

- `POST /v1/thinking` — <thinking>...</thinking> ブロック付き回答生成
- `think_loops` / `answer_loops` / `include_loop_states` パラメータ対応

#### Structured Output / JSON Mode (`open_mythos/structured.py`) [新規]

- `StructuredGenerator` — JSON Schema 準拠の構造化出力生成エンジン
- `SchemaValidator` — JSON Schema (object/array/string/number/boolean/integer) 検証
- 3 内蔵スキーマ: `AD_PERFORMANCE_SCHEMA` / `MARKETING_REPORT_SCHEMA` / `SEO_CONTENT_SCHEMA`
- `BUILTIN_SCHEMAS` ディクショナリ (`ad_performance` / `marketing_report` / `seo_content`)
- `_complete_json()` — 不完全 JSON の自動補完ヘルパー
- `_coerce_value()` — 型・enum・range の強制変換
- n_attempts リトライ + フォールバックで例外を投げない設計

#### DPO Fine-tuning (`scripts/train_dpo.py`) [新規]

- `compute_dpo_loss()` — DPO 損失 (Rafailov et al., 2023) の純 PyTorch 実装
- 参照モデルとのlog prob差分: `β·(log π - log π_ref)` の sigmoidal 比較
- `train_dpo()` — AdamW + 線形 warmup LR スケジューラ + 勾配クリッピング
- `generate_sample_data()` — テスト用 JSONL preference pair 自動生成
- `DPOConfig` — β / lr / epochs / warmup_steps 設定データクラス
- 学習中 reward_margin・accuracy をリアルタイム表示

#### テスト

- `tests/test_sprint10.py` — 52 テスト追加
- 全体: **560 PASS** (508 → +52)

---

## [0.13.0] — 2026-05-25

### Sprint 9: マーケティング評価強化 & バッチ API & v0.13.0

#### マーケティング特化評価 (`scripts/eval_marketing.py`) [新規]

- `evaluate_ctr_prediction()` — CTR/CVR/ROAS の MAE・RMSE・Spearman 相関評価
- `evaluate_content_quality()` — 品質スコア・LLMO 可視性の MAE・Spearman 評価
- `evaluate_persona_classification()` — ペルソナ分類精度・クラス数レポート
- `evaluate_ad_performance_tier()` — 広告 Tier 分類 Accuracy・high-tier F1・ordinal MAE
- `run_evaluation(task, records, out_dir)` — CSV レポート一括出力
- `TASK_EVALUATORS` ディスパッチテーブル（4 タスク対応）

#### A/B テスト統計的有意性検定 (`serve/ab_router.py`)

- `_significance_test(a, b)` — Welch t 検定（stdlib のみ、scipy 不要）
- `/ab/stats` レスポンスに `significance_test.p_value` / `significant` フィールド追加

#### バッチ推論 API (`serve/api.py`)

- `POST /v1/batch` — 最大 64 テキストの一括推論エンドポイント
- `BatchRequest` / `BatchResponse` / `BatchResponseItem` スキーマ
- タスク別ループ数自動適用・`total_latency_ms` レポート

#### テスト (`tests/test_sprint9.py`)

- 46 tests 追加 (468 → **514 PASS**)

---

## [0.12.0] — 2026-05-26

### Sprint 7: サービス統合 & データパイプライン & 分散推論

#### マーケティング・SEO/LLMO タスク追加 (`serve/api.py`)

- `TaskType` に `seo_content`, `llmo_optimize`, `ad_copy`, `persona_message`, `market_summary` を追加
- `_TASK_SYSTEM_PROMPTS` — タスク別日本語システムプロンプト（E-E-A-T / LLMO / PREP法 / PASONAの法則 等）
- `TASK_LOOPS` — タスク別推奨ループ数（`llmo_optimize: 8`, `seo_content: 6`, `ad_copy: 2` 等）

#### 新エンドポイント

- `POST /generate` — `GenerateRequest(prompt, task, system_prompt, n_loops, max_new_tokens)` → `GenerateResponse(text, task, latency_ms, prompt_len)`
- `GET /generate/stream` — SSE ストリーミング生成（task, prompt, max_new_tokens クエリパラメータ）
- `POST /agent` — `AgentRequest(task_input, task, session_id, system_prompt)` → `AgentResponse(response, session_id, turn, latency_ms)`（セッション管理付き）
- `DELETE /agent/{session_id}` — エージェントセッションリセット（存在しない場合 404）

#### `/health` 拡張

- `active_sessions` フィールド追加（現在セッション数）
- `endpoints` フィールド追加（利用可能エンドポイント一覧）

#### serve/ 統合テスト (`tests/test_sprint7_serve.py`)

- `serve.ab_router` — ルーティング決定性・A/B 集計・スキーマ検証 (6 tests)
- `serve.sla_router` — ループ数/バジェット解決・全タスク/モード網羅・設定更新 (7 tests)
- `serve.monitor` — PSI 計算・SQLite ログ・ドリフト検知・ベースライン設定 (9 tests)
- `serve.api` — MythosConfig 構築・TASK_LOOPS 網羅・Pydantic スキーマ検証 (8 tests)
- エンドポイント統合テスト: /generate, /agent, /health（40 tests）

#### データパイプライン テスト (`tests/test_sprint7_data.py`)

- `scripts.preprocess` — 4タスクのプロンプトビルダー・load_jsonl・split_dataset (20 tests)
- `scripts.csv_to_jsonl` — auto_detect_mapping 自動列推定 (5 tests)
- `stream_dataset` / `preprocess_stream` — ストリーミング統合 (8 tests)

#### HuggingFace Datasets ストリーミング統合 (`scripts/preprocess.py`)

- `stream_dataset(paths, chunk_size)` — JSONL を chunk 単位でメモリ効率よく読み込む
- `preprocess_stream(paths, task, chunk_size, tokenizer, max_length)` — ストリーム前処理

#### 分散推論サポート (`open_mythos/main.py`)

- `OpenMythos.to_distributed(device_ids, output_device)` — `nn.DataParallel` ラッパー
- CPU 環境では self をそのまま返す graceful fallback

Tests: 420+ PASS (up from 380)

---

## [0.11.0] — 2026-05-24

### Sprint 6.4: ベンチマーク & 評価 (`benchmark/`)

#### Perplexity 評価 (`benchmark/perplexity.py`)

- `evaluate_perplexity(model, token_ids, device, seq_len, stride, batch_size, n_loops)` — sliding window PPL (GPT-2 プロトコル)
- `_load_corpus(dataset_name, split)` — HuggingFace `datasets` 経由で WikiText-2/103 を取得
- `_tokenize_corpus(text, vocab_size)` — MythosTokenizer + byte-level fallback + vocab clip
- CLI: `--variant`, `--checkpoint`, `--dataset`, `--split`, `--seq-len`, `--stride`, `--n-loops`, `--max-tokens`

#### スループット & レイテンシ計測 (`benchmark/throughput.py`)

- `measure_latency(model, prompt_ids, device, max_new_tokens, batch_size, n_loops, warmup_iters)` — TTFT / TPOT / throughput / peak memory を `LatencyResult` dataclass で返却
- `sweep_batch_sizes(model, prompt_ids, device, batch_sizes, ...)` — バッチサイズ一覧をスイープ
- CUDA 同期付きタイミング (`torch.cuda.synchronize`)、CPU では `psutil` RSS フォールバック
- CLI: `--batch-sizes 1,2,4`, `--output-json` オプション

#### lm-evaluation-harness 統合 (`benchmark/lm_eval_harness.py`)

- `MythosLMEvalWrapper(LM)` — EleutherAI lm-eval の `LM` インターフェース実装
- `loglikelihood`, `loglikelihood_rolling`, `generate_until` の3メソッドを実装
- `lm_eval` 未インストール時は LAMBADA standalone fallback で精度評価可能
- `lm_eval.simple_evaluate()` 経由で HellaSwag / ARC / WinoGrande 等を実行可能

Tests: 380 PASS (up from 347)

---

## [0.10.0] — 2026-05-24

### Sprint 6.3: エージェント統合 (`open_mythos/agents.py`)

#### OpenMythosLLM — LangChain 互換アダプタ

- `OpenMythosLLM(model, device, max_new_tokens, temperature, top_k, top_p)`
- `run(prompt) -> str` — LangChain 不要のスタンドアロン呼び出し
- `stream(prompt) -> Iterator[str]` — トークン単位ストリーミング
- LangChain `BaseLLM` プロトコル実装: `_generate()` / `_stream()` / `_llm_type`
- `from_variant(variant, checkpoint, device)` / `from_pretrained(repo_id_or_path)`
- `langchain-core` 未インストール時は `run()` / `stream()` のみ利用可能、パイプライン API は `ImportError`
- 空プロンプト → BOS fallback で RoPE クラッシュを回避

#### MythosAgent — Swarms 互換ラッパー

- `run(task) -> str` / `stream_run(task) -> Iterator[str]`
- `__call__(task)` — Swarms パイプラインへの drop-in 対応
- `system_prompt` + 直近 2 ターン会話履歴を自動コンテキスト構築
- `reset()` で会話履歴をクリア
- `from_variant()` / `from_pretrained()` コンストラクタ
- `swarms` 未インストール時もスタンドアロンで動作

Tests: 347 PASS (up from 324)

---

## [0.9.0] — 2026-05-24

### Sprint 6.2: 訓練スクリプト高度化

#### TrainLogger — 統一実験ロギング (`open_mythos/logger_utils.py`)

- `TrainLogger(backend, run_name, project, config, log_dir)` — WandB / MLflow / TensorBoard / none を統一 API で切り替え
- 全バックエンドで `log(metrics, step)` / `log_artifact(path)` / `finish()` を提供
- `wandb` / `mlflow` / `tensorboard` 未インストール時は警告のみで `"none"` に graceful fallback
- コンテキストマネージャ対応 (`with TrainLogger(...) as log:`)

#### argparse CLI フラグ

- `_parse_args()` 追加で全ハイパーパラメータを CLI から上書き可能に
- `--resume PATH` — 特定チェックポイントを指定再開（省略時は `--ckpt-dir` の最新を自動選択）
- `--ckpt-dir` / `--ckpt-every` / `--keep-last` — チェックポイント管理
- `--logger {none,wandb,mlflow,tensorboard}` / `--run-name` / `--project` / `--log-dir`
- `--seq-len` / `--micro-batch` / `--lr` / `--wd` / `--warmup-steps` / `--eval-every` / `--no-grad-ckpt`
- `TrainLogger` をトレーニングループに統合（train/eval メトリクスを step 単位で記録）

Tests: 324 PASS (up from 305)

---

## [0.8.0] — 2026-05-24

### Sprint 6.1: 推論最適化

#### torch.compile 統合

- `OpenMythos.compile_model(mode, fullgraph, dynamic, backend)` — `TransformerBlock` / `RecurrentBlock` を個別にコンパイル（再帰ループのグラフ断絶を回避）
- `backend="eager"` にフォールバックして Windows CPU でも動作
- `PyTorch < 2.0` では自動スキップ

#### SDPA 最適化（GQA / MLA）

- GQA / MLA の手動 attention を `F.scaled_dot_product_attention` に置き換え
- Flash Attention 2 / memory-efficient / math backend を自動選択
- `flash_attn` 未インストール環境でも高速化（PyTorch 2.x SDPA カーネル）

#### KV cache ページング

- `MythosConfig` に `kv_page_size: int = 64` / `kv_max_pages: int = 0` を追加
- `OpenMythos.allocate_kv_cache(max_seq_len, device)` — ページ上限付きキャッシュ辞書を生成
- `OpenMythos.free_kv_cache(cache)` — レイヤーテンソルを解放してメタキーを保持（再利用可能）
- decode ステップで window 上限超過時に古いトークンを evict

Tests: 305 PASS (up from 286)

---

## [0.7.0] — 2026-05-24

### Sprint 5: Training 品質改善 & エコシステム拡張

#### Training quality

- `warmup_stable_decay(step, warmup, stable_end, total, max_lr, min_lr)` — 3-phase LR schedule (linear warmup → flat plateau → cosine decay, DeepSeek-V3 style)
- `OpenMythos.init_mup(base_dim=256)` — Maximal Update Parametrization; optimal LR transfers across model widths
- Gradient Noise Scale (GNS) monitoring in training loop: `gns = grad_norm² / micro_batch`, rolling window logged per step
- Early stopping: `es_patience=5`, `es_min_delta=1e-3`; halts after consecutive eval steps without improvement

#### CLI

- `mythos generate` — autoregressive generation from prompt (supports `--checkpoint`, `--variant`, `--stream`, `--temperature`, `--top-k`, `--top-p`, `--device`)
- `mythos info` — print param count and config summary for any variant
- Registered as `mythos` console script in `pyproject.toml`

#### Hugging Face Hub integration

- `OpenMythos.push_to_hub(repo_id, token)` — upload weights + config to the Hub
- `OpenMythos.from_pretrained(repo_id_or_path, token, map_location)` — load from local path, directory, or Hub repo

#### LoRA finetuning API

- `OpenMythos.enable_lora_finetuning()` — freeze all params except `LoRAAdapter` weights; returns `self` for chaining
- `OpenMythos.trainable_parameters()` — yield only grad-requiring params (for optimizer construction)

Tests: 286 PASS (up from 265)

---

## [0.6.0] — 2026-05-23

### Sprint 2: Inference 高度化

#### New generation methods

- `generate_beam()` — beam search with configurable `beam_width` and `length_penalty` (B=1)
- `generate_batch()` — parallel generation for variable-length prompt lists via left-padding
- `repetition_penalty` parameter added to `generate()` and `generate_stream()`
- `max_cache_len` (sliding window KV cache) parameter added to `generate()` and `generate_stream()`

#### Quantization

- `model.quantize("fp16")` — in-place FP16 cast; dtype-safe through ACT, LTI, and MoEFFN
- `model.quantize("int8")` — dynamic INT8 quantization of all `nn.Linear` layers

#### Internal fixes

- `MoEFFN.forward`: `weight` and `expert_out` now cast to `flat.dtype` for FP16/BF16 safety
- `RecurrentBlock`: `cumulative_p` and boolean masks cast to `h.dtype`
- `LTIInjection.forward`: `A` and `B` cast to `h.dtype`
- `ACTHalting.forward`: output cast to `h.dtype`

Tests: 257 PASS (up from 227)

---

## [0.5.0] — 2026-05-21

### Sprint 1: HyperloopMythos & Inference Engine

- `HyperloopMythos` two-phase inference engine (prefill `n_loops`, decode `decode_loops`)
- `generate_stream()` — streaming autoregressive generation
- `speculative_decode()` — self-speculative decoding with depth-based draft/target split
- `top_p` nucleus sampling in `_sample_token`
- `decode_loops` two-phase depth strategy: 2.54× decode speedup
- `_causal_mask` cache_len fix for incremental decoding
- LTI `get_A()` `.clamp(min=1e-6)` fix for float32 saturation
- `freqs_cis[:T]` slice fix for apply_rope broadcast errors
- Lint: ruff/black pass, `.gitattributes` CRLF/LF unification

Tests: 227 PASS

---

## [0.1.0] — 2026-05-01

Initial release: OpenMythos RDT Phase 0–4

- `OpenMythos` model with GQA / MLA attention
- `MoEFFN` fine-grained mixture of experts
- `LTIInjection` with spectral radius < 1 guarantee
- `ACTHalting` adaptive computation time
- `LoRAAdapter` depth-wise LoRA
- `RecurrentBlock` with loop index embedding
- `HyperloopMythos` v0.1 (basic)
- Pre-configured variants `mythos_1b` … `mythos_1t`
- `MythosTokenizer`
- Training script `training/3b_fine_web_edu.py`
