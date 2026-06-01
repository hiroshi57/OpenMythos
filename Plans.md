# OpenMythos — Sprint Plans
> 最終更新: 2026-06-01 | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 1〜9 のアーカイブ: `docs/archive/sprint-plans-1-9.md`
> Sprint 10〜19 のアーカイブ: `docs/archive/sprint-plans-10-19.md`

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
| 10 | LLMO生成 / Extended Thinking / Structured Output | 560 PASS | v0.14.0 | ae264dd |
| 11 | Tool Use / Long Context / RAG | 664 PASS | v0.15.0 | d557cd5 |
| 12 | ReAct / Prompt Cache / Conversation Memory | 729 PASS | v0.16.0 | 292fd88 |
| 13 | Mixture-of-Depths / SwarmOrchestrator | 836 PASS | — | d8d9f1e |
| 14 | GPU pretrain / Benchmark / GCP deploy | — | v0.17.0 | 8cc9c5f |
| 15 | 日本語形態素 / A/B test / drift検出 | 933 PASS | v0.18.0 | 05d8526 |
| 16 | SEOパイプライン / QS予測 / 広告バリアント | — | v0.19.0 | 0c98c3f |
| 17 | APIキー認証 / レート制限 / Docker本番化 | 1012 PASS | v0.20.0 | 87b669c |
| 18 | ファインチューニング実証 / ROAS / persona_ad_match / A/B | — | v0.21.0 | 89506bc |
| 19 | LLMO強化 (query_relevance / intent / LLMOOptimizer) | 1054 PASS | v0.22.0 | fca7e84 |

---

## Sprint 20: Living LLMO — 20の自己成長パターン & コンテンツパイプライン & v0.23.0

> ブランチ: `feature/sprint20-living-llmo`

### 自己成長パターン一覧

| # | パターン | カテゴリ | 何が育つか |
|---|---------|---------|-----------|
| P1 | Edit Delta Learning | フィードバック | 変換の優先度 |
| P2 | Acceptance Learning | フィードバック | 成功変換の優先度 |
| P3 | Rating Calibration | フィードバック | スコア精度 |
| P4 | Cross-Document Learning | フィードバック | 個人化 |
| P5 | Rejection Memory | 失敗 | NG パターン辞書 |
| P6 | Failure Memory | 失敗 | 逆効果変換の抑制 |
| P7 | Loop Escape Memory | 失敗 | 収束失敗パターン |
| P8 | Anti-Pattern Registry | 失敗 | 変換の組み合わせ禁止ルール |
| P9 | Pattern Mining | 成功 | 変換ランキング |
| P10 | Entity Vocabulary Growth | 成功 | entity 辞書 |
| P11 | Template Crystallization | 成功 | 文章テンプレート |
| P12 | Champion Promotion | 成功 | チャンピオン変換セット |
| P13 | Domain Specialization | 環境 | ドメイン別プロファイル |
| P14 | Temporal Decay | 環境 | パターンの鮮度 |
| P15 | Trend Adaptation | 環境 | トレンド対応 |
| P16 | Regret Minimization | 環境 | 後知恵学習 |
| P17 | Adaptive Target Calibration | 自律 | 目標スコアの精度 |
| P18 | Self-Benchmark | 自律 | スコアラーの校正 |
| P19 | Growth Cycle Scheduler | 自律 | 成長のタイミング |
| P20 | Meta-Learning | 自律 | 学習戦略そのもの |

### タスク

| task-id | 説明 | 状態 |
|---------|------|------|
| 20.1.1 | `open_mythos/content_pipeline.py` — ContentPipeline (keyword→outline→draft→LLMO最適化) | cc:TODO |
| 20.1.2 | `open_mythos/llmo.py` — `LLMOScorer.batch_score(texts)` | cc:TODO |
| 20.1.3 | `scripts/content_workflow.py` — CLI: `--keyword --target-score --intent` | cc:TODO |
| 20.2.1 | `open_mythos/llmo_feedback.py` — FeedbackStore + FeedbackAnalyzer (P1〜P4) | cc:TODO |
| 20.2.2 | `open_mythos/llmo.py` — `LLMOScorer.adapt_weights(store)` | cc:TODO |
| 20.3.1 | `open_mythos/llmo_growth.py` — RejectionMemory (P5) + FailureMemory (P6) | cc:TODO |
| 20.3.2 | `open_mythos/llmo_growth.py` — LoopEscapeMemory (P7) + AntiPatternRegistry (P8) | cc:TODO |
| 20.4.1 | `open_mythos/llmo_growth.py` — PatternMiner (P9) + EntityKnowledgeBase (P10) | cc:TODO |
| 20.4.2 | `open_mythos/llmo_growth.py` — TemplateLibrary (P11) + ChampionPromoter (P12) | cc:TODO |
| 20.5.1 | `open_mythos/llmo_adapt.py` — DomainSpecializer (P13) + TemporalDecay (P14) + TrendAdapter (P15) + RegretMinimizer (P16) | cc:TODO |
| 20.5.2 | `open_mythos/llmo_adapt.py` — AdaptiveTargetCalibrator (P17) + SelfBenchmark (P18) | cc:TODO |
| 20.5.3 | `open_mythos/llmo_adapt.py` — GrowthCycleScheduler (P19) + MetaLearner (P20) | cc:TODO |
| 20.6.1 | `open_mythos/llmo_adapt.py` — 自動トリガー: FeedbackCountTrigger / TimerTrigger / ScoreDriftTrigger / CompositeTrigger | cc:TODO |
| 20.6.2 | `open_mythos/llmo_adapt.py` — GrowthCycle.run() (20パターン順次実行・スナップショット保存) | cc:TODO |
| 20.7.1 | `open_mythos/llmo_history.py` — GrowthSnapshot / GrowthHistory / GrowthDiff | cc:TODO |
| 20.7.2 | `serve/api.py` — `/v1/llmo/growth/history` / `/snapshot` / `/diff` / `/trigger` / `/patterns` | cc:TODO |
| 20.7.3 | `serve/api.py` — `/v1/llmo/growth/report` — HTML 成長レポートページ | cc:TODO |
| 20.8.1 | `serve/api.py` — `/v1/content/generate` / `/v1/llmo/feedback` / `/v1/llmo/batch` / `/v1/llmo/growth/stats` | cc:TODO |
| 20.T | `tests/test_sprint20.py` — 60 tests | cc:TODO |
| 20.V | PyPI v0.23.0 — pyproject.toml + CHANGELOG Sprint 20 追加 | cc:TODO |

---

## Sprint 21: (計画中)

> 前提: Sprint 20 (Living LLMO) 完了後

---

## 進行中の作業メモ

- `master`: `fca7e84` — Sprint 1〜19 全完了 (1054 PASS / v0.22.0)
- `freqs_cis` は必ず `[:T]` スライスして渡すこと (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` は `.clamp(min=1e-6)` が必要 (float32 飽和防止)
