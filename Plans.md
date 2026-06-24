# OpenMythos — Sprint Plans
> 最終更新: 2026-06-16 (Sprint 60 完了) | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 60 完了: 広告キャンペーン管理 (CEP→コピー生成→評価フロー) → v0.63.0
> アーカイブ: Sprint 1〜9   → `docs/archive/sprint-plans-1-9.md`
>            Sprint 10〜19  → `docs/archive/sprint-plans-10-19.md`
>            Sprint 20〜25  → `docs/archive/sprint-plans-20-25.md`
>            Sprint 26〜35  → `docs/archive/sprint-plans-26-35.md`
>            Sprint 36〜43  → `docs/archive/sprint-plans-36-42.md`
>            Sprint 43〜51  → `docs/archive/sprint-plans-43-51.md`
>            Sprint 52〜58  → `docs/archive/sprint-plans-52-58.md`

---

## Sprint 全サマリー

| Sprint | テーマ | コアモジュール | テスト | Ver |
|--------|--------|--------------|--------|-----|
| 1〜9 | 基盤構築 / Serving / Marketing eval | `main.py` `serve/api.py` | 508 | v0.13 |
| 10〜19 | LLMO / Tool Use / RAG / ReAct / SEO | `llmo.py` `tools.py` `rag.py` | 1079 | v0.22 |
| 20〜30 | 育つAI P1〜P10 + 統合 | `debate.py`〜`growing_ai_orchestrator.py` | 1617 | v0.33 |
| 31〜35 | LoRA SFT / SQLite / FAISS / Benchmark | `lora_trainer.py` `error_memory.py` | 1822 | v0.38 |
| 36〜43 | API強化 / Hermes Layer2 | `serve/api.py` `hermes_orchestrator.py` | 2260 | v0.46 |
| 44 | **Vector DB 統合 + Instructor 構造化出力** | `skills/vector_store.py` `skills/instructor_extract.py` | 2322 | v0.47 |
| 45 | **HuggingFace Hub 統合** | `skills/hf_hub.py` | 2384 | v0.48 |
| 46 | **推論バックエンド統合** | `skills/inference_backends.py` | 2439 | v0.49 |
| 47 | **研究ツール統合** | `skills/research_tools.py` | 2486 | v0.50 |
| 48 | **マルチモーダル統合** | `skills/multimodal.py` | 2534 | v0.51 |
| 49 | **訓練最適化統合** | `skills/training_optimization.py` | 2583 | v0.52 |
| 50 | **エージェントフレームワーク強化** | `skills/agent_framework.py` | 2655 | v0.53 |
| 51 | **データ・検索ツール統合** | `skills/data_tools.py` | 2708 | v0.54 |
| 52 | **DevOps・クラウド統合** | `skills/devops_cloud.py` | 2756 | v0.55 |
| 53 | **セキュリティ統合** | `skills/security.py` | 2797 | v0.56 |
| 54 | **OpenAI Assistants API 統合** | `assistant.py` | 2862 | v0.57 |
| 55 | **ストリーミング & SSE 応答** | `open_mythos/streaming.py` | 2920 | v0.58 |
| 56 | **マルチプロバイダー LLM** | `skills/llm_providers.py` | 2972 | v0.59 |
| 57 | **LLM 評価フレームワーク** | `skills/evaluation.py` | 3026 | v0.60 |
| 58 | **LLMO ダッシュボード・CEP管理・競合分析** | `skills/llmo_dashboard.py` | 3078 | v0.61 |
| 59 | **自律脆弱性スキャン (harness 移植)** | `skills/vuln_scanner.py` | 3155 | v0.62 |
| 60 | **広告キャンペーン管理** | `skills/campaign_manager.py` | 3240 | v0.63 |
| 61 | **日本語対応トークナイザー** | `open_mythos/tokenizer_ja.py` | 3301 | v0.64 |
| 62 | **LLM コピー生成強化** | `skills/llm_copy_generator.py` | 3357 | v0.65 |
| 63 | **A/Bテスト + 分析DB + 形態素解析** | `skills/ab_test.py` `skills/campaign_analytics.py` `tokenizer_ja.py` | 3492 | v0.66 |
| 64 | **A/B+分析API + 形態素連携 + 予算最適化** | `serve/api.py` `skills/budget_optimizer.py` | 3542 | v0.67 |
| 65 | **Fusion マルチモデル融合 (OpenRouter移植)** | `skills/fusion.py` | 3593 | v0.68 |
| 66 | **Fusionストリーミング + A/B予算連携 + 異常検知** | `skills/campaign_orchestrator.py` `skills/anomaly_detector.py` | 3650 | v0.69 |
| 67 | **異常検知自動予算停止 + Fusionキャッシュ + 統合ダッシュボード** | `skills/fusion_cache.py` `campaign_orchestrator.py` | 3703 | v0.70 |
| 68 | **セキュリティインテリジェンス統合 + リスクカテゴリ分類** | `skills/security_intel.py` `skills/security.py` | 3780 | v0.71 |
| 69 | **時系列予測統合 TimesFM + マルチモデル** | `skills/time_series.py` | 3842 | v0.72 |
| 70 | **予測アラート統合 + レポートWebhook + NLQエージェント** | `skills/forecast_alert.py` `skills/report_dispatcher.py` `skills/nlq_agent.py` | 3937 | v0.73 |
| 71 | **主要都市地図ビジュアライザ** | `skills/city_map.py` `skills/map_renderer.py` | 4014 | v0.74 |
| 72 | **地図拡張 比較/編集/レポート** | `skills/map_comparator.py` `skills/map_editor.py` `skills/map_report.py` | 4085 | v0.75 |
| 73 | **地図アニメーション/経路探索/インポート** | `skills/map_animator.py` `skills/route_finder.py` `skills/map_importer.py` | 4161 | v0.76 |
| 74 | **混雑シミュレーション/アクセシビリティ/地下水位** | `skills/crowd_simulator.py` `skills/accessibility.py` `skills/groundwater.py` | 4238 | v0.77 |
| 75 | **環境センサー/乗り換え最適化/インフラダッシュボード** | `skills/env_sensor.py` `skills/transfer_optimizer.py` `skills/infra_dashboard.py` | 4319 | v0.78 |

> **累計テスト数**: 4319 PASS (Sprint 75: +81) — **Sprint 76 候補検討中**

---

## Sprint 59 詳細 (最新 / 完了)

### Sprint 59: 自律脆弱性スキャン — v0.62.0
> 参照: `external/defending-code-harness/` (anthropics/defending-code-reference-harness, Apache-2.0)
> harness の 4 ステージパイプラインを Python-native 静的解析に移植

| task-id | 説明 | 状態 |
|---------|------|------|
| 59.1 | `skills/vuln_scanner.py` — VulnSeverity/VulnCategory/PatchStatus (Enum 層) | cc:完了 |
| 59.2 | `skills/vuln_scanner.py` — ScanTarget (harness:TargetConfig) / VulnFinding (harness:CrashArtifact) | cc:完了 |
| 59.3 | `skills/vuln_scanner.py` — VerifyVerdict (harness:GraderVerdict 5基準) / PatchCandidate (T0/T1/T2) | cc:完了 |
| 59.4 | `skills/vuln_scanner.py` — ScanReport (harness:ReportVerdict) / ScanSession (harness:RunResult) | cc:完了 |
| 59.5 | `skills/vuln_scanner.py` — VulnStore / VulnScanner (find+recon) | cc:完了 |
| 59.6 | `skills/vuln_scanner.py` — VulnPatcher (T0/T1/T2 ラダー) / ScanReportEngine | cc:完了 |
| 59.7 | `serve/api.py` — `/v1/vuln/scan` `/v1/vuln/findings` `/v1/vuln/patch/{id}` `/v1/vuln/session/{id}/report` | cc:完了 |
| 59.T | `tests/test_sprint59.py` — 77 tests PASS (累計 3155) | cc:完了 |
| 59.V | PyPI v0.62.0 | cc:完了 |
| 59.X | `external/defending-code-harness/` クローン (read-only reference) | cc:完了 |

### harness 対応表
| harness | vuln_scanner.py | 役割 |
|---------|----------------|------|
| `TargetConfig` | `ScanTarget` | スキャン対象設定 |
| `CrashArtifact` | `VulnFinding` | 脆弱性 1 件 |
| `GraderVerdict` | `VerifyVerdict` | 5 基準スコアリング |
| `PatchVerdict (T0-T3)` | `PatchCandidate` | T0:構文/T1:消滅/T2:テスト |
| `ReportVerdict` | `ScanReport` | 悪用可能性分析 |
| `RunResult` | `ScanSession` | セッション全体 |

---

## Sprint 60 詳細 (最新 / 完了)

### Sprint 60: 広告キャンペーン管理 — v0.63.0
> CEP→コピー生成→評価→キャンペーン登録の全フローをワークフロー化

| task-id | 説明 | 状態 |
|---------|------|------|
| 60.1 | `skills/campaign_manager.py` — CampaignStatus/AdChannel/AdObjective (Enum 層) | cc:完了 |
| 60.2 | `skills/campaign_manager.py` — AdCopy/CampaignBudget/Campaign (データモデル) | cc:完了 |
| 60.3 | `skills/campaign_manager.py` — CampaignStore (CRUD) | cc:完了 |
| 60.4 | `skills/campaign_manager.py` — CopyGenerator (CEP→コピー生成) | cc:完了 |
| 60.5 | `skills/campaign_manager.py` — CampaignEvaluator/EvalResult (品質スコアリング) | cc:完了 |
| 60.6 | `skills/campaign_manager.py` — CampaignWorkflow/WorkflowResult (全フロー) | cc:完了 |
| 60.7 | `skills/campaign_manager.py` — CampaignReportEngine (Markdown/JSON レポート) | cc:完了 |
| 60.8 | `serve/api.py` — `/v1/campaign/*` 9エンドポイント | cc:完了 |
| 60.T | `tests/test_sprint60.py` — 85 tests PASS (累計 3240) | cc:完了 |

---

## Sprint 61 詳細 (最新 / 完了)

### Sprint 61: 日本語対応トークナイザー — v0.64.0
> GPT-2 英語依存を解消。外部ライブラリなしで日本語を処理する軽量トークナイザー。

| task-id | 説明 | 状態 |
|---------|------|------|
| 61.1 | `tokenizer_ja.py` — CharType/classify_char (文字種分類) | cc:完了 |
| 61.2 | `tokenizer_ja.py` — JaTokenizerConfig (設定) | cc:完了 |
| 61.3 | `tokenizer_ja.py` — JaVocab (語彙管理) | cc:完了 |
| 61.4 | `tokenizer_ja.py` — JaSentenceSplitter (文分割) | cc:完了 |
| 61.5 | `tokenizer_ja.py` — JaTokenizer (文字種境界分割 + N-gram) | cc:完了 |
| 61.6 | `tokenizer_ja.py` — JaTokenizerAdapter (MythosTokenizer 互換) | cc:完了 |
| 61.7 | `tokenizer_ja.py` — build_vocab_from_corpus (コーパス語彙構築) | cc:完了 |
| 61.T | `tests/test_sprint61.py` — 61 tests PASS (累計 3301) | cc:完了 |

---

## Sprint 62 詳細 (最新 / 完了)

### Sprint 62: LLM コピー生成強化 — v0.65.0
> CopyGenerator を LLM API 連携に拡張。API 未設定時はルールベースに自動フォールバック。

| task-id | 説明 | 状態 |
|---------|------|------|
| 62.1 | `llm_copy_generator.py` — CopyGenerationConfig / CopyGenerationResult | cc:完了 |
| 62.2 | `llm_copy_generator.py` — LLMCopyPromptBuilder (プロンプト構築) | cc:完了 |
| 62.3 | `llm_copy_generator.py` — LLMCopyParser (レスポンス解析: JSON/CodeBlock/Regex) | cc:完了 |
| 62.4 | `llm_copy_generator.py` — LLMCopyGenerator (LLM 生成 + フォールバック) | cc:完了 |
| 62.5 | `llm_copy_generator.py` — LLMCopyGeneratorFactory (from_env/from_mock/rule_based) | cc:完了 |
| 62.T | `tests/test_sprint62.py` — 56 tests PASS (累計 3357) | cc:完了 |

---

## Sprint 63 詳細 (最新 / 完了)

### Sprint 63: A/Bテスト + 分析DB + 形態素解析 — v0.66.0
> 候補 A/B/C を一括実装。広告効果測定の全レイヤーを整備。

#### 63A: A/B テスト基盤 (`skills/ab_test.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 63A.1 | VariantStatus/ABTestStatus/VariantStats/Variant | cc:完了 |
| 63A.2 | ABTest (状態遷移) / ABTestStore (CRUD) | cc:完了 |
| 63A.3 | TrafficAllocator (hash ベース重み付き振り分け) | cc:完了 |
| 63A.4 | ABTestAnalyzer (2標本比率 z 検定, 外部依存なし) / ABTestReportEngine | cc:完了 |

#### 63B: キャンペーン分析ダッシュボード (`skills/campaign_analytics.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 63B.1 | MetricType/MetricPoint/CampaignMetrics (時系列蓄積) | cc:完了 |
| 63B.2 | KpiCalculator (CTR/CVR/CPC/CPA/CPM/ROAS/ROI) | cc:完了 |
| 63B.3 | TrendAnalyzer (前期比/移動平均) | cc:完了 |
| 63B.4 | CampaignAnalyticsDashboard (横断集計/ランキング) / AnalyticsReportEngine | cc:完了 |

#### 63C: 日本語形態素解析強化 (`tokenizer_ja.py` 拡張)
| task-id | 説明 | 状態 |
|---------|------|------|
| 63C.1 | PartOfSpeech/DictionaryEntry/Morpheme | cc:完了 |
| 63C.2 | JaDictionary (デフォルト助詞・助動詞 + 最長一致辞書) | cc:完了 |
| 63C.3 | JaMorphologicalAnalyzer (最長一致法 + 名詞抽出) | cc:完了 |

| 63.T | `test_sprint63a/b/c.py` — 135 tests PASS (累計 3492) | cc:完了 |

---

## Sprint 64 詳細 (最新 / 完了)

### Sprint 64: A/B+分析API + 形態素連携 + 予算最適化 — v0.67.0
> 候補 A/B/C を一括実装。Sprint 63 の機能群を API 公開し、横断統合。
> **副次修正**: serve/api.py の Sprint 60 ブロック重複を解消（route 二重定義を削除）。

#### 64A: A/B + 分析ダッシュボード API (`serve/api.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 64A.1 | `/v1/abtest/*` 8 endpoints (作成/一覧/詳細/削除/開始/実績記録/レポート) | cc:完了 |
| 64A.2 | `/v1/analytics/*` 4 endpoints (記録/KPI/レポート/サマリー) | cc:完了 |
| 64A.3 | Sprint 60 重複ブロック削除 (route 二重定義の解消) | cc:完了 |

#### 64B: 形態素解析 → コピー生成連携 (`skills/campaign_manager.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 64B.1 | CopyGenerator(use_morphology=True) — 名詞抽出ベースのタグ生成 | cc:完了 |
| 64B.2 | 後方互換維持 (デフォルト False = 正規表現抽出) | cc:完了 |

#### 64C: 広告予算最適化 (`skills/budget_optimizer.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 64C.1 | AllocationStrategy/BudgetConstraint/BudgetAllocation/OptimizationResult | cc:完了 |
| 64C.2 | BudgetOptimizer (Equal/RoasWeighted/Performance/Proportional + 推奨戦略) | cc:完了 |
| 64C.3 | `/v1/budget/*` 2 endpoints (最適化/推奨戦略) | cc:完了 |

| 64.T | `tests/test_sprint64.py` — 50 tests PASS (累計 3542) | cc:完了 |

---

## Sprint 65 詳細 (最新 / 完了)

### Sprint 65: Fusion マルチモデル融合 — v0.68.0
> OpenRouter Fusion Server Tool を OpenMythos に移植。
> 参照: https://openrouter.ai/docs/guides/features/server-tools/fusion
> 仕組み: 候補モデル群 → 審査モデルが構造化分析 → 呼び出しモデルが最終回答合成。

| task-id | 説明 | 状態 |
|---------|------|------|
| 65.1 | `skills/fusion.py` — FusionRole/CandidateSpec/FusionConfig (設定層) | cc:完了 |
| 65.2 | `skills/fusion.py` — CandidateResponse/CandidateAnalysis/FusionAnalysis/FusionResult | cc:完了 |
| 65.3 | `skills/fusion.py` — FusionAnalysisParser (審査 JSON パース + フォールバック) | cc:完了 |
| 65.4 | `skills/fusion.py` — JudgeAnalyzer (審査ステージ, LLM + ヒューリスティック) | cc:完了 |
| 65.5 | `skills/fusion.py` — FusionEngine (3段パイプライン: 候補収集→審査→合成) | cc:完了 |
| 65.6 | `skills/fusion.py` — FusionEngineFactory (from_env/from_mock/rule_based) | cc:完了 |
| 65.7 | `serve/api.py` — `/v1/fusion/run` `/v1/fusion/status` | cc:完了 |
| 65.T | `tests/test_sprint65.py` — 51 tests PASS (累計 3593) | cc:完了 |

#### OpenRouter Fusion 対応表
| OpenRouter Fusion | fusion.py | 役割 |
|-------------------|-----------|------|
| candidate models | `CandidateSpec` / `CandidateResponse` | 候補回答生成 |
| judge model | `JudgeAnalyzer` → `FusionAnalysis` | 構造化分析 |
| caller model | `FusionEngine._synthesize` | 最終回答合成 |
| 既存 `MultiProviderRouter` を再利用（claude/openai/openmythos） | | |

---

## Sprint 66 詳細 (最新 / 完了)

### Sprint 66: Fusionストリーミング + A/B予算連携 + 異常検知 — v0.69.0
> 候補 A/B/C を一括実装。広告運用自動化レイヤーを完成。

#### 66A: Fusion ストリーミング (`skills/fusion.py` 拡張)
| task-id | 説明 | 状態 |
|---------|------|------|
| 66A.1 | FusionEngine.run_stream — 段階イベント (candidates/analysis/delta/done/error) | cc:完了 |
| 66A.2 | `/v1/fusion/stream` — SSE ストリーミング | cc:完了 |

#### 66B: A/B → 予算最適化 自動連携 (`skills/campaign_orchestrator.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 66B.1 | OrchestrationConfig/WinnerDecision/ReallocationPlan | cc:完了 |
| 66B.2 | CampaignOrchestrator — 勝者判定(有意差考慮) + 勝者ボーナス予算再配分 | cc:完了 |
| 66B.3 | `/v1/orchestrator/decide-winner` `/v1/orchestrator/reallocate` | cc:完了 |

#### 66C: KPI 異常検知アラート (`skills/anomaly_detector.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 66C.1 | AlertSeverity/AnomalyType/Alert/DetectorConfig | cc:完了 |
| 66C.2 | AnomalyDetector — z-score + 変化率 + stale 検知 | cc:完了 |
| 66C.3 | AlertStore / AnomalyReportEngine | cc:完了 |
| 66C.4 | `/v1/anomaly/{id}/detect` `/v1/anomaly/alerts` `/report/md` | cc:完了 |

| 66.T | `tests/test_sprint66.py` — 57 tests PASS (累計 3650) | cc:完了 |

---

---

## Sprint 67 詳細 (完了)

### Sprint 67: 異常検知自動予算停止 + Fusionキャッシュ + 統合ダッシュボード — v0.70.0
> 候補 A/B/C を一括実装。異常検知→自動凍結・Fusionキャッシュ・統合ダッシュボードの 3 レイヤーを整備。

#### 67A: 異常検知 → 自動予算停止 (`skills/campaign_orchestrator.py` 拡張)
| task-id | 説明 | 状態 |
|---------|------|------|
| 67A.1 | FreezeDecision / FrozenBudgetPlan (凍結判定データモデル) | cc:完了 |
| 67A.2 | CampaignOrchestrator.freeze_if_critical — Critical アラートで予算配分を自動凍結 | cc:完了 |
| 67A.3 | `/v1/orchestrator/freeze` — 凍結実行 API | cc:完了 |

#### 67B: Fusion 結果キャッシュ (`skills/fusion_cache.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 67B.1 | FusionCache — LRU + TTL、hit/miss 統計、eviction | cc:完了 |
| 67B.2 | CachedFusionEngine — FusionEngine ラッパー (stream はキャッシュ不使用) | cc:完了 |
| 67B.3 | `/v1/fusion/cached` `/v1/fusion/cache/stats` `/v1/fusion/cache/clear` | cc:完了 |

#### 67C: 広告運用 統合ダッシュボード API (`serve/api.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 67C.1 | `/v1/dashboard/summary` — KPI/アラート/A-B/予測を 1 エンドポイントに集約 | cc:完了 |
| 67C.2 | `/v1/dashboard/campaigns` — 全キャンペーン横断集計 | cc:完了 |
| 67C.3 | `/v1/dashboard/alerts/critical` — Critical アラート一覧 | cc:完了 |

| 67.T | `tests/test_sprint67.py` — 53 tests PASS (累計 3703) | cc:完了 |

---

## Sprint 68 詳細 (完了)

### Sprint 68: セキュリティインテリジェンス統合 + リスクカテゴリ分類 — v0.71.0
> VulnScanner (Sprint 59) に連動する脅威インテリジェンス収集・分類・対応プレイブック層を追加。

#### 68A: リスクカテゴリ分類 (`skills/security.py` 拡張)
| task-id | 説明 | 状態 |
|---------|------|------|
| 68A.1 | DiagnosisCategory (A〜F: 技術的脆弱性/フィッシング/コンプライアンス/インシデント/ガバナンス/AIリスク) | cc:完了 |
| 68A.2 | ThreatCategoryMapper — キーワードルール + 日本語パターン対応 | cc:完了 |
| 68A.3 | CategoryMatch (マッチ結果 + 信頼スコア) | cc:完了 |

#### 68B: セキュリティインテリジェンス (`skills/security_intel.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 68B.1 | ThreatSeverity / ThreatSource / ThreatCategory (Enum 層) | cc:完了 |
| 68B.2 | ResponsePlaybook / ThreatEnrichment / SecurityThreat (データモデル) | cc:完了 |
| 68B.3 | SecurityIntelStore (CRUD + フィルタ) / ThreatEnricher (LLM + ルールベース) | cc:完了 |
| 68B.4 | ThreatCollector (NVD/OSINT/内部ソース収集) / SecurityIntelDashboard / IntelReportEngine | cc:完了 |

#### 68C: セキュリティインテリジェンス API (`serve/api.py`)
| task-id | 説明 | 状態 |
|---------|------|------|
| 68C.1 | `/v1/intel/collect` `/v1/intel/threats` `/v1/intel/threats/{id}` (収集・一覧・詳細) | cc:完了 |
| 68C.2 | `/v1/intel/summary` `/v1/intel/feed/featured` `/v1/intel/report/md` (サマリー・注目・レポート) | cc:完了 |
| 68C.3 | `/v1/intel/category-map` — テキストからリスクカテゴリ分類 | cc:完了 |

| 68.T | `tests/test_sprint68.py` — 77 tests PASS (累計 3780) | cc:完了 |

---

## Sprint 69 詳細 (完了)

### Sprint 69: 時系列予測統合 TimesFM + マルチモデル — v0.72.0
> Google TimesFM アーキテクチャを軽量移植。LinearTrend / Mock / TimesFM の 3 モデルを統合。

| task-id | 説明 | 状態 |
|---------|------|------|
| 69.1 | `skills/time_series.py` — ForecastPoint / ForecastResult (データモデル) | cc:完了 |
| 69.2 | `skills/time_series.py` — LinearTrendForecaster (最小二乗法トレンド) | cc:完了 |
| 69.3 | `skills/time_series.py` — MockForecaster (テスト用決定論的予測) | cc:完了 |
| 69.4 | `skills/time_series.py` — TimesFMForecaster (patch embedding + self-attention) | cc:完了 |
| 69.5 | `skills/time_series.py` — TimesFMForecasterFactory (from_env/from_mock/linear) | cc:完了 |
| 69.6 | `skills/time_series.py` — CampaignForecaster / ForecastStore / ForecastReportEngine | cc:完了 |
| 69.7 | `serve/api.py` — `/v1/forecast/{id}` `/v1/forecast/{id}/all` `/v1/forecast/batch` | cc:完了 |
| 69.8 | `serve/api.py` — `/v1/forecast/{id}/history` `/v1/forecast/report/md/{fid}` `/v1/forecast/models` | cc:完了 |
| 69.T | `tests/test_sprint69.py` — 62 tests PASS (累計 3842) | cc:完了 |

#### TimesFM 対応表
| Google TimesFM | time_series.py | 役割 |
|----------------|----------------|------|
| patch embedding | `TimesFMForecaster._patch_embed` | 時系列分割→埋め込み |
| self-attention | `TimesFMForecaster._attention` | パターン抽出 |
| projection head | `TimesFMForecaster._project` | 予測値生成 |
| multimodel factory | `TimesFMForecasterFactory` | LinearTrend/Mock/TimesFM 切替 |

---

## Sprint 70 候補テーマ

| Option | テーマ | コアモジュール | 理由 |
|--------|--------|--------------|------|
| **A** | **予測×異常検知 統合アラート** | `skills/forecast_alert.py` | 予測値を AnomalyDetector に流し込み将来アラートを生成 |
| **B** | **レポート自動配信 Webhook** | `skills/report_dispatcher.py` | 定期レポートを Slack/webhook に POST、`/v1/report/dispatch` |
| **C** | **自然言語クエリ (NLQ) インターフェース** | `skills/nlq_agent.py` | 「先週の CTR は？」を API クエリに変換するエージェント |

---

## Sprint 71 詳細 (最新 / 完了)

### Sprint 71: 主要都市地図ビジュアライザ — v0.74.0
> 参照: [tokyo-danmenzu.pages.dev](https://tokyo-danmenzu.pages.dev/?view=3d&aux=1&lbl=1&lang=ja#12/35.67/139.75/0/50)
> 東京・大阪・名古屋・横浜・福岡の地下鉄路線 GeoJSON + 地質層データ + SVG 断面図レンダラー

| task-id | 説明 | 状態 |
|---------|------|------|
| 71A | `skills/city_map.py` — CityMapDataset: 5都市メトロデータ + 地質層プリセット + GeoJSON | cc:完了 |
| 71B | `skills/map_renderer.py` — SVGCrossSectionRenderer + CrossSectionEngine | cc:完了 |
| 71C | `serve/api.py` — `/v1/map/*` 7エンドポイント (cities/lines/stations/geology/geojson/cross-section/summary) | cc:完了 |
| 71T | `tests/test_sprint71.py` — 77 PASS (累計 4014) | cc:完了 |

## Sprint 72 詳細 (最新 / 完了)

### Sprint 72: 地図拡張 比較/編集/レポート — v0.75.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 72A | `skills/map_comparator.py` — MapComparator: 2都市断面比較SVG + DepthStats + geology diff | cc:完了 |
| 72B | `skills/map_editor.py` — MapEditor: Line/Station/Geology CRUD + 変更履歴 | cc:完了 |
| 72C | `skills/map_report.py` — MapReportEngine: 単都市/複数都市 Markdown レポート | cc:完了 |
| 72D | `serve/api.py` — `/v1/map/compare/*` `/v1/map-editor/*` `/v1/map/*/report/md` `/v1/map/report/compare` | cc:完了 |
| 72T | `tests/test_sprint72.py` — 71 PASS (累計 4085) | cc:完了 |

## Sprint 73 詳細 (最新 / 完了)

### Sprint 73: 地図アニメーション/経路探索/インポート — v0.76.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 73A | `skills/map_animator.py` — SMIL SVG アニメ (1960/1980/2000/2020 地質変化) | cc:完了 |
| 73B | `skills/route_finder.py` — Dijkstra 最短経路 + 乗換コスト | cc:完了 |
| 73C | `skills/map_importer.py` — CSV/GeoJSON 一括インポート | cc:完了 |
| 73D | `serve/api.py` — 9エンドポイント追加 | cc:完了 |
| 73T | `tests/test_sprint73.py` — 76 PASS (累計 4161) | cc:完了 |

## Sprint 74 詳細 (最新 / 完了)

### Sprint 74: 混雑シミュレーション/アクセシビリティ/地下水位 — v0.77.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 74A | `skills/crowd_simulator.py` — 時間帯別混雑 + CrowdDataset 16駅 | cc:完了 |
| 74B | `skills/accessibility.py` — 8機能100点スコアリング + AccessibilityDataset | cc:完了 |
| 74C | `skills/groundwater.py` — 4因子浸水リスク評価 + GroundwaterDataset | cc:完了 |
| 74D | `serve/api.py` — 11エンドポイント追加 | cc:完了 |
| 74T | `tests/test_sprint74.py` — 77 PASS (累計 4238) | cc:完了 |

## Sprint 75 候補テーマ

| Option | テーマ | コアモジュール | 理由 |
|--------|--------|--------------|------|
| **A** | **駅環境センサー統合** | `skills/env_sensor.py` | 気温・湿度・CO2・騒音レベルの駅内環境モニタリング |
| **B** | **乗り換え最適化** | `skills/transfer_optimizer.py` | 混雑・アクセシビリティ・所要時間を統合した最適乗換提案 |
| **C** | **都市インフラダッシュボード** | `skills/infra_dashboard.py` | 混雑・アクセシビリティ・地下水位を統合した都市インフラ可視化 |

---

## 技術的知見メモ

- `freqs_cis` は必ず `[:T]` スライス (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` に `.clamp(min=1e-6)` 必要 (float32 飽和防止)
- `store or Store()` は空ストア (len=0→falsy) を別インスタンスに差し替える → `is not None` チェックを使う
- `ConsensusEngine.score(texts)` が正しい API (build_consensus/compute_agreement は存在しない)
- テスト間レート制限干渉: `tests/conftest.py` の `reset_rate_limiter` fixture で全スイート実行時に自動リセット
- `VulnScanner.scan_source()` の focus_areas フィルタは title の部分一致 (harness の focus_area partition に対応)
