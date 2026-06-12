# OpenMythos — Sprint Plans
> 最終更新: 2026-06-12 (Sprint 61 完了) | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 61 完了: Claude Fable 5 / Mythos 5 × N-day→N-hour サイバー防衛 → v0.64.0
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
| 60 | **広告キャンペーン管理 + A/Bテスト基盤** | `skills/campaign_manager.py` `skills/ab_test.py` | 3238 | v0.63 |
| 61 | **Claude Fable 5 / Mythos 5 × N-hour サイバー防衛** | `skills/cyber_defense.py` `skills/llm_providers.py` | 3352 | v0.64 |

> **累計テスト数**: ~3352 PASS (Sprint 61: +114)

---

## Sprint 61 詳細 (最新 / 完了)

### Sprint 61: Claude Fable 5 / Mythos 5 × N-day→N-hour サイバー防衛 — v0.64.0
> 参照: `external/defending-code-harness/` + Anthropic Security Research (https://red.anthropic.com/2026/n-days/)
> 「N-day という表現は危険なほど誤解を招く。現実は N-hour だ。」— Anthropic Security

| task-id | 説明 | 状態 |
|---------|------|------|
| 61.L | `skills/llm_providers.py` — ClaudeModelTier(HAIKU_5/FABLE_5/MYTHOS_5) + HFInferenceProvider(Lily-Cyber-7B) + list_model_tiers() | cc:完了 |
| 61.C | `skills/cyber_defense.py` — ThreatLevel/IndicatorType/IncidentStatus/ThreatIndicator/ThreatIntelReport/Incident/ForensicsArtifact | cc:完了 |
| 61.C | `skills/cyber_defense.py` — IncidentStore/ThreatIntelAnalyzer/IncidentResponder/ForensicsAI/CyberDefenseOrchestrator(マルチエージェント) | cc:完了 |
| 61.S | `skills/security.py` — AISecurityEnhancer (Claude Fable 5: エグゼクティブサマリー/優先度付けリメディエーション/修正アドバイス) | cc:完了 |
| 61.V | `skills/vuln_scanner.py` — PatchUrgency(N-hour SLA)/NDayRecord/NDayVulnTracker/HFCodeBERTClassifier/MythosVulnAnalyzer(Claude Mythos 5) | cc:完了 |
| 61.A | `serve/api.py` — GET /v1/model/tiers + POST /v1/cyber/threat + /v1/cyber/incident CRUD + /v1/cyber/forensics + /v1/cyber/defend | cc:完了 |
| 61.A | `serve/api.py` — POST /v1/cyber/classify/code + POST /v1/cyber/nday/register + PATCH /v1/cyber/nday/{id}/patch + GET /v1/cyber/nday/* | cc:完了 |
| 61.T | `tests/test_sprint61.py` — 114 tests PASS (累計 3352) | cc:完了 |
| 61.V | PyPI v0.64.0 | cc:完了 |

### モデル対応表
| OpenMythos 名称 | Anthropic モデル | 用途 |
|----------------|----------------|------|
| Claude Fable 5 | `claude-sonnet-4-5` | 一般向け・ペンテストサマリー (AISecurityEnhancer) |
| Claude Mythos 5 | `claude-opus-4` | サイバー防衛特化・N-hour パッチ分析 (MythosVulnAnalyzer) |

### HuggingFace 流用モデル (Apache-2.0)
| モデル | 用途 |
|-------|------|
| `segolilylabs/Lily-Cybersecurity-7B-v0.2` | SOC Q&A / 脅威アドバイス (HFInferenceProvider) |
| `RayenLLM/Vulnerability_Detection_Using_CodeBERT` | コード脆弱性分類 (HFCodeBERTClassifier) |

---

## Sprint 60 詳細 (完了)

### Sprint 60: 広告キャンペーン管理 + A/Bテスト基盤 — v0.63.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 60.A | `skills/campaign_manager.py` — CampaignStatus/AdFormat/CopyRequest/AdCopy/CampaignMetrics/CampaignEntry | cc:完了 |
| 60.A | `skills/campaign_manager.py` — CampaignStore/CopyGenerator/CampaignEvaluator/CampaignWorkflow/CampaignReportEngine | cc:完了 |
| 60.A | `serve/api.py` — `/v1/campaign` CRUD + `/v1/campaign/{id}/run` ワークフロー + メトリクス/レポート | cc:完了 |
| 60.C | `skills/ab_test.py` — VariantStatus/TestStatus/StatMethod/Variant/ABTest/ABTestStore | cc:完了 |
| 60.C | `skills/ab_test.py` — StatEngine(χ²/Welch-t/Bayes)/ABTestAnalyzer/ABTestRunner/ABReportEngine | cc:完了 |
| 60.C | `serve/api.py` — `/v1/ab/test` CRUD + start/record/analyze/stop/report | cc:完了 |
| 60.T | `tests/test_sprint60.py` — 83 tests PASS (累計 3238) | cc:完了 |
| 60.V | PyPI v0.63.0 | cc:完了 |

---

## Sprint 59 詳細 (完了)

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

## Sprint 60 候補テーマ

| Option | テーマ | コアモジュール | 理由 |
|--------|--------|--------------|------|
| **A** | **広告キャンペーン管理** | `skills/campaign_manager.py` | CEP→コピー→評価の全フローをワークフロー化 |
| B | **日本語対応トークナイザー** | `open_mythos/tokenizer_ja.py` | GPT-2英語依存を解消。日本語広告コピー学習の前提 |
| C | **A/Bテスト基盤** | `skills/ab_test.py` | 複数コピー案の効果測定フレームワーク |

---

## 技術的知見メモ

- `freqs_cis` は必ず `[:T]` スライス (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` に `.clamp(min=1e-6)` 必要 (float32 飽和防止)
- `store or Store()` は空ストア (len=0→falsy) を別インスタンスに差し替える → `is not None` チェックを使う
- `ConsensusEngine.score(texts)` が正しい API (build_consensus/compute_agreement は存在しない)
- テスト間レート制限干渉: `tests/conftest.py` の `reset_rate_limiter` fixture で全スイート実行時に自動リセット
- `VulnScanner.scan_source()` の focus_areas フィルタは title の部分一致 (harness の focus_area partition に対応)
