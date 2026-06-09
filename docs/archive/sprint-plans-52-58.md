# Sprint 52〜58 アーカイブ

> アーカイブ日: 2026-06-09 | Plans.md メンテナンスにより移動

---

## Sprint 58: LLMO ダッシュボード・CEP管理・競合分析 — v0.61.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 58.1 | `skills/llmo_dashboard.py` — CepCategory/CepEntry/CepStore (CEP管理) | cc:完了 |
| 58.2 | `skills/llmo_dashboard.py` — MentionSnapshot/CompetitorEntry/CompetitorAnalysis | cc:完了 |
| 58.3 | `skills/llmo_dashboard.py` — LlmoDashboard: 定点観測・時系列・競合比較 | cc:完了 |
| 58.4 | `skills/llmo_dashboard.py` — LlmoReportEngine: Markdown/JSON レポート生成 | cc:完了 |
| 58.5 | `serve/api.py` — `/v1/cep` CRUD / `/v1/llmo/snapshot` / `/v1/llmo/dashboard/{brand}` | cc:完了 |
| 58.6 | `serve/api.py` — `/v1/llmo/competitor` + `/v1/llmo/competitor/analyze` 競合分析 | cc:完了 |
| 58.T | `tests/test_sprint58.py` — 52 tests PASS | cc:完了 |
| 58.V | PyPI v0.61.0 | cc:完了 |

---

## Sprint 57: LLM 評価フレームワーク — v0.60.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 57.1 | `skills/evaluation.py` — EvalMetric/EvalSample/EvalResult/BenchmarkReport | cc:完了 |
| 57.2 | `skills/evaluation.py` — TextEvaluator: BLEU/ROUGE/長さ/多様性 | cc:完了 |
| 57.3 | `skills/evaluation.py` — AdEvaluator: LLMO6指標+CTR予測+ブランド適合 | cc:完了 |
| 57.4 | `skills/evaluation.py` — BenchmarkRunner + EvalLeaderboard | cc:完了 |
| 57.5 | `serve/api.py` — `/v1/eval/benchmark` `/v1/eval/benchmark/md` `/v1/eval/leaderboard` | cc:完了 |
| 57.T | `tests/test_sprint57.py` — 54 tests PASS | cc:完了 |
| 57.V | PyPI v0.60.0 | cc:完了 |

---

## Sprint 56: マルチプロバイダー LLM — v0.59.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 56.1 | `skills/llm_providers.py` — ProviderType/LLMRequest/LLMResponse/ProviderConfig | cc:完了 |
| 56.2 | `skills/llm_providers.py` — ClaudeProvider / OpenAIProvider / OpenMythosProvider | cc:完了 |
| 56.3 | `skills/llm_providers.py` — MultiProviderRouter: 優先順位・フォールバック | cc:完了 |
| 56.4 | `serve/api.py` — `/v1/llm/complete` `/v1/llm/providers` | cc:完了 |
| 56.T | `tests/test_sprint56.py` — 52 tests PASS | cc:完了 |
| 56.V | PyPI v0.59.0 | cc:完了 |

---

## Sprint 55: ストリーミング & SSE 応答 — v0.58.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 55.1 | `open_mythos/streaming.py` — StreamDelta/StreamChunk/StreamEvent/StreamSession | cc:完了 |
| 55.2 | `open_mythos/streaming.py` — StreamingRunner: ジェネレータ方式 yield / done 終端 | cc:完了 |
| 55.3 | `open_mythos/streaming.py` — StreamBuffer: チャンク蓄積・全文復元・エラーハンドリング | cc:完了 |
| 55.4 | `serve/api.py` — `/v1/chat/stream` (SSE) エンドポイント追加 | cc:完了 |
| 55.5 | `serve/api.py` — `/v1/threads/{id}/runs/stream` Assistants Run ストリーミング | cc:完了 |
| 55.T | `tests/test_sprint55.py` — 58 tests PASS | cc:完了 |
| 55.V | PyPI v0.58.0 | cc:完了 |

---

## Sprint 54: OpenAI Assistants API 統合 — v0.57.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 54.1 | `open_mythos/assistant.py` — AssistantTool / AssistantObject / Thread / MessageContent / Message / RunUsage / Run | cc:完了 |
| 54.2 | `open_mythos/assistant.py` — AssistantStore (CRUD: assistants/threads/messages/runs) | cc:完了 |
| 54.3 | `open_mythos/assistant.py` — AssistantRunner (LLM実行・応答追加) / get_default_store / reset_default_store | cc:完了 |
| 54.4 | `serve/api.py` — `/v1/assistants` (CRUD) `/v1/threads` (CRUD) `/v1/threads/{id}/messages` `/v1/threads/{id}/runs` | cc:完了 |
| 54.T | `tests/test_sprint54.py` — 65 tests PASS (累計 2862) | cc:完了 |
| 54.V | PyPI v0.57.0 | cc:完了 |

---

## Sprint 53: セキュリティ統合 — v0.56.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 53.1 | `open_mythos/skills/security.py` — PentestFinding / PentestReport / WebPentester | cc:完了 |
| 53.2 | `open_mythos/skills/security.py` — DependencyInfo / ForensicsReport / OSSForensics | cc:完了 |
| 53.3 | `serve/api.py` — `/v1/security/scan` `/v1/security/report/md` `/v1/security/oss/analyze` `/v1/security/oss/sbom` | cc:完了 |
| 53.T | `tests/test_sprint53.py` — 41 tests PASS (累計 2797) | cc:完了 |
| 53.V | PyPI v0.56.0 | cc:完了 |

---

## Sprint 52: DevOps・クラウド統合 — v0.55.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 52.1 | `open_mythos/skills/devops_cloud.py` — ModalFunctionConfig / ModalRunResult / ModalRunner | cc:完了 |
| 52.2 | `open_mythos/skills/devops_cloud.py` — ContainerInfo / BuildResult / DockerManager | cc:完了 |
| 52.3 | `open_mythos/skills/devops_cloud.py` — WatchRule / FileEvent / FileWatcher | cc:完了 |
| 52.4 | `open_mythos/skills/devops_cloud.py` — SLiMeConfig / SLiMeResult / SLiMeModel | cc:完了 |
| 52.5 | `serve/api.py` — `/v1/modal/*` `/v1/docker/*` `/v1/watch/config` `/v1/slime/fit` | cc:完了 |
| 52.T | `tests/test_sprint52.py` — 48 tests PASS (累計 2756) | cc:完了 |
| 52.V | PyPI v0.55.0 | cc:完了 |

---

## Sprint 51: データ・検索ツール統合 — v0.54.0

| task-id | 説明 | 状態 |
|---------|------|------|
| 51.1 | `open_mythos/skills/data_tools.py` — SearXNGResult / SearXNGSearcher | cc:完了 |
| 51.2 | `open_mythos/skills/data_tools.py` — DomainInfo / DomainIntelligence | cc:完了 |
| 51.3 | `open_mythos/skills/data_tools.py` — CurationRule / CurationResult / NemoCurator | cc:完了 |
| 51.4 | `open_mythos/skills/data_tools.py` — CodeSymbol / CodeWiki / CodeWikiGenerator | cc:完了 |
| 51.5 | `open_mythos/skills/data_tools.py` — APICallResult / APIDebugger | cc:完了 |
| 51.6 | `serve/api.py` — `/v1/search/searxng` `/v1/domain/lookup` `/v1/data/curate` `/v1/code/wiki` `/v1/api/rest` `/v1/api/graphql` | cc:完了 |
| 51.T | `tests/test_sprint51.py` — 53 tests PASS (累計 2708) | cc:完了 |
| 51.V | PyPI v0.54.0 | cc:完了 |
