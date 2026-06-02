# Sprint 20〜25 アーカイブ — 「育つAI」P1〜P6
> アーカイブ日: 2026-06-02 | 詳細: `Plans.md` Sprint 26〜29 参照

---

## Sprint 20: 討議型集合知 — DebateOrchestrator & v0.23.0

> パターン P1 / ブランチ: `feature/sprint20-living-llmo` (PR #17 merged)

| task-id | 説明 | 状態 |
|---------|------|------|
| 20.1 | `debate.py` — `DebateConfig` / `DebateRound` / `DebateResult` dataclass | cc:完了 |
| 20.2 | `debate.py` — `DebateOrchestrator.run()` — Propose→Critique→Refine→Consensus 4フェーズ | cc:完了 |
| 20.3 | `debate.py` — `ConsensusEngine` — Jaccard類似度合意収束 (agreement_score / bi-gram) | cc:完了 |
| 20.4 | `serve/api.py` — `/v1/debate/run` | cc:完了 |
| 20.T | `tests/test_sprint20.py` — 59 tests PASS | cc:完了 |
| 20.V | PyPI v0.23.0 | cc:完了 |

---

## Sprint 21: KPI駆動自己改善 — KPIAgent & v0.24.0

> パターン P2

| task-id | 説明 | 状態 |
|---------|------|------|
| 21.1 | `kpi_agent.py` — `KPIDefinition` / `KPISnapshot` / `GapReport` / `Action` / `ActionPlan` | cc:完了 |
| 21.2 | `kpi_agent.py` — `KPIAgent.measure()` | cc:完了 |
| 21.3 | `kpi_agent.py` — `KPIAgent.analyze()` / `plan()` | cc:完了 |
| 21.4 | `kpi_agent.py` — `KPIAgent.execute()` | cc:完了 |
| 21.5 | `kpi_agent.py` — `KPIAgent.improve_loop()` | cc:完了 |
| 21.6 | `serve/api.py` — `/v1/kpi/measure` / `/improve` | cc:完了 |
| 21.T | `tests/test_sprint21.py` — 66 tests PASS | cc:完了 |
| 21.V | PyPI v0.24.0 | cc:完了 |

---

## Sprint 22: ボトルネック発見・解消 — ProfilerAgent & v0.25.0

> パターン P3

| task-id | 説明 | 状態 |
|---------|------|------|
| 22.1 | `profiler.py` — `StageMetrics` / `ProfileResult` / `BottleneckReport` / `AutoFixResult` | cc:完了 |
| 22.2 | `profiler.py` — `PipelineProfiler.run()` | cc:完了 |
| 22.3 | `profiler.py` — `BottleneckDetector.detect()` — IQR法 + 相対深刻度比較 | cc:完了 |
| 22.4 | `profiler.py` — `ProfilerAgent.auto_fix()` | cc:完了 |
| 22.5 | `serve/api.py` — `/v1/profile/run` / `/fix` / `/report` | cc:完了 |
| 22.T | `tests/test_sprint22.py` — 61 tests PASS | cc:完了 |
| 22.V | PyPI v0.25.0 | cc:完了 |

> **バグ修正**: `BottleneckDetector` がレイテンシを先に評価するため全ステージ高速時に誤判定 →
> latency/score 両方を検出し相対深刻度 (`score_rel > lat_rel`) で優先順を決定するよう修正済み

---

## Sprint 23: 外部要因適応 — ExternalSignalAgent & v0.26.0

> パターン P4

| task-id | 説明 | 状態 |
|---------|------|------|
| 23.1 | `external_signal.py` — `ExternalSignal` / `ImpactEstimate` / `CounterAction` / `ExternalSignalResult` | cc:完了 |
| 23.2 | `external_signal.py` — `SignalDetector.detect()` | cc:完了 |
| 23.3 | `external_signal.py` — `ImpactEstimator.estimate()` | cc:完了 |
| 23.4 | `external_signal.py` — `ExternalSignalAgent.run()` | cc:完了 |
| 23.5 | `serve/api.py` — `/v1/signal/detect` / `/counter` | cc:完了 |
| 23.T | `tests/test_sprint23.py` — 60 tests PASS | cc:完了 |
| 23.V | PyPI v0.26.0 | cc:完了 |

---

## Sprint 24: ミスから学習 — ErrorMemory & MistakeGuard & v0.27.0

> パターン P5

| task-id | 説明 | 状態 |
|---------|------|------|
| 24.1 | `error_memory.py` — `MistakeRecord` / `ErrorMemoryStore` / `MistakeClassifier` / `PreventionRule` / `RuleExtractor` / `GuardResult` / `MistakeGuard` | cc:完了 |
| 24.2〜5 | `ErrorMemoryStore` / `MistakeClassifier` / `RuleExtractor` / `MistakeGuard` 各メソッド実装 | cc:完了 |
| 24.6 | `serve/api.py` — `/v1/mistakes/record` / `/rules` / `/check` | cc:完了 |
| 24.T | `tests/test_sprint24.py` — 40 tests PASS | cc:完了 |
| 24.V | PyPI v0.27.0 | cc:完了 |

---

## Sprint 25: 継続的自己蒸留 — SelfDistillLoop & v0.28.0

> パターン P6

| task-id | 説明 | 状態 |
|---------|------|------|
| 25.1〜4 | `self_distill.py` — `DistillSample/Dataset/OutputFilter/SFTResult/SelfDistillConfig/RoundResult/Result/Collector/Loop` | cc:完了 |
| 25.5 | `serve/api.py` — `/v1/distill/run` / `/status` | cc:完了 |
| 25.T | `tests/test_sprint25.py` — 40 tests PASS | cc:完了 |
| 25.V | PyPI v0.28.0 | cc:完了 |

> **注意**: `SelfDistillLoop._simulate_sft()` は GPU 訓練なし。本番化時は `scripts/finetune.py` の `LoraTrainer` に差し替える。
