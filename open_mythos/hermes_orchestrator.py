"""
Sprint 43 — HermesOrchestrator: Layer 2 Ultracode Mode Orchestrator

Claude Code の ultracode パターンを Hermes Agent として実装:

  ┌─────────┬──────────────────────────────────────────────────────┐
  │ Phase   │ 内容                                                 │
  ├─────────┼──────────────────────────────────────────────────────┤
  │ Plan    │ タスクを分解してサブタスクリストを生成               │
  │ Spawn   │ 複数エージェントスペックを同時生成                   │
  │ Parallel│ asyncio で並列実行 (call_fn による注入可能 HTTP)     │
  │ Verify  │ 各結果の品質チェック・スコアリング                   │
  │ Report  │ 統合レポートを生成して返す                           │
  └─────────┴──────────────────────────────────────────────────────┘

Layer 1: 各 MythosAgent (ローカル推論エンジン)
Layer 2: HermesOrchestrator (asyncio 並列オーケストレーション)

設計:
    SubTask             -- 分解されたサブタスク単位
    SubAgentSpec        -- スポーン設定 (endpoint + payload)
    HermesAgentResult   -- 単一エージェント実行結果
    VerificationResult  -- 品質検証結果
    HermesReport        -- 最終統合レポート
    TaskDecomposer      -- Phase 1: Plan (キーワード駆動分解)
    AgentSpawner        -- Phase 2: Spawn (スペック生成)
    ParallelExecutor    -- Phase 3: asyncio 並列実行
    ResultVerifier      -- Phase 4: 品質チェック
    ReportBuilder       -- Phase 5: 統合レポート生成
    HermesOrchestrator  -- メインコーディネーター

CallFn インターフェイス::

    def call_fn(endpoint: str, payload: Dict[str, Any]) -> str:
        ...

    # テスト時はモック:
    call_fn = lambda ep, pl: f"[result for {pl.get('prompt','')[:20]}]"

    # 本番: httpx や requests ベースの HTTP クライアント

使い方::

    from open_mythos.hermes_orchestrator import HermesOrchestrator

    hermes = HermesOrchestrator(max_subtasks=4, max_concurrent=3)
    report = hermes.run("SEO記事を書いて SERP 1位を目指す")

    print(f"サブタスク数: {len(report.subtasks)}")
    print(f"成功率: {report.success_rate:.0%}")
    print(f"最終出力: {report.final_output[:100]}")

async 版::

    import asyncio
    report = asyncio.run(hermes.run_async("SEO記事を書いて SERP 1位を目指す"))
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# CallFn type alias
# ---------------------------------------------------------------------------

#: (endpoint: str, payload: dict) -> str
#: テスト・本番両方で差し替え可能な HTTP 呼び出し抽象
CallFn = Callable[[str, Dict[str, Any]], str]


# ---------------------------------------------------------------------------
# Phase 1: Plan — SubTask & TaskDecomposer
# ---------------------------------------------------------------------------

@dataclass
class SubTask:
    """分解されたサブタスク単位"""
    task_id: str
    name: str
    description: str
    priority: int = 1                          # 1=高 2=中 3=低
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SubTask(id={self.task_id!r}, name={self.name!r}, priority={self.priority})"


# ドメイン → サブタスクテンプレート
_DOMAIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "seo": {
        "keywords": ["seo", "検索", "コンテンツ", "記事", "キーワード", "llmo", "serp", "検索順位"],
        "subtasks": [
            ("keyword_research",   "キーワードリサーチ",   "ターゲットキーワードの調査・選定・競合分析"),
            ("content_design",     "コンテンツ設計",       "見出し構成と E-E-A-T コンテンツ戦略の設計"),
            ("body_generation",    "本文生成",             "SEO 最適化された高品質な本文の生成"),
            ("final_proofread",    "最終校正",             "品質チェック・メタデータ整備・内部リンク確認"),
        ],
    },
    "analysis": {
        "keywords": ["分析", "調査", "データ", "レポート", "analysis", "research", "insight", "リサーチ"],
        "subtasks": [
            ("data_collection",    "データ収集",     "必要なデータ・情報の収集・整理"),
            ("analysis_exec",      "分析実行",       "収集データの定量・定性分析"),
            ("report_generation",  "レポート生成",   "分析結果の構造化レポート化"),
        ],
    },
    "creative": {
        "keywords": ["広告", "コピー", "クリエイティブ", "creative", "ad", "copy", "デザイン", "キャッチ"],
        "subtasks": [
            ("concept_design",     "コンセプト設計",   "クリエイティブコンセプトとターゲット訴求の設計"),
            ("copy_generation",    "コピー生成",       "複数パターンの広告コピー案生成"),
            ("evaluation",         "評価・選定",       "各案の品質評価と最適案の選定"),
        ],
    },
    "plan": {
        "keywords": ["計画", "タスク", "plan", "workflow", "実装", "開発", "設計", "フロー", "ステップ"],
        "subtasks": [
            ("requirements",       "要件分析",       "タスク要件の詳細分析と受け入れ条件の定義"),
            ("decomposition",      "分解設計",       "実行可能なサブタスクへの詳細分解"),
            ("execution",          "実行",           "各サブタスクの実行と中間結果の記録"),
            ("integration",        "統合・検証",     "結果の統合と品質検証・DoD 確認"),
        ],
    },
    "review": {
        "keywords": ["レビュー", "評価", "review", "assess", "check", "確認", "検証", "品質"],
        "subtasks": [
            ("criteria_setup",     "評価基準設定",   "レビュー観点と採点基準の設定"),
            ("review_exec",        "レビュー実行",   "複数観点からの詳細レビュー実行"),
            ("feedback_report",    "フィードバック", "改善提案とアクションアイテムのレポート"),
        ],
    },
}

_FALLBACK_DOMAIN = "general"
_GENERAL_SUBTASKS = [
    ("info_gathering",     "情報収集",   "タスクに必要な情報・コンテキストの収集"),
    ("main_execution",     "処理実行",   "メインタスクの処理・生成実行"),
    ("output_integration", "結果統合",   "処理結果の統合・構造化・整理"),
]


class TaskDecomposer:
    """
    Phase 1 — Plan: ゴール文字列をサブタスクリストに分解する。

    キーワードマッチングでドメインを判定し、テンプレートからサブタスクを生成。
    複数ドメインに該当する場合はスコアの高いドメインを優先。
    """

    def __init__(self, templates: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._templates = templates or _DOMAIN_TEMPLATES

    def detect_domain(self, goal: str, context: Dict[str, Any]) -> str:
        """ゴール文字列からドメインを判定する"""
        text = (goal + " " + str(context)).lower()
        scores: Dict[str, int] = {}
        for domain, cfg in self._templates.items():
            score = sum(1 for kw in cfg["keywords"] if kw in text)
            if score > 0:
                scores[domain] = score
        if not scores:
            return _FALLBACK_DOMAIN
        return max(scores, key=lambda d: scores[d])

    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_subtasks: int = 4,
    ) -> List[SubTask]:
        """
        goal と context から SubTask リストを生成する。

        Args:
            goal: ユーザーゴール文字列
            context: 追加コンテキスト
            max_subtasks: 最大サブタスク数 (1–8)

        Returns:
            依存関係付き SubTask のリスト
        """
        context = context or {}
        max_subtasks = max(1, min(max_subtasks, 8))
        domain = self.detect_domain(goal, context)

        if domain == _FALLBACK_DOMAIN:
            templates = _GENERAL_SUBTASKS[:max_subtasks]
        else:
            templates = self._templates[domain]["subtasks"][:max_subtasks]

        subtasks: List[SubTask] = []
        prev_id: Optional[str] = None

        for i, tpl in enumerate(templates):
            tid, name, desc = tpl[0], tpl[1], tpl[2]
            task_id = f"{tid}-{uuid.uuid4().hex[:6]}"
            depends = [prev_id] if prev_id and i > 0 else []
            st = SubTask(
                task_id=task_id,
                name=name,
                description=desc,
                priority=i + 1,
                depends_on=depends,
                metadata={"domain": domain, "goal_fragment": goal[:80]},
            )
            subtasks.append(st)
            prev_id = task_id

        return subtasks


# ---------------------------------------------------------------------------
# Phase 2: Spawn — SubAgentSpec & AgentSpawner
# ---------------------------------------------------------------------------

@dataclass
class SubAgentSpec:
    """
    1 エージェントのスポーン仕様。

    endpoint は Layer 1 API のパス (e.g. "/generate", "/v1/chat/completions")。
    payload は endpoint に渡す JSON ボディ。
    """
    agent_id: str
    task: SubTask
    endpoint: str
    payload: Dict[str, Any]
    timeout_s: float = 30.0

    def __repr__(self) -> str:
        return (
            f"SubAgentSpec(agent_id={self.agent_id!r}, "
            f"task={self.task.name!r}, endpoint={self.endpoint!r})"
        )


class AgentSpawner:
    """
    Phase 2 — Spawn: SubTask リストから SubAgentSpec リストを生成する。

    各タスクに対して適切なエンドポイントとペイロードを自動設定する。
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")

    def spawn(
        self,
        subtasks: List[SubTask],
        max_new_tokens: int = 256,
    ) -> List[SubAgentSpec]:
        """
        SubTask リストから SubAgentSpec リストを生成する。

        Args:
            subtasks: Plan フェーズで生成されたサブタスクリスト
            max_new_tokens: 生成トークン数上限

        Returns:
            SubAgentSpec のリスト (subtasks と 1:1 対応)
        """
        specs: List[SubAgentSpec] = []
        for st in subtasks:
            agent_id = f"agent-{st.task_id}"
            prompt = (
                f"タスク: {st.name}\n"
                f"説明: {st.description}\n"
                f"目標: {st.metadata.get('goal_fragment', '')}\n\n"
                "上記タスクを実行し、詳細な結果を出力してください。"
            )
            spec = SubAgentSpec(
                agent_id=agent_id,
                task=st,
                endpoint=f"{self._base_url}/generate",
                payload={
                    "prompt": prompt,
                    "task": "general",
                    "max_new_tokens": max_new_tokens,
                },
                timeout_s=30.0,
            )
            specs.append(spec)
        return specs


# ---------------------------------------------------------------------------
# Phase 3: Parallel Execute — HermesAgentResult & ParallelExecutor
# ---------------------------------------------------------------------------

@dataclass
class HermesAgentResult:
    """単一エージェント実行結果"""
    agent_id: str
    task_id: str
    task_name: str
    output: str
    latency_ms: float
    success: bool
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.success and bool(self.output.strip())

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERR:{self.error}"
        return (
            f"HermesAgentResult(agent={self.agent_id!r}, "
            f"task={self.task_name!r}, status={status}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


async def _execute_single(
    spec: SubAgentSpec,
    call_fn: CallFn,
) -> HermesAgentResult:
    """単一エージェントを asyncio で実行する内部ヘルパー"""
    t0 = time.perf_counter()
    try:
        # call_fn は同期関数を想定 → to_thread でノンブロッキング化
        output: str = await asyncio.to_thread(call_fn, spec.endpoint, spec.payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        return HermesAgentResult(
            agent_id=spec.agent_id,
            task_id=spec.task.task_id,
            task_name=spec.task.name,
            output=str(output),
            latency_ms=latency_ms,
            success=True,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        return HermesAgentResult(
            agent_id=spec.agent_id,
            task_id=spec.task.task_id,
            task_name=spec.task.name,
            output="",
            latency_ms=latency_ms,
            success=False,
            error=str(exc),
        )


class ParallelExecutor:
    """
    Phase 3 — Parallel Execute: asyncio で複数エージェントを並列実行する。

    ``max_concurrent`` を超えないように asyncio.Semaphore でスロットリング。
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        self._max_concurrent = max(1, max_concurrent)

    async def execute(
        self,
        specs: List[SubAgentSpec],
        call_fn: CallFn,
    ) -> List[HermesAgentResult]:
        """
        specs を並列実行して HermesAgentResult リストを返す。

        実行順は asyncio.gather() による並列。
        結果は specs と同じ順序に並び替えて返す。

        Args:
            specs: AgentSpawner が生成した SubAgentSpec リスト
            call_fn: (endpoint, payload) -> str の注入可能な呼び出し関数

        Returns:
            HermesAgentResult のリスト (specs と同じ順序)
        """
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _guarded(spec: SubAgentSpec) -> HermesAgentResult:
            async with sem:
                return await _execute_single(spec, call_fn)

        return list(await asyncio.gather(*[_guarded(s) for s in specs]))


# ---------------------------------------------------------------------------
# Phase 4: Verify — VerificationResult & ResultVerifier
# ---------------------------------------------------------------------------

_ERROR_MARKERS = {"error", "exception", "traceback", "エラー", "失敗", "[error]"}
_QUALITY_MIN_LEN = 20   # 最低文字数
_QUALITY_GOOD_LEN = 100 # 品質スコア満点の目安


@dataclass
class VerificationResult:
    """単一エージェント結果の品質検証結果"""
    agent_id: str
    task_id: str
    task_name: str
    passed: bool
    score: float           # 0.0 〜 1.0
    issues: List[str]
    verified_output: str   # 検証後の出力 (クリーンアップ済み)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"VerificationResult(agent={self.agent_id!r}, "
            f"task={self.task_name!r}, {status}, score={self.score:.2f})"
        )


class ResultVerifier:
    """
    Phase 4 — Verify: 各エージェント結果の品質をチェックする。

    チェック項目:
    1. 実行成功フラグ
    2. 出力の非空確認
    3. 最小長チェック
    4. エラーマーカー検出
    5. 長さベーススコアリング
    """

    def __init__(
        self,
        min_len: int = _QUALITY_MIN_LEN,
        good_len: int = _QUALITY_GOOD_LEN,
    ) -> None:
        self._min_len = min_len
        self._good_len = good_len

    def verify_one(self, result: HermesAgentResult) -> VerificationResult:
        """単一 HermesAgentResult を検証する"""
        issues: List[str] = []
        score = 0.0

        # 1. 実行成功チェック
        if not result.success:
            issues.append(f"実行エラー: {result.error}")
            return VerificationResult(
                agent_id=result.agent_id,
                task_id=result.task_id,
                task_name=result.task_name,
                passed=False,
                score=0.0,
                issues=issues,
                verified_output="",
            )

        output = result.output.strip()

        # 2. 空出力チェック
        if not output:
            issues.append("出力が空")
            return VerificationResult(
                agent_id=result.agent_id,
                task_id=result.task_id,
                task_name=result.task_name,
                passed=False,
                score=0.0,
                issues=issues,
                verified_output="",
            )

        # 3. 最小長チェック
        if len(output) < self._min_len:
            issues.append(f"出力が短すぎる ({len(output)} < {self._min_len} chars)")

        # 4. エラーマーカー検出
        output_lower = output.lower()
        found_markers = [m for m in _ERROR_MARKERS if m in output_lower]
        if found_markers:
            issues.append(f"エラーマーカー検出: {found_markers}")

        # 5. 長さベーススコアリング
        length_ratio = min(len(output) / self._good_len, 1.0)
        marker_penalty = 0.3 * len(found_markers)
        score = max(0.0, min(1.0, length_ratio - marker_penalty))

        # 最小長未満は最大 0.4
        if len(output) < self._min_len:
            score = min(score, 0.4)

        passed = score >= 0.3 and not result.error

        return VerificationResult(
            agent_id=result.agent_id,
            task_id=result.task_id,
            task_name=result.task_name,
            passed=passed,
            score=score,
            issues=issues,
            verified_output=output,
        )

    def verify(self, results: List[HermesAgentResult]) -> List[VerificationResult]:
        """HermesAgentResult リスト全体を検証する"""
        return [self.verify_one(r) for r in results]


# ---------------------------------------------------------------------------
# Phase 5: Report — HermesReport & ReportBuilder
# ---------------------------------------------------------------------------

@dataclass
class HermesReport:
    """
    HermesOrchestrator.run() の最終統合レポート

    全フェーズの情報と最終合成出力を含む。
    """
    run_id: str
    goal: str
    subtasks: List[SubTask]
    agent_results: List[HermesAgentResult]
    verification_results: List[VerificationResult]
    final_output: str
    overall_score: float       # 0.0 〜 1.0
    total_latency_ms: float
    success_rate: float        # 成功したエージェント数 / 全エージェント数
    phase_timings: Dict[str, float]   # フェーズ名 → ms

    def summary(self) -> str:
        n = len(self.subtasks)
        ok = sum(1 for r in self.verification_results if r.passed)
        return (
            f"[HermesReport] run={self.run_id} | goal={self.goal[:40]!r} | "
            f"subtasks={n} | passed={ok}/{n} | "
            f"score={self.overall_score:.2f} | "
            f"latency={self.total_latency_ms:.0f}ms"
        )

    def __repr__(self) -> str:
        return self.summary()


class ReportBuilder:
    """Phase 5 — Report: 全フェーズ結果を統合した HermesReport を生成する"""

    def build(
        self,
        run_id: str,
        goal: str,
        subtasks: List[SubTask],
        agent_results: List[HermesAgentResult],
        verification_results: List[VerificationResult],
        phase_timings: Dict[str, float],
    ) -> HermesReport:
        """
        全フェーズの結果を受け取り HermesReport を生成する。

        final_output は検証済み出力を順番に結合したもの。
        overall_score は各 VerificationResult のスコアの平均。
        """
        total_latency_ms = sum(phase_timings.values())

        # 成功率
        n = len(agent_results)
        success_count = sum(1 for r in agent_results if r.success)
        success_rate = success_count / n if n > 0 else 0.0

        # 総合スコア
        scores = [v.score for v in verification_results]
        overall_score = sum(scores) / len(scores) if scores else 0.0

        # 最終出力: 検証通過したものを順序付けて統合
        passed_sections: List[str] = []
        for vr in verification_results:
            if vr.passed and vr.verified_output:
                passed_sections.append(
                    f"### {vr.task_name}\n{vr.verified_output}"
                )
        if not passed_sections:
            # 全失敗時: 最高スコアの出力を使う
            best = max(verification_results, key=lambda v: v.score, default=None)
            if best and best.verified_output:
                passed_sections.append(best.verified_output)
            else:
                passed_sections.append(f"[HermesOrchestrator] ゴール「{goal}」の実行が完了しましたが、検証通過の出力がありません。")

        final_output = "\n\n".join(passed_sections)

        return HermesReport(
            run_id=run_id,
            goal=goal,
            subtasks=subtasks,
            agent_results=agent_results,
            verification_results=verification_results,
            final_output=final_output,
            overall_score=overall_score,
            total_latency_ms=total_latency_ms,
            success_rate=success_rate,
            phase_timings=phase_timings,
        )


# ---------------------------------------------------------------------------
# Default call_fn — ローカルシミュレーション
# ---------------------------------------------------------------------------

def _local_call_fn(endpoint: str, payload: Dict[str, Any]) -> str:
    """
    デフォルトの call_fn: HTTP を使わずローカルでシミュレーションする。

    本番環境では httpx や requests ベースの実装に置き換える:

        import httpx
        def http_call_fn(endpoint, payload):
            r = httpx.post(endpoint, json=payload, timeout=30)
            r.raise_for_status()
            return r.json().get("text", "")
    """
    prompt = payload.get("prompt", payload.get("task_input", "入力なし"))
    # モデルが無い環境でも動作するシミュレーション出力
    short_prompt = prompt[:60].replace("\n", " ")
    return (
        f"[HermesAgent] タスク完了。\n"
        f"プロンプト: {short_prompt}\n"
        f"分析結果: このタスクは正常に処理されました。\n"
        f"詳細: 要件を満たす高品質な出力を生成しました。\n"
        f"品質スコア: 0.85 / 1.0"
    )


# ---------------------------------------------------------------------------
# HermesOrchestrator — メインコーディネーター
# ---------------------------------------------------------------------------

class HermesOrchestrator:
    """
    Hermes Agent Layer 2 — Ultracode Mode Orchestrator

    Plan → Spawn → Parallel Execute → Verify → Report の 5 フェーズを
    協調して実行する最上位オーケストレーター。

    Args:
        base_url: Layer 1 API のベース URL (デフォルト: localhost:8000)
        max_subtasks: Plan フェーズで生成する最大サブタスク数 (1–8)
        max_concurrent: 並列実行する最大エージェント数 (1–16)
        max_new_tokens: 各エージェントへの生成トークン数上限
        call_fn: HTTP 呼び出し抽象 (テスト時はモックに差し替え可能)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_subtasks: int = 4,
        max_concurrent: int = 3,
        max_new_tokens: int = 256,
        call_fn: Optional[CallFn] = None,
    ) -> None:
        self._base_url     = base_url
        self._max_subtasks = max(1, min(max_subtasks, 8))
        self._max_concurrent = max(1, min(max_concurrent, 16))
        self._max_new_tokens = max_new_tokens
        self._call_fn      = call_fn or _local_call_fn

        # コンポーネント初期化
        self._decomposer  = TaskDecomposer()
        self._spawner     = AgentSpawner(base_url=base_url)
        self._executor    = ParallelExecutor(max_concurrent=max_concurrent)
        self._verifier    = ResultVerifier()
        self._reporter    = ReportBuilder()

    # ------------------------------------------------------------------
    # Phase 1: Plan
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SubTask]:
        """
        ゴールをサブタスクリストに分解する。

        Args:
            goal: ユーザーゴール文字列
            context: 付加コンテキスト情報

        Returns:
            SubTask のリスト
        """
        return self._decomposer.decompose(
            goal=goal,
            context=context,
            max_subtasks=self._max_subtasks,
        )

    # ------------------------------------------------------------------
    # Phase 2: Spawn
    # ------------------------------------------------------------------

    def spawn(self, subtasks: List[SubTask]) -> List[SubAgentSpec]:
        """
        SubTask リストからエージェントスペックを生成する。

        Args:
            subtasks: plan() が返した SubTask リスト

        Returns:
            SubAgentSpec のリスト
        """
        return self._spawner.spawn(
            subtasks=subtasks,
            max_new_tokens=self._max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Phase 3: Parallel Execute (async)
    # ------------------------------------------------------------------

    async def parallel_execute(
        self,
        specs: List[SubAgentSpec],
    ) -> List[HermesAgentResult]:
        """
        SubAgentSpec リストを asyncio で並列実行する。

        Args:
            specs: spawn() が返した SubAgentSpec リスト

        Returns:
            HermesAgentResult のリスト (specs と同じ順序)
        """
        return await self._executor.execute(specs, self._call_fn)

    # ------------------------------------------------------------------
    # Phase 4: Verify
    # ------------------------------------------------------------------

    def verify(
        self,
        results: List[HermesAgentResult],
    ) -> List[VerificationResult]:
        """
        エージェント実行結果を品質チェックする。

        Args:
            results: parallel_execute() が返した HermesAgentResult リスト

        Returns:
            VerificationResult のリスト
        """
        return self._verifier.verify(results)

    # ------------------------------------------------------------------
    # Phase 5: Report
    # ------------------------------------------------------------------

    def report(
        self,
        run_id: str,
        goal: str,
        subtasks: List[SubTask],
        agent_results: List[HermesAgentResult],
        verification_results: List[VerificationResult],
        phase_timings: Dict[str, float],
    ) -> HermesReport:
        """
        全フェーズ結果を統合した HermesReport を生成する。

        Returns:
            HermesReport
        """
        return self._reporter.build(
            run_id=run_id,
            goal=goal,
            subtasks=subtasks,
            agent_results=agent_results,
            verification_results=verification_results,
            phase_timings=phase_timings,
        )

    # ------------------------------------------------------------------
    # Full pipeline: async
    # ------------------------------------------------------------------

    async def run_async(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> HermesReport:
        """
        Plan → Spawn → Parallel Execute → Verify → Report を非同期で実行する。

        Args:
            goal: ユーザーゴール文字列
            context: 付加コンテキスト情報

        Returns:
            HermesReport
        """
        run_id = uuid.uuid4().hex[:12]
        timings: Dict[str, float] = {}

        # Phase 1: Plan
        t0 = time.perf_counter()
        subtasks = self.plan(goal, context)
        timings["plan_ms"] = (time.perf_counter() - t0) * 1000

        # Phase 2: Spawn
        t0 = time.perf_counter()
        specs = self.spawn(subtasks)
        timings["spawn_ms"] = (time.perf_counter() - t0) * 1000

        # Phase 3: Parallel Execute
        t0 = time.perf_counter()
        agent_results = await self.parallel_execute(specs)
        timings["execute_ms"] = (time.perf_counter() - t0) * 1000

        # Phase 4: Verify
        t0 = time.perf_counter()
        verification_results = self.verify(agent_results)
        timings["verify_ms"] = (time.perf_counter() - t0) * 1000

        # Phase 5: Report
        t0 = time.perf_counter()
        hermes_report = self.report(
            run_id=run_id,
            goal=goal,
            subtasks=subtasks,
            agent_results=agent_results,
            verification_results=verification_results,
            phase_timings=timings,
        )
        timings["report_ms"] = (time.perf_counter() - t0) * 1000

        return hermes_report

    # ------------------------------------------------------------------
    # Full pipeline: sync wrapper
    # ------------------------------------------------------------------

    def run(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> HermesReport:
        """
        run_async() の同期ラッパー。

        既存の event loop が走っている場合は ThreadPoolExecutor を使う
        (Jupyter / FastAPI 非同期コンテキストで安全に呼び出せる)。

        Args:
            goal: ユーザーゴール文字列
            context: 付加コンテキスト情報

        Returns:
            HermesReport
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 既存の event loop 内から呼ばれた場合: 別スレッドで asyncio.run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.run_async(goal, context))
                return future.result()
        else:
            return asyncio.run(self.run_async(goal, context))
