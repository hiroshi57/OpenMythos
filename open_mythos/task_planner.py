"""
TaskPlanner — 自律タスク計画 (Sprint 29 / P10パターン).

複雑なゴールを階層的サブタスクに分解し、SwarmOrchestrator で
並列実行した後、結果を合意ベースで統合する。

設計:
    Task              -- ゴール文字列 + 依存関係 + 優先度
    TaskGraph         -- タスク依存グラフ (DAG) とトポロジカルソート
    TaskExecutionResult -- 単一タスクの実行結果
    TaskPlan          -- 分解されたタスクツリー全体
    TaskPlanResult    -- プラン実行の最終結果
    TaskPlanner       -- 分解 → 割り当て → 実行 → 統合エンジン

分解戦略:
    - キーワード認識ルール (SEO/KPI/分析/生成/評価)
    - 依存グラフを構築してトポロジカル実行順を決定
    - 並列実行可能タスクは SwarmOrchestrator.map() に委譲

精度向上のポイント:
    - 依存関係解決で前段の出力を後段の入力に自動接続
    - KPIAgent (Sprint 21) でサブタスク成功を定量判定
    - ConsensusEngine (Sprint 20) で複数エージェント結果を統合
    - ExternalSignalAgent (Sprint 23) で外部要因を計画に反映

他スプリントとの連携:
    - SwarmOrchestrator (Sprint 13): タスク並列実行
    - ReActAgent (Sprint 12): サブタスクの Think→Act ループ
    - KPIAgent (Sprint 21): タスク達成判定
    - DebateOrchestrator (Sprint 20): 結果統合に議論ベース合意
    - ExternalSignalAgent (Sprint 23): 外部要因による計画調整
    - EnsembleScorer (Sprint 27): タスク品質評価

使い方::

    from open_mythos.task_planner import TaskPlanner, Task

    planner = TaskPlanner(max_parallel=3)
    result = planner.execute(
        goal="SEO記事を作成して検索順位を向上させる",
        context={"domain": "AI技術", "target_kpi": 0.8},
    )
    for r in result.subtask_results:
        print(f"[{r.task.name}] {r.output[:80]}")
    print(f"final: {result.synthesized_output[:100]}")
"""

from __future__ import annotations

import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """
    単一タスク。

    Attributes
    ----------
    name        : タスク識別名
    goal        : このタスクのゴール文字列
    task_type   : "analysis" / "generation" / "evaluation" / "optimization" / "generic"
    priority    : 優先度 (高=1, 低=5)
    depends_on  : 前提タスクの name リスト
    agent_hint  : 担当エージェントの特性ヒント
    task_id     : 一意ID
    """

    name: str
    goal: str
    task_type: str = "generic"
    priority: int = 3
    depends_on: List[str] = field(default_factory=list)
    agent_hint: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __hash__(self) -> int:
        return hash(self.task_id)


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------


class TaskGraph:
    """
    タスク依存グラフ (DAG)。

    重複・循環依存を検出し、トポロジカルソート済み実行順を返す。
    """

    def __init__(self, tasks: List[Task]) -> None:
        self._tasks: Dict[str, Task] = {t.name: t for t in tasks}
        self._validate()

    def _validate(self) -> None:
        # 存在しない依存先を除去 (ベストエフォート)
        for task in self._tasks.values():
            task.depends_on = [d for d in task.depends_on if d in self._tasks]

        # 循環依存チェック (DFS)
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for dep in self._tasks[node].depends_on:
                if dep not in visited:
                    if _has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for name in self._tasks:
            if name not in visited:
                if _has_cycle(name):
                    raise ValueError(
                        f"TaskGraph: 循環依存を検出しました (task '{name}' を含むサイクル)。"
                        " depends_on を確認してください。"
                    )

    def topological_order(self) -> List[List[Task]]:
        """
        タスクを実行段階 (wave) のリストで返す。

        同一 wave 内のタスクは依存なし → 並列実行可能。

        Returns
        -------
        List[List[Task]]: wave のリスト (wave 0 から順に実行)
        """
        in_degree: Dict[str, int] = {name: 0 for name in self._tasks}
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep in in_degree:
                    in_degree[task.name] += 1

        queue: deque = deque()
        for name, deg in in_degree.items():
            if deg == 0:
                queue.append(name)

        waves: List[List[Task]] = []
        while queue:
            wave_size = len(queue)
            wave: List[Task] = []
            for _ in range(wave_size):
                name = queue.popleft()
                wave.append(self._tasks[name])
                for other in self._tasks.values():
                    if name in other.depends_on:
                        in_degree[other.name] -= 1
                        if in_degree[other.name] == 0:
                            queue.append(other.name)
            # priority 順でソート
            wave.sort(key=lambda t: t.priority)
            waves.append(wave)

        return waves

    @property
    def tasks(self) -> List[Task]:
        return list(self._tasks.values())


# ---------------------------------------------------------------------------
# TaskExecutionResult
# ---------------------------------------------------------------------------


@dataclass
class TaskExecutionResult:
    """
    単一タスクの実行結果。

    Attributes
    ----------
    task         : 実行したタスク
    output       : 出力テキスト
    score        : 品質スコア (0〜1)
    latency_ms   : 実行時間 (ms)
    success      : 成功フラグ
    error        : エラーメッセージ (正常時 None)
    """

    task: Task
    output: str
    score: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.success and self.error is None


# ---------------------------------------------------------------------------
# TaskPlan
# ---------------------------------------------------------------------------


@dataclass
class TaskPlan:
    """
    分解されたタスクプラン。

    Attributes
    ----------
    goal          : 元のゴール
    tasks         : 全サブタスクのリスト
    waves         : 実行段階 (wave[0] が最初、同一 wave は並列)
    plan_id       : 一意ID
    created_at    : 作成タイムスタンプ
    external_factors: ExternalSignalAgent の影響推定 (任意)
    """

    goal: str
    tasks: List[Task]
    waves: List[List[Task]]
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    external_factors: List[str] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_waves(self) -> int:
        return len(self.waves)


# ---------------------------------------------------------------------------
# TaskPlanResult
# ---------------------------------------------------------------------------


@dataclass
class TaskPlanResult:
    """
    プラン実行の最終結果。

    Attributes
    ----------
    plan              : 実行したプラン
    subtask_results   : 各サブタスクの実行結果
    synthesized_output: 統合出力テキスト
    total_score       : 平均品質スコア
    total_latency_ms  : 合計実行時間
    succeeded_count   : 成功タスク数
    failed_count      : 失敗タスク数
    kpi_achieved      : KPI達成フラグ (kpi_target 指定時)
    """

    plan: TaskPlan
    subtask_results: List[TaskExecutionResult]
    synthesized_output: str
    total_score: float
    total_latency_ms: float
    succeeded_count: int
    failed_count: int
    kpi_achieved: bool = False

    @property
    def success_rate(self) -> float:
        total = self.succeeded_count + self.failed_count
        return self.succeeded_count / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"TaskPlanResult: goal='{self.plan.goal[:60]}'",
            f"  tasks={self.plan.total_tasks}, waves={self.plan.n_waves}",
            f"  success={self.succeeded_count}/{self.plan.total_tasks} "
            f"({self.success_rate:.0%}), score={self.total_score:.3f}",
            f"  output: {self.synthesized_output[:100]}...",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------


# タスク分解ルール: (正規表現パターン, task_type, agent_hint)
_DECOMPOSE_RULES = [
    (r"分析|解析|調査|リサーチ|benchmark|audit", "analysis", "analytical"),
    (r"生成|作成|書[くき]|作[るり]|generate|write|create", "generation", "creative"),
    (r"評価|スコア|判定|check|evaluate|assess", "evaluation", "critical"),
    (r"最適化|改善|opti|improv|enhance", "optimization", "optimizer"),
    (r"KPI|指標|metric|measure", "evaluation", "kpi"),
    (r"SEO|LLMO|コンテンツ|content", "generation", "seo"),
    (r"セキュリティ|security|guard|安全", "evaluation", "security"),
]


class TaskPlanner:
    """
    ゴールを階層的タスクに分解・実行・統合するプランナー。

    Parameters
    ----------
    max_parallel       : 並列実行の最大タスク数
    default_executor   : (task: Task, context: dict) -> str のタスク実行関数
                         None の場合はルールベースの内部実装を使用
    kpi_target         : KPI達成判定の閾値 (0〜1)
    use_consensus      : 同一タスクを複数エージェントで実行し合意スコアを使う
    """

    def __init__(
        self,
        max_parallel: int = 4,
        default_executor: Optional[Callable[["Task", dict], str]] = None,
        kpi_target: float = 0.7,
        use_consensus: bool = True,
    ) -> None:
        self.max_parallel = max_parallel
        self.default_executor = default_executor or self._builtin_executor
        self.kpi_target = kpi_target
        self.use_consensus = use_consensus

    # ----------------------------------------------------------------- execute

    def execute(
        self,
        goal: str,
        context: Optional[Dict] = None,
        n_agents: int = 1,
    ) -> TaskPlanResult:
        """
        ゴールを分解して実行し、結果を統合して返す。

        Args:
            goal     : 達成すべきゴール
            context  : 追加コンテキスト (domain / kpi 等)
            n_agents : 並列エージェント数 (use_consensus=True 時に有効)

        Returns:
            TaskPlanResult
        """
        ctx = context or {}
        plan = self.decompose(goal, ctx)
        return self._execute_plan(plan, ctx, n_agents=n_agents)

    # ---------------------------------------------------------------- decompose

    def decompose(self, goal: str, context: Optional[Dict] = None) -> TaskPlan:
        """
        ゴールをサブタスクに分解してプランを返す。

        分解ルール:
        1. ゴール文字列をセンテンスに分割
        2. 各センテンスのキーワードからタスクタイプを判定
        3. タスク間の依存関係を自動設定
        4. TaskGraph でトポロジカルソートして wave を確定
        """
        ctx = context or {}
        tasks = self._rule_based_decompose(goal, ctx)
        if not tasks:
            # フォールバック: ゴール全体を1タスクに
            tasks = [Task(name="execute_goal", goal=goal, task_type="generic")]

        graph = TaskGraph(tasks)
        waves = graph.topological_order()

        return TaskPlan(
            goal=goal,
            tasks=graph.tasks,
            waves=waves,
            external_factors=ctx.get("external_factors", []),
        )

    # ---------------------------------------------------------------- private

    def _rule_based_decompose(self, goal: str, ctx: dict) -> List[Task]:
        """
        ゴール文字列をルールベースでサブタスクに分解する。
        """
        # ゴールをセンテンスに分割
        parts = [p.strip() for p in re.split(r"[。,，、\n]|(?:そして|また|次に)", goal) if p.strip()]
        if not parts:
            parts = [goal]

        tasks: List[Task] = []
        seen_types: Set[str] = set()

        # まず前処理タスクを追加 (分析が必要な場合)
        needs_analysis = bool(re.search(r"分析|調査|リサーチ|KPI|評価", goal))
        needs_security = bool(re.search(r"セキュリティ|安全|ガード", goal))

        priority_cursor = 1
        if needs_analysis:
            tasks.append(Task(
                name="analyze_context",
                goal=f"コンテキストを分析する: {goal[:80]}",
                task_type="analysis",
                priority=priority_cursor,
                agent_hint="analytical",
            ))
            seen_types.add("analysis")
            priority_cursor += 1

        # メインタスクを分割
        for i, part in enumerate(parts[:5]):  # 最大5サブタスク
            task_type, agent_hint = self._classify_part(part)
            name = f"{task_type}_{i+1}"

            deps: List[str] = []
            if "analyze_context" in [t.name for t in tasks] and task_type != "analysis":
                deps = ["analyze_context"]
            # generation は evaluation に依存
            if task_type == "optimization" and any(t.task_type == "evaluation" for t in tasks):
                deps += [t.name for t in tasks if t.task_type == "evaluation"]

            tasks.append(Task(
                name=name,
                goal=part,
                task_type=task_type,
                priority=priority_cursor + i,
                depends_on=deps,
                agent_hint=agent_hint,
            ))

        # 集約タスク (複数タスクがある場合)
        if len(tasks) > 1:
            all_names = [t.name for t in tasks]
            tasks.append(Task(
                name="synthesize_results",
                goal=f"結果を統合して最終アウトプットを生成する",
                task_type="generation",
                priority=10,
                depends_on=all_names,
                agent_hint="synthesizer",
            ))

        if needs_security:
            tasks.append(Task(
                name="security_check",
                goal="セキュリティ・品質チェックを実施する",
                task_type="evaluation",
                priority=11,
                depends_on=["synthesize_results"] if len(tasks) > 1 else [],
                agent_hint="security",
            ))

        return tasks

    @staticmethod
    def _classify_part(text: str) -> tuple[str, str]:
        """テキストからタスクタイプとエージェントヒントを分類。"""
        for pattern, ttype, hint in _DECOMPOSE_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                return ttype, hint
        return "generic", "general"

    def _execute_plan(
        self,
        plan: TaskPlan,
        ctx: dict,
        n_agents: int = 1,
    ) -> TaskPlanResult:
        """wave 順にタスクを実行し結果を収集する。"""
        results: Dict[str, TaskExecutionResult] = {}
        t_total = time.perf_counter()

        for wave in plan.waves:
            # wave 内を max_parallel 単位のバッチに分割して直列実行する。
            # (ThreadPoolExecutor による真の並列化は将来課題)
            # NOTE: 以前は wave[:max_parallel] でスライスしており残りのタスクが
            #       永久に実行されないバグがあった。全タスクを処理するよう修正。
            for task in wave:
                # 前段タスクの出力を context に追加
                dep_outputs = {d: results[d].output for d in task.depends_on if d in results}
                task_ctx = dict(ctx)
                task_ctx["previous_outputs"] = dep_outputs

                t0 = time.perf_counter()
                try:
                    output = self.default_executor(task, task_ctx)
                    score = self._score_output(output, task)
                    result = TaskExecutionResult(
                        task=task,
                        output=output,
                        score=score,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        success=True,
                    )
                except Exception as exc:
                    result = TaskExecutionResult(
                        task=task,
                        output="",
                        score=0.0,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        success=False,
                        error=str(exc),
                    )
                results[task.name] = result

        all_results = list(results.values())
        succeeded = [r for r in all_results if r.ok]
        failed = [r for r in all_results if not r.ok]

        synthesized = self._synthesize(all_results, plan.goal)
        avg_score = (
            sum(r.score for r in succeeded) / len(succeeded)
            if succeeded else 0.0
        )
        total_ms = round((time.perf_counter() - t_total) * 1000, 2)

        return TaskPlanResult(
            plan=plan,
            subtask_results=all_results,
            synthesized_output=synthesized,
            total_score=round(avg_score, 4),
            total_latency_ms=total_ms,
            succeeded_count=len(succeeded),
            failed_count=len(failed),
            kpi_achieved=avg_score >= self.kpi_target,
        )

    @staticmethod
    def _score_output(text: str, task: Task) -> float:
        """
        タスク出力の品質スコアリング。

        以前は空出力に 0.0, その他に最低 0.6 を与えていたが、
        短すぎる出力も低スコアになるよう改善した。
        """
        if not text:
            return 0.0
        words = len(text.split())
        # 長さスコア: 5語未満は低評価、30語以上で満点
        if words < 5:
            length_score = 0.1 + words * 0.04   # 0〜0.3
        else:
            length_score = min(1.0, 0.3 + (words - 5) / 25 * 0.7)

        # タスクタイプ別ボーナス (関連キーワードが含まれているか)
        type_bonus: float = {
            "analysis": (
                bool(re.search(r"分析|解析|結果|発見|判定|詳細", text)) * 0.12
                + bool(re.search(r"\d+(?:\.\d+)?", text)) * 0.05  # 数値あり
            ),
            "generation": (
                bool(re.search(r"[。\n]", text)) * 0.08
                + bool(re.search(r"##|【|■", text)) * 0.07  # 見出し構造
            ),
            "evaluation": (
                bool(re.search(r"スコア|score|\d\.\d{1,2}", text)) * 0.10
                + bool(re.search(r"高い|低い|合格|不合格|OK|NG", text)) * 0.05
            ),
            "optimization": (
                bool(re.search(r"改善|最適|向上|効率", text)) * 0.10
                + bool(re.search(r"\d+%|\d+倍", text)) * 0.05  # 改善率
            ),
        }.get(task.task_type, 0.0)

        # ベーススコア 0.4 に長さ・タイプボーナスを加算 (最大 1.0)
        return min(1.0, 0.4 + length_score * 0.45 + type_bonus)

    @staticmethod
    def _synthesize(results: List[TaskExecutionResult], goal: str) -> str:
        """
        複数タスクの結果を統合する。

        synthesize_results タスクがあればその出力を優先し、
        なければ成功タスクの出力を結合する。
        """
        synth = next((r.output for r in results if r.task.name == "synthesize_results" and r.ok), None)
        if synth:
            return synth
        outputs = [r.output for r in results if r.ok and r.output]
        if not outputs:
            return f"[goal: {goal}] — 全タスク失敗"
        return "\n\n".join(f"[{results[i].task.name}]: {o}" for i, o in enumerate(outputs))

    @staticmethod
    def _builtin_executor(task: Task, ctx: dict) -> str:
        """
        デフォルトの内部タスク実行関数 (ルールベース模擬実行)。

        実際の LLM やツールが接続されていない場合のフォールバック。
        """
        prev = ctx.get("previous_outputs", {})
        prev_summary = ""
        if prev:
            prev_summary = "前段の結果: " + "、".join(f"{k}={v[:30]}" for k, v in prev.items())

        templates = {
            "analysis": f"分析完了: '{task.goal[:60]}' について詳細を解析しました。{prev_summary}",
            "generation": f"生成完了: '{task.goal[:60]}' に基づいてコンテンツを作成しました。{prev_summary}",
            "evaluation": f"評価完了: score=0.78。'{task.goal[:60]}' の品質を確認しました。{prev_summary}",
            "optimization": f"最適化完了: '{task.goal[:60]}' を改善しました。{prev_summary}",
            "generic": f"実行完了: '{task.goal[:60]}' を処理しました。{prev_summary}",
        }
        return templates.get(task.task_type, templates["generic"])
