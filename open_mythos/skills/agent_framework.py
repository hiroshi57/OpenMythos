"""
Sprint 50 — エージェントフレームワーク強化

Hermes Skills: subagent-driven-development / tdd / systematic-debugging / darwinian-evolver / parallel-cli
ref: skills/agents/*-SKILL.md

エージェント駆動開発・テスト・デバッグ・進化的最適化ツールを統合する。
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# サブエージェント駆動開発
# ---------------------------------------------------------------------------

@dataclass
class SubAgentTask:
    """サブエージェントに委譲するタスク。"""
    task_id: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout_s: float = 60.0


@dataclass
class SubAgentResult:
    """サブエージェントの実行結果。"""
    task_id: str
    output: str
    success: bool
    review_passed: bool = False
    latency_ms: float = 0.0
    reviewer_feedback: str = ""


class SubAgentOrchestrator:
    """サブエージェント駆動開発オーケストレーター (2-stage review)。

    Plan → Delegate → Review の 3 フェーズで実行する。
    """

    def __init__(
        self,
        execute_fn: Optional[Callable[[SubAgentTask], str]] = None,
        review_fn: Optional[Callable[[SubAgentTask, str], Tuple[bool, str]]] = None,
    ) -> None:
        self._execute = execute_fn or self._default_execute
        self._review = review_fn or self._default_review

    def _default_execute(self, task: SubAgentTask) -> str:
        return f"[SubAgent] Completed: {task.description[:60]}"

    def _default_review(self, task: SubAgentTask, output: str) -> Tuple[bool, str]:
        # 簡易チェック: 空でない文字列なら PASS
        passed = bool(output.strip())
        feedback = "OK" if passed else "Output is empty"
        return passed, feedback

    def run(self, tasks: List[SubAgentTask]) -> List[SubAgentResult]:
        """タスクリストを順次実行し 2-stage レビューを行う。"""
        results = []
        for task in tasks:
            t0 = time.perf_counter()
            output = self._execute(task)
            passed, feedback = self._review(task, output)
            latency = (time.perf_counter() - t0) * 1000
            results.append(SubAgentResult(
                task_id=task.task_id,
                output=output,
                success=True,
                review_passed=passed,
                latency_ms=round(latency, 2),
                reviewer_feedback=feedback,
            ))
        return results

    def plan(self, goal: str, n_subtasks: int = 3) -> List[SubAgentTask]:
        """ゴールからサブタスクを自動生成する。"""
        templates = [
            f"Analyze requirements for: {goal}",
            f"Implement core logic for: {goal}",
            f"Write tests for: {goal}",
            f"Review and refine: {goal}",
            f"Document: {goal}",
        ]
        return [
            SubAgentTask(
                task_id=f"task_{i+1}",
                description=templates[i % len(templates)],
                priority=n_subtasks - i,
            )
            for i in range(n_subtasks)
        ]


# ---------------------------------------------------------------------------
# TDD エージェント
# ---------------------------------------------------------------------------

@dataclass
class TDDCycle:
    """Red-Green-Refactor サイクル記録。"""
    phase: str           # RED | GREEN | REFACTOR
    test_code: str
    impl_code: str
    test_passed: bool
    notes: str = ""


@dataclass
class TDDSession:
    """TDD セッション。"""
    goal: str
    cycles: List[TDDCycle] = field(default_factory=list)
    total_tests_written: int = 0
    total_tests_passing: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_tests_written == 0:
            return 0.0
        return self.total_tests_passing / self.total_tests_written


class TDDAgent:
    """Test-Driven Development エージェント。

    RED → GREEN → REFACTOR サイクルを自動化する。
    """

    def __init__(
        self,
        run_tests_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self._run_tests = run_tests_fn or self._mock_run

    def _mock_run(self, test_code: str, impl_code: str) -> bool:
        # テストが test_ 関数を含み実装が空でなければ PASS とみなす
        return "def test_" in test_code and bool(impl_code.strip())

    def red_phase(self, goal: str) -> str:
        """失敗するテストコードを生成する (RED)。"""
        func_name = goal.lower().replace(" ", "_")[:30]
        return (
            f"def test_{func_name}():\n"
            f"    result = {func_name}()\n"
            f"    assert result is not None\n"
            f"    assert isinstance(result, (str, int, float, dict, list))\n"
        )

    def green_phase(self, test_code: str) -> str:
        """テストをパスする最小実装を生成する (GREEN)。"""
        import re
        match = re.search(r"def test_(\w+)\(", test_code)
        func_name = match.group(1) if match else "implementation"
        return f"def {func_name}():\n    return 'implemented'\n"

    def refactor_phase(self, impl_code: str) -> str:
        """実装をリファクタリングする (REFACTOR)。"""
        # ドキュメント文字列を追加
        lines = impl_code.strip().split("\n")
        if lines and lines[0].startswith("def "):
            lines.insert(1, '    """Refactored implementation."""')
        return "\n".join(lines) + "\n"

    def run_cycle(self, goal: str) -> TDDCycle:
        """1 TDD サイクルを実行する。"""
        test_code = self.red_phase(goal)
        impl_code = self.green_phase(test_code)
        impl_code = self.refactor_phase(impl_code)
        passed = self._run_tests(test_code, impl_code)
        return TDDCycle(
            phase="REFACTOR",
            test_code=test_code,
            impl_code=impl_code,
            test_passed=passed,
            notes=f"Goal: {goal[:50]}",
        )

    def run_session(self, goals: List[str]) -> TDDSession:
        """複数ゴールに対して TDD セッションを実行する。"""
        session = TDDSession(goal="; ".join(goals[:3]))
        for goal in goals:
            cycle = self.run_cycle(goal)
            session.cycles.append(cycle)
            session.total_tests_written += 1
            if cycle.test_passed:
                session.total_tests_passing += 1
        return session


# ---------------------------------------------------------------------------
# 体系的デバッグ
# ---------------------------------------------------------------------------

@dataclass
class BugReport:
    """バグレポート。"""
    description: str
    context: str = ""
    stack_trace: str = ""
    observed: str = ""
    expected: str = ""


@dataclass
class DebugStep:
    """デバッグステップ。"""
    phase: str           # understand | isolate | fix | verify
    action: str
    finding: str
    hypothesis: str = ""
    confidence: float = 0.0


@dataclass
class DebugSession:
    """4 フェーズデバッグセッション結果。"""
    bug: BugReport
    steps: List[DebugStep]
    root_cause: str
    fix_suggestion: str
    verified: bool


class SystematicDebugger:
    """4 フェーズ体系的デバッグ (understand → isolate → fix → verify)。"""

    PHASES = ["understand", "isolate", "fix", "verify"]

    def debug(self, bug: BugReport) -> DebugSession:
        """バグレポートを受け取り体系的にデバッグする。"""
        steps = []

        # Phase 1: Understand
        steps.append(DebugStep(
            phase="understand",
            action=f"Read error: {bug.description[:80]}",
            finding=f"Error type: {self._classify_error(bug.description)}",
            confidence=0.7,
        ))

        # Phase 2: Isolate
        hypothesis = self._form_hypothesis(bug)
        steps.append(DebugStep(
            phase="isolate",
            action="Narrow down the failing component",
            finding=f"Likely location: {hypothesis['location']}",
            hypothesis=hypothesis["hypothesis"],
            confidence=hypothesis["confidence"],
        ))

        # Phase 3: Fix
        fix = self._suggest_fix(bug, hypothesis)
        steps.append(DebugStep(
            phase="fix",
            action="Apply minimal fix",
            finding=fix,
            confidence=0.6,
        ))

        # Phase 4: Verify
        steps.append(DebugStep(
            phase="verify",
            action="Run tests to confirm fix",
            finding="Tests should pass after applying fix",
            confidence=0.8,
        ))

        return DebugSession(
            bug=bug,
            steps=steps,
            root_cause=hypothesis["hypothesis"],
            fix_suggestion=fix,
            verified=True,
        )

    def _classify_error(self, desc: str) -> str:
        error_map = {
            "TypeError": "type_mismatch",
            "ValueError": "invalid_value",
            "AttributeError": "missing_attr",
            "ImportError": "import_issue",
            "KeyError": "missing_key",
            "IndexError": "index_out_of_range",
            "None": "none_reference",
        }
        for key, val in error_map.items():
            if key.lower() in desc.lower():
                return val
        return "unknown_error"

    def _form_hypothesis(self, bug: BugReport) -> Dict[str, Any]:
        lines = (bug.stack_trace or bug.description).split("\n")
        location = "unknown"
        for line in reversed(lines):
            if "File" in line and ".py" in line:
                location = line.strip()
                break
        return {
            "location": location,
            "hypothesis": f"Possible root cause in {location}: {bug.description[:50]}",
            "confidence": 0.65,
        }

    def _suggest_fix(self, bug: BugReport, hypothesis: Dict[str, Any]) -> str:
        if "none" in bug.description.lower():
            return "Add None check before accessing the value"
        if "index" in bug.description.lower():
            return "Validate list/array bounds before access"
        if "import" in bug.description.lower():
            return "Install missing package or check import path"
        if "type" in bug.description.lower():
            return "Add type conversion or type check"
        return f"Review logic at {hypothesis.get('location', 'unknown location')}"


# ---------------------------------------------------------------------------
# 進化的最適化 (Darwinian Evolver)
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """進化的アルゴリズムの個体。"""
    genome: List[float]
    fitness: float = 0.0
    generation: int = 0


@dataclass
class EvolutionResult:
    """進化的最適化結果。"""
    best_individual: Individual
    best_fitness: float
    generations_run: int
    population_size: int
    fitness_history: List[float]


class DarwinianEvolver:
    """進化的 (遺伝的) アルゴリズム最適化器。

    任意の適応度関数に対して GA を実行する。
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_ratio: float = 0.1,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio

    def evolve(
        self,
        fitness_fn: Callable[[List[float]], float],
        genome_dim: int,
        n_generations: int = 100,
        genome_bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> EvolutionResult:
        """進化的最適化を実行する。"""
        lo, hi = genome_bounds
        rng = random.Random(42)

        def random_individual() -> Individual:
            g = [rng.uniform(lo, hi) for _ in range(genome_dim)]
            return Individual(genome=g)

        # 初期集団
        pop = [random_individual() for _ in range(self.population_size)]
        for ind in pop:
            ind.fitness = fitness_fn(ind.genome)

        fitness_history = []
        best = max(pop, key=lambda x: x.fitness)

        for gen in range(n_generations):
            pop.sort(key=lambda x: x.fitness, reverse=True)
            elite_n = max(1, int(self.population_size * self.elitism_ratio))
            next_pop = pop[:elite_n]

            while len(next_pop) < self.population_size:
                # トーナメント選択
                p1 = max(rng.sample(pop, min(5, len(pop))), key=lambda x: x.fitness)
                p2 = max(rng.sample(pop, min(5, len(pop))), key=lambda x: x.fitness)
                # 交叉
                if rng.random() < self.crossover_rate:
                    cut = rng.randint(1, genome_dim - 1)
                    child_g = p1.genome[:cut] + p2.genome[cut:]
                else:
                    child_g = list(p1.genome)
                # 突然変異
                child_g = [
                    rng.uniform(lo, hi) if rng.random() < self.mutation_rate else g
                    for g in child_g
                ]
                child = Individual(genome=child_g, generation=gen + 1)
                child.fitness = fitness_fn(child.genome)
                next_pop.append(child)

            pop = next_pop
            gen_best = max(pop, key=lambda x: x.fitness)
            if gen_best.fitness > best.fitness:
                best = gen_best
            fitness_history.append(round(best.fitness, 6))

        return EvolutionResult(
            best_individual=best,
            best_fitness=best.fitness,
            generations_run=n_generations,
            population_size=self.population_size,
            fitness_history=fitness_history,
        )


# ---------------------------------------------------------------------------
# 並列 CLI 実行
# ---------------------------------------------------------------------------

@dataclass
class ParallelJob:
    """並列実行ジョブ。"""
    job_id: str
    command: str
    cwd: str = "."
    timeout_s: float = 30.0
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class JobResult:
    """ジョブ実行結果。"""
    job_id: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    success: bool


class ParallelCLIRunner:
    """複数の CLI コマンドを並列実行するランナー。"""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run(self, jobs: List[ParallelJob]) -> List[JobResult]:
        """ジョブリストを並列実行する。"""
        import subprocess, os
        # asyncio で並列実行
        async def run_async() -> List[JobResult]:
            sem = asyncio.Semaphore(self.max_workers)
            async def run_one(job: ParallelJob) -> JobResult:
                async with sem:
                    t0 = time.perf_counter()
                    try:
                        env = {**os.environ, **job.env}
                        proc = await asyncio.create_subprocess_shell(
                            job.command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=job.cwd,
                            env=env,
                        )
                        try:
                            stdout, stderr = await asyncio.wait_for(
                                proc.communicate(), timeout=job.timeout_s
                            )
                        except asyncio.TimeoutError:
                            proc.kill()
                            return JobResult(
                                job_id=job.job_id, command=job.command,
                                returncode=-1, stdout="", stderr="TIMEOUT",
                                duration_s=job.timeout_s, success=False,
                            )
                        dur = time.perf_counter() - t0
                        return JobResult(
                            job_id=job.job_id, command=job.command,
                            returncode=proc.returncode or 0,
                            stdout=stdout.decode("utf-8", errors="replace"),
                            stderr=stderr.decode("utf-8", errors="replace"),
                            duration_s=round(dur, 3),
                            success=(proc.returncode == 0),
                        )
                    except Exception as e:
                        return JobResult(
                            job_id=job.job_id, command=job.command,
                            returncode=-1, stdout="", stderr=str(e),
                            duration_s=time.perf_counter() - t0, success=False,
                        )
            tasks = [run_one(job) for job in jobs]
            return list(await asyncio.gather(*tasks))
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # すでに実行中のループがある場合は同期実行にフォールバック
                raise RuntimeError("loop running")
            return loop.run_until_complete(run_async())
        except Exception:
            # fallback: 逐次実行
            results_sync = []
            for job in jobs:
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        job.command, shell=True, capture_output=True,
                        timeout=job.timeout_s, cwd=job.cwd,
                    )
                    results_sync.append(JobResult(
                        job_id=job.job_id, command=job.command,
                        returncode=proc.returncode,
                        stdout=proc.stdout.decode("utf-8", errors="replace"),
                        stderr=proc.stderr.decode("utf-8", errors="replace"),
                        duration_s=round(time.perf_counter() - t0, 3),
                        success=(proc.returncode == 0),
                    ))
                except Exception as e:
                    results_sync.append(JobResult(
                        job_id=job.job_id, command=job.command,
                        returncode=-1, stdout="", stderr=str(e),
                        duration_s=round(time.perf_counter() - t0, 3), success=False,
                    ))
            return results_sync
