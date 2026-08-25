"""
Sprint 80F — TraceCompiler: Skill-Guided Mining and Compilation of LLM
Agent Traces into Mostly Deterministic Workflows

arXiv:2608.02680 の OpenMythos Python 移植。

設計概要:
    LLMエージェントの実行トレースを収集 → 共通スキルパターンを採掘 →
    決定論的ワークフローにコンパイル → WorkflowExecutor で再実行。

パイプライン:
    AgentTrace[] ──► TraceMiner ──► SkillPattern[]
                                        │
                 AgentTrace[] ──► TraceCompiler ──► CompiledWorkflow
                                                          │
                                 WorkflowExecutor ◄───────┘
                                        │
                                 ExecutionResult

論文対応表:
    論文                  | trace_compiler.py
    ─────────────────────|──────────────────────────────────
    Trace                | AgentTrace
    Step / Action        | TraceStep
    Skill                | SkillPattern
    Skill Library        | SkillStore
    Compiled Workflow    | CompiledWorkflow
    Workflow Step        | WorkflowStep (det. / llm 2種)
    Determinism Score    | CompiledWorkflow.determinism_score
    Mining (LCS法)       | TraceMiner.mine_skills
    Compilation          | TraceCompiler.compile
    Execution            | WorkflowExecutor.execute
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ─── 定数 ──────────────────────────────────────────────────────────

MIN_SKILL_FREQUENCY = 2    # スキルとして認定する最低出現頻度
MIN_SKILL_LENGTH    = 2    # スキルとして認定する最低ステップ数
MAX_WORKFLOW_STEPS  = 100  # コンパイル後の最大ステップ数


# ─── Enums ────────────────────────────────────────────────────────

class StepType(str, Enum):
    """エージェントトレースのステップ種別。"""
    LLM_CALL   = "llm_call"    # LLM API 呼び出し
    TOOL_CALL  = "tool_call"   # 外部ツール呼び出し
    DECISION   = "decision"    # 分岐判断
    RETRIEVAL  = "retrieval"   # RAG / 検索
    TRANSFORM  = "transform"   # データ変換
    OUTPUT     = "output"      # 最終出力


class WorkflowStepKind(str, Enum):
    """コンパイル済みワークフローのステップ種別。"""
    DETERMINISTIC = "deterministic"  # スキルパターンから確定的に生成
    LLM_ASSISTED  = "llm_assisted"   # LLM が必要（不確定）


class CompilationStatus(str, Enum):
    SUCCESS    = "success"
    PARTIAL    = "partial"    # 一部のみコンパイル可能
    FAILED     = "failed"


class ExecutionStatus(str, Enum):
    SUCCESS    = "success"
    FAILED     = "failed"
    PARTIAL    = "partial"


# ─── Data Classes: Trace 層 ───────────────────────────────────────

@dataclass
class TraceStep:
    """エージェント実行の 1 ステップ。"""
    step_id:    str
    step_type:  StepType
    action:     str            # 実行したアクション名
    inputs:     Dict[str, Any] = field(default_factory=dict)
    outputs:    Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    success:    bool = True
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "action": self.action,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "timestamp": self.timestamp,
        }

    def signature(self) -> str:
        """ステップの「型シグネチャ」。マイニング時の比較キーとして使用。"""
        return f"{self.step_type.value}:{self.action}"


@dataclass
class AgentTrace:
    """LLMエージェントの 1 回の実行トレース全体。"""
    trace_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    task:       str = ""
    steps:      List[TraceStep] = field(default_factory=list)
    success:    bool = True
    total_ms:   float = 0.0
    metadata:   Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def signatures(self) -> List[str]:
        """ステップのシグネチャリスト（マイニング用）。"""
        return [s.signature() for s in self.steps]

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "step_count": len(self.steps),
            "success": self.success,
            "total_ms": round(self.total_ms, 2),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
        }


# ─── Data Classes: Skill 層 ───────────────────────────────────────

@dataclass
class SkillPattern:
    """複数のトレースから採掘されたスキルパターン。"""
    skill_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:        str = ""
    signatures:  List[str] = field(default_factory=list)  # ステップシグネチャ列
    frequency:   int = 0            # 何本のトレースに出現したか
    avg_success_rate: float = 1.0   # 出現時の成功率
    source_traces: List[str] = field(default_factory=list)  # trace_id リスト
    created_at:  float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        return len(self.signatures)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "signatures": self.signatures,
            "length": self.length,
            "frequency": self.frequency,
            "avg_success_rate": round(self.avg_success_rate, 3),
            "source_traces": self.source_traces,
        }


@dataclass
class SkillStore:
    """スキルパターンの CRUD ストア。"""
    _skills: Dict[str, SkillPattern] = field(default_factory=dict)

    def add(self, skill: SkillPattern) -> None:
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> Optional[SkillPattern]:
        return self._skills.get(skill_id)

    def list_all(self) -> List[SkillPattern]:
        return sorted(self._skills.values(), key=lambda s: -s.frequency)

    def count(self) -> int:
        return len(self._skills)

    def top_k(self, k: int = 10) -> List[SkillPattern]:
        return self.list_all()[:k]

    def remove(self, skill_id: str) -> bool:
        if skill_id in self._skills:
            del self._skills[skill_id]
            return True
        return False


# ─── Data Classes: Workflow 層 ────────────────────────────────────

@dataclass
class WorkflowStep:
    """コンパイル済みワークフローの 1 ステップ。"""
    wf_step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind:       WorkflowStepKind = WorkflowStepKind.LLM_ASSISTED
    action:     str = ""
    step_type:  StepType = StepType.LLM_CALL
    skill_id:   Optional[str] = None   # 由来スキル
    expected_inputs:  List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    fallback_to_llm:  bool = True

    def to_dict(self) -> dict:
        return {
            "wf_step_id": self.wf_step_id,
            "kind": self.kind.value,
            "action": self.action,
            "step_type": self.step_type.value,
            "skill_id": self.skill_id,
            "expected_inputs": self.expected_inputs,
            "expected_outputs": self.expected_outputs,
            "fallback_to_llm": self.fallback_to_llm,
        }


@dataclass
class CompiledWorkflow:
    """
    トレースとスキルから生成された「ほぼ決定論的」ワークフロー。
    determinism_score = 決定論的ステップ数 / 全ステップ数
    """
    workflow_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name:         str = ""
    source_task:  str = ""
    steps:        List[WorkflowStep] = field(default_factory=list)
    status:       CompilationStatus = CompilationStatus.SUCCESS
    skill_ids:    List[str] = field(default_factory=list)
    source_traces: List[str] = field(default_factory=list)
    created_at:   float = field(default_factory=time.time)

    @property
    def determinism_score(self) -> float:
        """0.0（全 LLM）〜 1.0（全決定論的）。"""
        if not self.steps:
            return 0.0
        det = sum(1 for s in self.steps if s.kind == WorkflowStepKind.DETERMINISTIC)
        return det / len(self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def deterministic_steps(self) -> List[WorkflowStep]:
        return [s for s in self.steps if s.kind == WorkflowStepKind.DETERMINISTIC]

    @property
    def llm_steps(self) -> List[WorkflowStep]:
        return [s for s in self.steps if s.kind == WorkflowStepKind.LLM_ASSISTED]

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "source_task": self.source_task,
            "step_count": self.step_count,
            "determinism_score": round(self.determinism_score, 3),
            "status": self.status.value,
            "skill_ids": self.skill_ids,
            "source_traces": self.source_traces,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class CompilationResult:
    """TraceCompiler.compile() の戻り値。"""
    workflow:       CompiledWorkflow
    status:         CompilationStatus
    skills_applied: int
    steps_compiled: int
    steps_total:    int
    warnings:       List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.steps_compiled / self.steps_total

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow.workflow_id,
            "status": self.status.value,
            "skills_applied": self.skills_applied,
            "steps_compiled": self.steps_compiled,
            "steps_total": self.steps_total,
            "coverage": round(self.coverage, 3),
            "determinism_score": round(self.workflow.determinism_score, 3),
            "warnings": self.warnings,
        }


# ─── Data Classes: Execution 層 ───────────────────────────────────

@dataclass
class StepExecutionResult:
    """WorkflowStep の実行結果。"""
    wf_step_id: str
    success:    bool
    outputs:    Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    used_llm:   bool = False
    error:      Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "wf_step_id": self.wf_step_id,
            "success": self.success,
            "outputs": self.outputs,
            "duration_ms": round(self.duration_ms, 2),
            "used_llm": self.used_llm,
            "error": self.error,
        }


@dataclass
class ExecutionResult:
    """WorkflowExecutor.execute() の戻り値。"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    workflow_id:  str = ""
    status:       ExecutionStatus = ExecutionStatus.SUCCESS
    step_results: List[StepExecutionResult] = field(default_factory=list)
    total_ms:     float = 0.0
    started_at:   float = field(default_factory=time.time)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.step_results if r.success)

    @property
    def llm_calls(self) -> int:
        return sum(1 for r in self.step_results if r.used_llm)

    @property
    def step_count(self) -> int:
        return len(self.step_results)

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "total_ms": round(self.total_ms, 2),
            "success_count": self.success_count,
            "step_count": self.step_count,
            "llm_calls": self.llm_calls,
            "started_at": self.started_at,
            "step_results": [r.to_dict() for r in self.step_results],
        }


# ─── TraceMiner (論文: Skill Mining) ─────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    """最長共通部分列 (LCS) の長さを計算 (DP)。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _lcs_sequence(a: List[str], b: List[str]) -> List[str]:
    """LCS の実際のシーケンスを復元する。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # バックトレース
    result: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(result))


class TraceMiner:
    """
    複数の AgentTrace から共通スキルパターンを採掘する。
    論文 §3.2 "Skill Mining via LCS" の実装。
    """

    def __init__(
        self,
        min_frequency: int = MIN_SKILL_FREQUENCY,
        min_length: int = MIN_SKILL_LENGTH,
    ):
        self.min_frequency = min_frequency
        self.min_length = min_length

    def mine_skills(self, traces: List[AgentTrace]) -> List[SkillPattern]:
        """
        トレース群から LCS ベースでスキルパターンを採掘する。
        成功トレースのみを対象とする（論文に準拠）。
        """
        successful = [t for t in traces if t.success]
        if len(successful) < self.min_frequency:
            return []

        # ペアワイズ LCS で共通部分列候補を収集
        candidate_map: Dict[str, List[str]] = {}  # frozen_key → sig_list

        for i in range(len(successful)):
            for j in range(i + 1, len(successful)):
                sig_a = successful[i].signatures()
                sig_b = successful[j].signatures()
                common = _lcs_sequence(sig_a, sig_b)
                if len(common) >= self.min_length:
                    key = "||".join(common)
                    if key not in candidate_map:
                        candidate_map[key] = common

        if not candidate_map:
            return []

        # 各候補の出現頻度をカウント
        skills: List[SkillPattern] = []
        for key, sigs in candidate_map.items():
            freq = 0
            src_traces: List[str] = []
            successes = 0
            for trace in successful:
                tsigs = trace.signatures()
                # sigs が tsigs の部分列として現れるか確認
                if self._is_subsequence(sigs, tsigs):
                    freq += 1
                    src_traces.append(trace.trace_id)
                    if trace.success:
                        successes += 1

            if freq >= self.min_frequency:
                skill = SkillPattern(
                    name=self._auto_name(sigs),
                    signatures=sigs,
                    frequency=freq,
                    avg_success_rate=successes / max(freq, 1),
                    source_traces=src_traces,
                )
                skills.append(skill)

        # 重複除去: 完全に包含されるスキルを削除（長いほうを残す）
        skills = self._deduplicate(skills)
        return sorted(skills, key=lambda s: (-s.frequency, -s.length))

    @staticmethod
    def _is_subsequence(sub: List[str], seq: List[str]) -> bool:
        """sub が seq の部分列かどうか判定。O(n)。"""
        it = iter(seq)
        return all(s in it for s in sub)

    @staticmethod
    def _auto_name(sigs: List[str]) -> str:
        """シグネチャ列から自動名称を生成。"""
        parts = []
        for s in sigs[:3]:
            action = s.split(":", 1)[-1]
            parts.append(action)
        name = "→".join(parts)
        if len(sigs) > 3:
            name += f"…(+{len(sigs)-3})"
        return name

    @staticmethod
    def _deduplicate(skills: List[SkillPattern]) -> List[SkillPattern]:
        """
        より短いスキルがより長いスキルに完全包含され、かつ長いスキルの頻度が
        同等以上の場合のみ除去する。
        頻度が低い長スキルで高頻度な短スキルを消さないようにする。
        """
        result: List[SkillPattern] = []
        for candidate in skills:
            dominated = False
            for other in skills:
                if other is candidate:
                    continue
                if (other.length > candidate.length and
                        other.frequency >= candidate.frequency and
                        TraceMiner._is_subsequence(candidate.signatures, other.signatures)):
                    dominated = True
                    break
            if not dominated:
                result.append(candidate)
        return result


# ─── TraceStore ───────────────────────────────────────────────────

class TraceStore:
    """AgentTrace の CRUD ストア。"""

    def __init__(self) -> None:
        self._traces: Dict[str, AgentTrace] = {}

    def add(self, trace: AgentTrace) -> None:
        self._traces[trace.trace_id] = trace

    def get(self, trace_id: str) -> Optional[AgentTrace]:
        return self._traces.get(trace_id)

    def list_all(self) -> List[AgentTrace]:
        return list(self._traces.values())

    def list_successful(self) -> List[AgentTrace]:
        return [t for t in self._traces.values() if t.success]

    def count(self) -> int:
        return len(self._traces)

    def remove(self, trace_id: str) -> bool:
        if trace_id in self._traces:
            del self._traces[trace_id]
            return True
        return False


# ─── TraceCompiler (論文: Compilation) ───────────────────────────

class TraceCompiler:
    """
    AgentTrace[] + SkillPattern[] → CompiledWorkflow。
    論文 §3.3 "Workflow Compilation" の実装。

    アルゴリズム:
        1. 代表トレース（最長の成功トレース）を選択
        2. 各ステップを既存スキルとマッチング
        3. マッチしたステップ列 → DETERMINISTIC WorkflowStep
        4. マッチしないステップ → LLM_ASSISTED WorkflowStep
    """

    def compile(
        self,
        traces: List[AgentTrace],
        skills: List[SkillPattern],
        name: str = "",
    ) -> CompilationResult:
        successful = [t for t in traces if t.success]
        if not successful:
            empty_wf = CompiledWorkflow(
                name=name or "empty",
                status=CompilationStatus.FAILED,
            )
            return CompilationResult(
                workflow=empty_wf,
                status=CompilationStatus.FAILED,
                skills_applied=0,
                steps_compiled=0,
                steps_total=0,
                warnings=["成功トレースが存在しません"],
            )

        # 代表トレース: 最もステップ数が多い成功トレース
        representative = max(successful, key=lambda t: len(t.steps))

        wf_steps: List[WorkflowStep] = []
        skills_applied: int = 0
        steps_compiled: int = 0
        warnings: List[str] = []
        skill_ids_used: List[str] = []

        sigs = representative.signatures()
        i = 0
        while i < len(sigs):
            # 最長マッチするスキルを探す（greedy）
            best_skill: Optional[SkillPattern] = None
            best_len = 0
            for skill in sorted(skills, key=lambda s: -s.length):
                sk_sigs = skill.signatures
                if i + len(sk_sigs) <= len(sigs):
                    if sigs[i:i + len(sk_sigs)] == sk_sigs:
                        if len(sk_sigs) > best_len:
                            best_skill = skill
                            best_len = len(sk_sigs)

            if best_skill is not None:
                # スキルにマッチ → DETERMINISTIC ステップ群を生成
                for k, sig in enumerate(best_skill.signatures):
                    step = representative.steps[i + k]
                    wf_steps.append(WorkflowStep(
                        kind=WorkflowStepKind.DETERMINISTIC,
                        action=step.action,
                        step_type=step.step_type,
                        skill_id=best_skill.skill_id,
                        expected_inputs=list(step.inputs.keys()),
                        expected_outputs=list(step.outputs.keys()),
                    ))
                    steps_compiled += 1
                if best_skill.skill_id not in skill_ids_used:
                    skill_ids_used.append(best_skill.skill_id)
                    skills_applied += 1
                i += best_len
            else:
                # マッチなし → LLM_ASSISTED
                step = representative.steps[i]
                wf_steps.append(WorkflowStep(
                    kind=WorkflowStepKind.LLM_ASSISTED,
                    action=step.action,
                    step_type=step.step_type,
                    expected_inputs=list(step.inputs.keys()),
                    expected_outputs=list(step.outputs.keys()),
                ))
                i += 1

        # ステップ数上限チェック
        if len(wf_steps) > MAX_WORKFLOW_STEPS:
            wf_steps = wf_steps[:MAX_WORKFLOW_STEPS]
            warnings.append(f"ステップ数が上限 ({MAX_WORKFLOW_STEPS}) を超えたため切り捨てました")

        det_ratio = (steps_compiled / len(wf_steps)) if wf_steps else 0.0
        status = (
            CompilationStatus.SUCCESS if det_ratio >= 0.5
            else CompilationStatus.PARTIAL if steps_compiled > 0
            else CompilationStatus.FAILED
        )

        wf = CompiledWorkflow(
            name=name or f"workflow-{representative.trace_id}",
            source_task=representative.task,
            steps=wf_steps,
            status=status,
            skill_ids=skill_ids_used,
            source_traces=[t.trace_id for t in successful],
        )

        return CompilationResult(
            workflow=wf,
            status=status,
            skills_applied=skills_applied,
            steps_compiled=steps_compiled,
            steps_total=len(representative.steps),
            warnings=warnings,
        )


# ─── WorkflowExecutor ─────────────────────────────────────────────

StepHandler = Callable[[WorkflowStep, Dict[str, Any]], Dict[str, Any]]


class WorkflowExecutor:
    """
    CompiledWorkflow を実行する。
    論文 §3.4 "Workflow Execution with Deterministic Priority"

    - DETERMINISTIC ステップ: ハンドラ関数で確定実行
    - LLM_ASSISTED ステップ: llm_handler にフォールバック
    """

    def __init__(
        self,
        step_handlers: Optional[Dict[str, StepHandler]] = None,
        llm_handler: Optional[StepHandler] = None,
    ):
        # action 名 → ハンドラ関数のマップ
        self._handlers: Dict[str, StepHandler] = step_handlers or {}
        self._llm_handler: Optional[StepHandler] = llm_handler

    def register_handler(self, action: str, handler: StepHandler) -> None:
        self._handlers[action] = handler

    def execute(
        self,
        workflow: CompiledWorkflow,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        ctx: Dict[str, Any] = dict(context or {})
        step_results: List[StepExecutionResult] = []
        start = time.time()

        for step in workflow.steps:
            step_start = time.time()
            used_llm = False
            error: Optional[str] = None
            outputs: Dict[str, Any] = {}
            success = False

            try:
                if step.kind == WorkflowStepKind.DETERMINISTIC and step.action in self._handlers:
                    handler = self._handlers[step.action]
                    outputs = handler(step, ctx)
                    success = True
                elif self._llm_handler is not None:
                    outputs = self._llm_handler(step, ctx)
                    used_llm = True
                    success = True
                else:
                    # ハンドラなし → スキップ（partial success）
                    outputs = {}
                    success = True  # スキップは失敗扱いにしない
                    error = f"no_handler:{step.action}"

                ctx.update(outputs)

            except Exception as exc:
                error = str(exc)
                success = False

            step_results.append(StepExecutionResult(
                wf_step_id=step.wf_step_id,
                success=success,
                outputs=outputs,
                duration_ms=(time.time() - step_start) * 1000,
                used_llm=used_llm,
                error=error,
            ))

        total_ms = (time.time() - start) * 1000
        all_success = all(r.success for r in step_results)
        any_success = any(r.success for r in step_results)

        status = (
            ExecutionStatus.SUCCESS if all_success
            else ExecutionStatus.PARTIAL if any_success
            else ExecutionStatus.FAILED
        )

        return ExecutionResult(
            workflow_id=workflow.workflow_id,
            status=status,
            step_results=step_results,
            total_ms=total_ms,
        )


# ─── WorkflowStore ───────────────────────────────────────────────

class WorkflowStore:
    """CompiledWorkflow の CRUD ストア。"""

    def __init__(self) -> None:
        self._workflows: Dict[str, CompiledWorkflow] = {}

    def add(self, wf: CompiledWorkflow) -> None:
        self._workflows[wf.workflow_id] = wf

    def get(self, workflow_id: str) -> Optional[CompiledWorkflow]:
        return self._workflows.get(workflow_id)

    def list_all(self) -> List[CompiledWorkflow]:
        return sorted(self._workflows.values(), key=lambda w: -w.determinism_score)

    def count(self) -> int:
        return len(self._workflows)

    def top_k_by_determinism(self, k: int = 5) -> List[CompiledWorkflow]:
        return self.list_all()[:k]


# ─── TraceCompilerPipeline (全体ファサード) ────────────────────────

class TraceCompilerPipeline:
    """
    TraceMiner → TraceCompiler → WorkflowStore の全パイプラインを
    1 オブジェクトで管理するファサード。
    """

    def __init__(
        self,
        min_frequency: int = MIN_SKILL_FREQUENCY,
        min_skill_length: int = MIN_SKILL_LENGTH,
    ):
        self.trace_store    = TraceStore()
        self.skill_store    = SkillStore()
        self.workflow_store = WorkflowStore()
        self._miner    = TraceMiner(min_frequency=min_frequency, min_length=min_skill_length)
        self._compiler = TraceCompiler()

    def ingest(self, trace: AgentTrace) -> None:
        """トレースを取り込む。"""
        self.trace_store.add(trace)

    def mine(self) -> List[SkillPattern]:
        """保存済みトレースからスキルを採掘してスキルストアに登録。"""
        traces = self.trace_store.list_all()
        skills = self._miner.mine_skills(traces)
        for skill in skills:
            self.skill_store.add(skill)
        return skills

    def compile(self, name: str = "") -> CompilationResult:
        """保存済みトレース + スキルからワークフローをコンパイル。"""
        traces = self.trace_store.list_all()
        skills = self.skill_store.list_all()
        result = self._compiler.compile(traces, skills, name=name)
        if result.status != CompilationStatus.FAILED:
            self.workflow_store.add(result.workflow)
        return result

    def run(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
        executor: Optional[WorkflowExecutor] = None,
    ) -> Optional[ExecutionResult]:
        """コンパイル済みワークフローを実行。"""
        wf = self.workflow_store.get(workflow_id)
        if wf is None:
            return None
        exec_ = executor or WorkflowExecutor()
        return exec_.execute(wf, context)

    def summary(self) -> dict:
        return {
            "traces": self.trace_store.count(),
            "skills": self.skill_store.count(),
            "workflows": self.workflow_store.count(),
            "top_skills": [s.to_dict() for s in self.skill_store.top_k(3)],
        }
