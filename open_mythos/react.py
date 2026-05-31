"""
OpenMythos ReAct (Reasoning + Acting) エージェントループ。

ReAct フレームワーク (Yao et al., 2022) のオープン実装。
モデルが自律的に「考える → ツールを使う → 観察する」を繰り返し、
複雑なタスクを段階的に解決する。

設計:
    AgentStep    -- 1ステップの思考/行動/観察コンテナ
    AgentResult  -- エージェント実行の最終結果
    ReActAgent   -- Think → Act → Observe ループエンジン

使い方::

    from open_mythos.react import ReActAgent
    from open_mythos.tools import ToolRegistry

    model = OpenMythos(cfg)
    registry = ToolRegistry.default()  # マーケ特化4ツール
    agent = ReActAgent(model, registry, max_iterations=5)

    result = agent.run(
        task="Jasper AI の広告費を調べて、ROI 200% になる予算を計算して",
    )
    print(result.final_answer)
    for step in result.steps:
        print(f"  [{step.step_type}] {step.content[:80]}")
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from open_mythos.tools import (
    ToolRegistry,
    ToolCall,
    ToolResult,
    execute_tool_call,
    parse_tool_calls,
    build_tool_prompt,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """
    ReAct エージェントの1ステップ。

    step_type は次のいずれか:
        "thought"     -- モデルの思考 (Reason フェーズ)
        "action"      -- ツール呼び出し要求 (Act フェーズ)
        "observation" -- ツール実行結果 (Observe フェーズ)
        "answer"      -- 最終回答
    """

    step_type: str
    content: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    iteration: int = 0
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    """ReAct エージェント実行の最終結果。"""

    task: str
    """入力タスク。"""

    final_answer: str
    """エージェントが生成した最終回答。"""

    steps: list[AgentStep]
    """実行ステップの履歴。"""

    iterations_used: int
    """使用したイテレーション数。"""

    total_latency_ms: float
    """合計実行時間 (ms)。"""

    stopped_reason: str = "completed"
    """停止理由: "completed" / "max_iterations" / "no_tools_needed" / "error" """

    @property
    def thought_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.step_type == "thought"]

    @property
    def action_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.step_type == "action"]

    @property
    def observation_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.step_type == "observation"]

    @property
    def n_tool_calls(self) -> int:
        return len(self.action_steps)


# ---------------------------------------------------------------------------
# ReActAgent
# ---------------------------------------------------------------------------


class ReActAgent:
    """
    ReAct (Reasoning + Acting) エージェントループ。

    モデルが生成したテキストを解析し、ツール呼び出しを検出して実行する。
    ツールが不要と判断したとき、または最大イテレーションに達したときに停止する。

    Args:
        model          -- OpenMythos モデルインスタンス
        registry       -- ToolRegistry (利用可能なツール)
        max_iterations -- 最大 Think-Act-Observe ループ回数 (デフォルト: 6)
        max_new_tokens -- 各ステップの最大生成トークン数
        temperature    -- サンプリング温度
        loops          -- モデルの decode_loops 数
        device         -- torch device
        stop_tokens    -- 停止トークン文字列のリスト
    """

    _FINAL_ANSWER_PATTERN = re.compile(
        r'(?:Final Answer|最終回答|Answer|回答)\s*[:：]\s*(.+)',
        re.IGNORECASE | re.DOTALL,
    )
    _THOUGHT_PATTERN = re.compile(
        r'(?:Thought|思考|考え)\s*[:：]\s*(.+?)(?=(?:Action|ツール|Final|$))',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        model: "OpenMythos",
        registry: ToolRegistry,
        max_iterations: int = 6,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        loops: int = 4,
        device: str = "cpu",
        stop_tokens: Optional[list[str]] = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.max_iterations = max_iterations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.loops = loops
        self.device = device
        self.stop_tokens = stop_tokens or ["Final Answer:", "最終回答:", "Answer:"]

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

    def run(self, task: str, system_prompt: str = "") -> AgentResult:
        """
        タスクを ReAct ループで解決する。

        Args:
            task          -- ユーザーのタスク/質問
            system_prompt -- カスタムシステムプロンプト (省略可)

        Returns:
            AgentResult
        """
        t_total = time.perf_counter()
        steps: list[AgentStep] = []

        # システムプロンプト構築
        tool_prompt = build_tool_prompt(self.registry.list_tools())
        base_system = system_prompt or (
            "あなたはタスクを段階的に解決するエージェントです。\n"
            "Thought: で考えを述べ、ツールが必要なら <tool_call> で呼び出してください。\n"
            "最終回答は 'Final Answer: <回答>' の形式で出力してください。\n\n"
        )
        full_system = base_system + "\n\n" + tool_prompt

        # コンテキスト初期化
        context = full_system + f"\n\nTask: {task}\n\n"

        final_answer = ""
        stopped_reason = "completed"
        iteration = -1

        for iteration in range(self.max_iterations):
            t_step = time.perf_counter()

            # --- Think: モデルに思考させる ---
            thought_text = self._generate(context + "Thought: ", max_new_tokens=self.max_new_tokens)

            step_latency = (time.perf_counter() - t_step) * 1000
            thought_step = AgentStep(
                step_type="thought",
                content=thought_text,
                iteration=iteration,
                latency_ms=step_latency,
            )
            steps.append(thought_step)

            # 最終回答チェック
            answer_match = self._FINAL_ANSWER_PATTERN.search(thought_text)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                steps.append(AgentStep(
                    step_type="answer",
                    content=final_answer,
                    iteration=iteration,
                ))
                stopped_reason = "completed"
                break

            stop_found = any(tok.lower() in thought_text.lower() for tok in self.stop_tokens)
            if stop_found:
                final_answer = thought_text.split(":")[-1].strip() if ":" in thought_text else thought_text
                steps.append(AgentStep(step_type="answer", content=final_answer, iteration=iteration))
                stopped_reason = "completed"
                break

            # --- Act: ツール呼び出しを検出 ---
            tool_calls = parse_tool_calls(thought_text)

            if not tool_calls:
                # ツール不要 → 直接回答とみなす
                final_answer = thought_text.strip()
                steps.append(AgentStep(step_type="answer", content=final_answer, iteration=iteration))
                stopped_reason = "no_tools_needed"
                break

            # ツール呼び出しを実行
            for tc in tool_calls:
                t_act = time.perf_counter()
                action_step = AgentStep(
                    step_type="action",
                    content=f"Calling tool: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})",
                    tool_call=tc,
                    iteration=iteration,
                    latency_ms=0.0,
                )
                steps.append(action_step)

                # --- Observe: ツール実行 ---
                result: ToolResult = execute_tool_call(tc, self.registry)
                obs_latency = (time.perf_counter() - t_act) * 1000

                obs_content = (
                    json.dumps(result.content, ensure_ascii=False, indent=2)
                    if result.success and result.content is not None
                    else f"Error: {result.error}"
                )
                obs_step = AgentStep(
                    step_type="observation",
                    content=f"Tool '{tc.name}' returned:\n{obs_content}",
                    tool_result=result,
                    iteration=iteration,
                    latency_ms=round(obs_latency, 2),
                )
                steps.append(obs_step)

                # コンテキストに観察を追加
                context += (
                    f"Thought: {thought_text}\n"
                    f"Action: {action_step.content}\n"
                    f"Observation: {obs_content}\n\n"
                )
        else:
            # max_iterations に達した
            stopped_reason = "max_iterations"
            if not final_answer:
                final_answer = "最大イテレーション数に達しました。途中結果を参照してください。"
                steps.append(AgentStep(step_type="answer", content=final_answer, iteration=self.max_iterations - 1))

        total_ms = (time.perf_counter() - t_total) * 1000

        return AgentResult(
            task=task,
            final_answer=final_answer,
            steps=steps,
            iterations_used=min(iteration + 1, self.max_iterations) if iteration >= 0 else 0,
            total_latency_ms=round(total_ms, 2),
            stopped_reason=stopped_reason,
        )

    # ------------------------------------------------------------------
    # 内部生成
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """プロンプトからテキストを生成する。"""
        vsize = self.model.cfg.vocab_size
        max_prompt = max(1, self.model.cfg.max_seq_len - max_new_tokens - 4)

        # 文字レベルトークナイズ (テスト/デモ用)
        ids = [ord(c) % vsize for c in prompt[:max_prompt]]
        if not ids:
            ids = [0]

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        cur = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(cur, n_loops=self.loops)
                next_logits = logits[0, -1, :] / max(self.temperature, 1e-8)
                probs = F.softmax(next_logits, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())
                generated.append(next_tok)
                cur = torch.cat([cur, torch.tensor([[next_tok]], device=self.device)], dim=1)
                if next_tok == vsize - 1:
                    break

        chars = []
        for i in generated:
            try:
                c = chr(i % 128)
                if c.isprintable() or c in "\n\t ":
                    chars.append(c)
            except (ValueError, OverflowError):
                pass
        return "".join(chars)


# ---------------------------------------------------------------------------
# ユーティリティ: エージェント結果のフォーマット
# ---------------------------------------------------------------------------


def format_agent_trace(result: AgentResult, max_content_len: int = 120) -> str:
    """
    AgentResult をデバッグ用テキストに整形する。

    Args:
        result          -- AgentResult
        max_content_len -- 各ステップのコンテンツ最大長

    Returns:
        整形されたテキスト
    """
    lines = [
        f"=== ReAct Agent Trace ===",
        f"Task: {result.task}",
        f"Iterations: {result.iterations_used} / stopped: {result.stopped_reason}",
        f"Tool calls: {result.n_tool_calls}",
        f"Total time: {result.total_latency_ms:.1f}ms",
        "",
    ]

    for step in result.steps:
        icon = {"thought": "💭", "action": "🔧", "observation": "👁", "answer": "✅"}.get(step.step_type, "•")
        content = step.content[:max_content_len]
        if len(step.content) > max_content_len:
            content += "..."
        lines.append(f"  [{step.iteration}] {icon} {step.step_type.upper()}: {content}")

    lines.append(f"\nFinal Answer: {result.final_answer}")
    return "\n".join(lines)
