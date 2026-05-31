"""
SwarmOrchestrator — parallel multi-agent coordination for OpenMythos.

設計思想
--------
複数の ``MythosAgent`` インスタンスが同一モデルを共有しながら
スレッドプールで並列実行する。各エージェントは独立した会話履歴と
呼び出しごとに新規生成される KV キャッシュを持つため、
推論は本質的にステートレスかつスレッドセーフ。

Strategies
----------
map       : N タスク → N エージェントに割り当てて並列実行
broadcast : 1 タスク → 全エージェントに並列送信し全応答を収集
pipeline  : 1 タスク → N エージェントを直列でリファイン（前段出力を次段入力）
vote      : broadcast + 多数決で最も多い応答を選択

Usage::

    from open_mythos import SwarmOrchestrator, SwarmConfig
    from open_mythos.variants import mythos_nano
    from open_mythos.main import OpenMythos

    model = OpenMythos(mythos_nano()).eval()
    cfg = SwarmConfig(n_agents=4, strategy="vote")

    with SwarmOrchestrator(model, cfg, max_new_tokens=64) as swarm:
        result = swarm.run("SEO記事の構成を提案してください")
        print(result.final_output)
        print(f"成功率: {result.success_rate:.0%}")
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from open_mythos.agents import MythosAgent
from open_mythos.main import OpenMythos


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SwarmConfig:
    """
    SwarmOrchestrator の動作設定。

    Attributes
    ----------
    n_agents           : 生成するエージェント数（モデルは共有）
    max_workers        : スレッドプールのワーカー数（通常 n_agents と同じでよい）
    strategy           : デフォルト実行戦略
                         ``"map"`` | ``"broadcast"`` | ``"pipeline"`` | ``"vote"``
    agent_name_prefix  : 各エージェント名の prefix（``"agent_0"`` … ``"agent_N-1"``）
    """

    n_agents: int = 4
    max_workers: int = 4
    strategy: str = "broadcast"
    agent_name_prefix: str = "agent"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SwarmAgentResult:
    """
    スワーム実行における 1 エージェントの結果。

    Attributes
    ----------
    task_id    : この結果が対応するタスクの識別子
    agent_id   : ゼロベースのエージェント番号
    agent_name : エージェント名文字列
    output     : 生成テキスト（エラー時は空文字列）
    latency_ms : エージェント呼び出しの所要時間（ミリ秒）
    error      : エラーメッセージ（正常時は None）
    """

    task_id: str
    agent_id: int
    agent_name: str
    output: str
    latency_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """エラーなく完了した場合 True。"""
        return self.error is None


@dataclass
class SwarmResult:
    """
    スワーム実行の集約結果。

    Attributes
    ----------
    strategy          : 使用した実行戦略
    results           : 各エージェントの ``SwarmAgentResult`` リスト
    final_output      : 戦略に応じた最終出力テキスト
    total_latency_ms  : スワーム全体の所要時間（ミリ秒）
    """

    strategy: str
    results: List[SwarmAgentResult]
    final_output: str
    total_latency_ms: float

    @property
    def n_agents(self) -> int:
        """実行されたエージェント数。"""
        return len(self.results)

    @property
    def n_successful(self) -> int:
        """エラーなく完了したエージェント数。"""
        return sum(1 for r in self.results if r.ok)

    @property
    def success_rate(self) -> float:
        """成功率 (0.0 – 1.0)。"""
        if not self.results:
            return 0.0
        return self.n_successful / len(self.results)

    def agent_outputs(self) -> List[str]:
        """成功したエージェントの出力テキスト一覧。"""
        return [r.output for r in self.results if r.ok]


# ---------------------------------------------------------------------------
# SwarmOrchestrator
# ---------------------------------------------------------------------------


class SwarmOrchestrator:
    """
    OpenMythos 用並列マルチエージェントオーケストレーター。

    同一モデルを共有する複数の ``MythosAgent`` をスレッドプールで
    並列実行する。各推論呼び出しは独立した KV キャッシュを生成するため
    本質的にスレッドセーフ。

    コンテキストマネージャとして利用するとスレッドプールが自動的にクリーン
    アップされる::

        with SwarmOrchestrator(model, cfg) as swarm:
            result = swarm.run("タスク入力")

    Args
    ----
    model           : 推論に使用する ``OpenMythos`` インスタンス（全エージェント共有）
    cfg             : ``SwarmConfig``（省略時はデフォルト値）
    device          : torch デバイス文字列
    system_prompt   : 全エージェント共通のシステムプロンプト
    max_new_tokens  : 1 呼び出しあたりの最大生成トークン数
    temperature     : サンプリング温度
    top_k           : Top-K サンプリング（0 で無効）
    top_p           : Nucleus サンプリング閾値
    """

    def __init__(
        self,
        model: OpenMythos,
        cfg: Optional[SwarmConfig] = None,
        *,
        device: str = "cpu",
        system_prompt: str = "",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> None:
        self._model = model
        self.cfg = cfg or SwarmConfig()
        self._agents: List[MythosAgent] = [
            MythosAgent(
                model=model,
                device=device,
                agent_name=f"{self.cfg.agent_name_prefix}_{i}",
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            for i in range(self.cfg.n_agents)
        ]
        self._pool = ThreadPoolExecutor(max_workers=self.cfg.max_workers)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SwarmOrchestrator":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """スレッドプールをシャットダウンする。"""
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        agent_id: int,
        task_id: str,
        task_input: str,
        reset_after: bool = True,
    ) -> SwarmAgentResult:
        """
        エージェントを 1 回呼び出し、結果を ``SwarmAgentResult`` で返す。

        例外は握りつぶして ``error`` フィールドに記録するため、スレッドプール
        内の他のタスクへの影響を防ぐ。

        Args:
            agent_id    : エージェントインデックス
            task_id     : 結果に付与する識別子
            task_input  : エージェントへの入力テキスト
            reset_after : 呼び出し後にエージェント履歴をリセットするか
        """
        agent = self._agents[agent_id]
        t0 = time.perf_counter()
        try:
            output = agent.run(task_input)
            error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001
            output = ""
            error = str(exc)
        finally:
            if reset_after:
                agent.reset()
        latency_ms = (time.perf_counter() - t0) * 1000
        return SwarmAgentResult(
            task_id=task_id,
            agent_id=agent_id,
            agent_name=agent.agent_name,
            output=output,
            latency_ms=latency_ms,
            error=error,
        )

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def map(self, tasks: List[str]) -> SwarmResult:
        """
        N タスクを N エージェントに並列割り当てして実行する。

        タスク数がエージェント数を超える場合はラウンドロビンで割り当てる。
        結果は入力タスクと同じ順序で返る。

        Args:
            tasks : 独立したタスク文字列のリスト

        Returns:
            ``final_output`` は各タスクの出力を改行結合した文字列。
        """
        t_start = time.perf_counter()
        n = len(tasks)
        future_to_idx: Dict[Future, int] = {}
        ordered: List[Optional[SwarmAgentResult]] = [None] * n

        for i, task in enumerate(tasks):
            agent_id = i % len(self._agents)
            f = self._pool.submit(self._run_agent, agent_id, f"map_{i}", task, True)
            future_to_idx[f] = i

        for f in as_completed(future_to_idx):
            ordered[future_to_idx[f]] = f.result()

        results = [r for r in ordered if r is not None]
        total_ms = (time.perf_counter() - t_start) * 1000
        return SwarmResult(
            strategy="map",
            results=results,
            final_output="\n".join(r.output for r in results),
            total_latency_ms=total_ms,
        )

    def broadcast(self, task: str) -> SwarmResult:
        """
        同一タスクを全エージェントに並列送信し、全応答を収集する。

        各エージェントは独立して応答を生成するため、多様な出力が得られる。
        呼び出し後は全エージェント履歴をリセットする。

        Args:
            task : 全エージェントに送る入力テキスト

        Returns:
            ``final_output`` は全エージェントの出力を改行結合した文字列。
        """
        t_start = time.perf_counter()
        future_to_id: Dict[Future, int] = {}

        for i in range(len(self._agents)):
            f = self._pool.submit(
                self._run_agent, i, f"broadcast_{i}", task, True
            )
            future_to_id[f] = i

        ordered: List[Optional[SwarmAgentResult]] = [None] * len(self._agents)
        for f in as_completed(future_to_id):
            ordered[future_to_id[f]] = f.result()

        results = [r for r in ordered if r is not None]
        total_ms = (time.perf_counter() - t_start) * 1000
        return SwarmResult(
            strategy="broadcast",
            results=results,
            final_output="\n".join(r.output for r in results),
            total_latency_ms=total_ms,
        )

    def pipeline(
        self,
        task: str,
        stages: Optional[List[str]] = None,
    ) -> SwarmResult:
        """
        直列 N 段パイプライン。各エージェントが前段の出力をリファインする。

        ``stages`` が指定された場合、各段の先頭に対応する指示文を付加して
        エージェントに渡す（例: ``["要約:", "翻訳:", "校正:"]``）。
        段数がエージェント数より少ない場合は余った段にデフォルトシステム
        プロンプトが適用される。

        全パイプライン完了後、全エージェントの履歴はリセットされる。

        Args:
            task   : 最初の段への入力テキスト
            stages : オプションの段ごとの指示プレフィックスリスト

        Returns:
            ``final_output`` は最後に成功したエージェントの出力。
        """
        t_start = time.perf_counter()
        results: List[SwarmAgentResult] = []
        current_input = task

        for i, agent in enumerate(self._agents):
            stage_input = current_input
            if stages and i < len(stages):
                stage_input = f"{stages[i]}\n\n{current_input}"
            result = self._run_agent(
                i, f"stage_{i}", stage_input, reset_after=False
            )
            results.append(result)
            if result.ok and result.output:
                current_input = result.output

        # パイプライン完了後に全エージェントの履歴をリセット
        for agent in self._agents:
            agent.reset()

        total_ms = (time.perf_counter() - t_start) * 1000
        return SwarmResult(
            strategy="pipeline",
            results=results,
            final_output=current_input,
            total_latency_ms=total_ms,
        )

    def vote(self, task: str) -> SwarmResult:
        """
        多数決集約。全エージェントに同一タスクを broadcast し
        最も多い出力テキストを ``final_output`` として返す。

        同数の場合は出力長が中央値に最も近い候補を選ぶ。

        Args:
            task : 全エージェントに送る入力テキスト

        Returns:
            ``final_output`` は多数決勝者。
        """
        broadcast_result = self.broadcast(task)
        outputs = broadcast_result.agent_outputs()

        if not outputs:
            winner = ""
        else:
            counts = Counter(outputs)
            max_count = max(counts.values())
            candidates = [o for o, c in counts.items() if c == max_count]
            if len(candidates) == 1:
                winner = candidates[0]
            else:
                # タイブレーク: 中央値に最も近い長さの候補を選ぶ
                med = sorted(len(o) for o in outputs)[len(outputs) // 2]
                winner = min(candidates, key=lambda o: abs(len(o) - med))

        return SwarmResult(
            strategy="vote",
            results=broadcast_result.results,
            final_output=winner,
            total_latency_ms=broadcast_result.total_latency_ms,
        )

    def run(
        self,
        task: Union[str, List[str]],
        strategy: Optional[str] = None,
        stages: Optional[List[str]] = None,
    ) -> SwarmResult:
        """
        統一エントリポイント — 戦略に応じてディスパッチ。

        ``task`` がリストの場合は戦略によらず ``map`` が使われる。
        ``strategy`` 省略時は ``cfg.strategy`` が使われる。

        Args:
            task     : 単一の入力文字列またはリスト
            strategy : 戦略オーバーライド
                       (``"map"`` / ``"broadcast"`` / ``"pipeline"`` / ``"vote"``)
            stages   : ``pipeline`` 戦略時のステージ指示リスト

        Returns:
            SwarmResult
        """
        if isinstance(task, list):
            return self.map(task)
        s = strategy or self.cfg.strategy
        if s == "pipeline":
            return self.pipeline(task, stages=stages)
        if s == "vote":
            return self.vote(task)
        if s == "map":
            return self.map([task])
        return self.broadcast(task)
