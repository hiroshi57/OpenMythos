"""
DebateOrchestrator — 討議型集合知 (Sprint 20 / P1パターン).

複数エージェントが Propose → Critique → Refine → Consensus の4フェーズを
繰り返し、単独エージェントより高品質な意思決定を行う。

設計:
    DebateConfig       -- ラウンド数・エージェント数・合意閾値設定
    DebateRound        -- 1ラウンド (propose/critique/refine 各出力)
    DebateResult       -- 討議全体の結果 (consensus + agreement_score)
    ConsensusEngine    -- スコアリング合意収束アルゴリズム
    DebateOrchestrator -- 4フェーズ討議エンジン

使い方::

    from open_mythos.debate import DebateOrchestrator, DebateConfig
    from open_mythos.variants import mythos_nano
    from open_mythos.main import OpenMythos

    model = OpenMythos(mythos_nano()).eval()
    cfg = DebateConfig(n_agents=3, n_rounds=2)

    with DebateOrchestrator(model, cfg) as debate:
        result = debate.run("SEO記事の最適な構成は何か？")
        print(result.consensus)
        print(f"agreement_score: {result.agreement_score:.2f}")
        print(f"confidence: {result.confidence:.2f}")
"""

from __future__ import annotations

import math
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from open_mythos.agents import MythosAgent
from open_mythos.main import OpenMythos


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DebateConfig:
    """
    DebateOrchestrator の動作設定。

    Attributes
    ----------
    n_agents           : 討議エージェント数 (最低2)
    n_rounds           : 討議ラウンド数
    consensus_threshold: この agreement_score を超えたら早期終了 (0.0〜1.0)
    max_workers        : スレッドプールワーカー数
    agent_name_prefix  : エージェント名プレフィックス
    """

    n_agents: int = 3
    n_rounds: int = 2
    consensus_threshold: float = 0.75
    max_workers: int = 4
    agent_name_prefix: str = "debater"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DebateRound:
    """
    1討議ラウンドの出力。

    Attributes
    ----------
    round_num    : ラウンド番号 (0-indexed)
    proposals    : 各エージェントの提案 (agent_id -> text)
    critiques    : 各エージェントの批評 (agent_id -> text)
    refinements  : 各エージェントの洗練案 (agent_id -> text)
    agreement_score : このラウンドの合意スコア (0.0〜1.0)
    latency_ms   : ラウンド実行時間
    """

    round_num: int
    proposals: Dict[int, str] = field(default_factory=dict)
    critiques: Dict[int, str] = field(default_factory=dict)
    refinements: Dict[int, str] = field(default_factory=dict)
    agreement_score: float = 0.0
    latency_ms: float = 0.0


@dataclass
class DebateResult:
    """
    討議全体の最終結果。

    Attributes
    ----------
    topic          : 討議トピック
    rounds         : 各ラウンドの DebateRound リスト
    consensus      : 最終合意テキスト
    agreement_score: 最終合意スコア (0.0〜1.0)
    confidence     : 合意の信頼度 (ラウンド収束具合)
    n_rounds_used  : 実際に実行したラウンド数
    total_latency_ms: 討議全体の実行時間
    early_stopped  : 早期終了フラグ
    """

    topic: str
    rounds: List[DebateRound]
    consensus: str
    agreement_score: float
    confidence: float
    n_rounds_used: int
    total_latency_ms: float
    early_stopped: bool = False

    @property
    def improved_over_solo(self) -> bool:
        """単独エージェント比で合意スコアが閾値を超えているか。"""
        return self.agreement_score >= 0.6

    def round_scores(self) -> List[float]:
        """ラウンドごとの agreement_score リスト。"""
        return [r.agreement_score for r in self.rounds]


# ---------------------------------------------------------------------------
# ConsensusEngine
# ---------------------------------------------------------------------------


class ConsensusEngine:
    """
    複数エージェントの出力から合意スコアと代表テキストを算出する。

    アルゴリズム:
        1. 各テキストを単語集合に変換
        2. 全ペアの Jaccard 類似度を計算
        3. 平均類似度を agreement_score とする
        4. 最も平均類似度が高いテキストを代表 (consensus) とする
    """

    def score(self, texts: List[str]) -> tuple[str, float]:
        """
        テキストリストから (consensus_text, agreement_score) を返す。

        Args:
            texts: 比較対象のテキストリスト (空でない要素のみ使用)

        Returns:
            (consensus_text, agreement_score) のタプル
        """
        valid = [t.strip() for t in texts if t.strip()]
        if not valid:
            return "", 0.0
        if len(valid) == 1:
            return valid[0], 1.0

        sets = [self._to_word_set(t) for t in valid]
        n = len(sets)

        # 各テキストと他全テキストの平均 Jaccard 類似度
        avg_sims: List[float] = []
        for i in range(n):
            sims = [self._jaccard(sets[i], sets[j]) for j in range(n) if j != i]
            avg_sims.append(sum(sims) / len(sims))

        best_idx = max(range(n), key=lambda i: avg_sims[i])
        agreement = sum(avg_sims) / n
        return valid[best_idx], min(agreement, 1.0)

    @staticmethod
    def _to_word_set(text: str) -> set:
        # ASCII 単語分割を試み、得られなければ文字 bi-gram にフォールバック
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if words:
            return set(words)
        # 日本語等: 文字 bi-gram
        chars = [c for c in text if not c.isspace()]
        if len(chars) >= 2:
            return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}
        return set(chars) if chars else {"_empty_"}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def confidence(self, round_scores: List[float]) -> float:
        """
        ラウンドスコア系列から収束信頼度を計算する。

        単調増加していれば高く、振動していれば低い。
        """
        if len(round_scores) <= 1:
            return round_scores[0] if round_scores else 0.0
        deltas = [round_scores[i] - round_scores[i - 1] for i in range(1, len(round_scores))]
        positive_fraction = sum(1 for d in deltas if d >= 0) / len(deltas)
        final_score = round_scores[-1]
        return min(final_score * (0.5 + 0.5 * positive_fraction), 1.0)


# ---------------------------------------------------------------------------
# DebateOrchestrator
# ---------------------------------------------------------------------------


class DebateOrchestrator:
    """
    4フェーズ討議型集合知オーケストレーター。

    Propose → Critique → Refine → Consensus の4フェーズを n_rounds 繰り返し、
    合意スコアが consensus_threshold を超えた時点で早期終了する。

    Args
    ----
    model           : 推論に使用する OpenMythos インスタンス (全エージェント共有)
    cfg             : DebateConfig (省略時はデフォルト)
    device          : torch デバイス文字列
    system_prompt   : 全エージェント共通システムプロンプト
    max_new_tokens  : 1回の生成あたりの最大トークン数
    temperature     : サンプリング温度
    top_k           : Top-K サンプリング
    top_p           : Nucleus サンプリング
    """

    def __init__(
        self,
        model: OpenMythos,
        cfg: Optional[DebateConfig] = None,
        *,
        device: str = "cpu",
        system_prompt: str = "",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> None:
        self._model = model
        self.cfg = cfg or DebateConfig()
        n = max(2, self.cfg.n_agents)
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
            for i in range(n)
        ]
        self._pool = ThreadPoolExecutor(max_workers=self.cfg.max_workers)
        self._consensus_engine = ConsensusEngine()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DebateOrchestrator":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> DebateResult:
        """
        トピックについて討議を実行し DebateResult を返す。

        Args:
            topic: 討議するトピック/質問

        Returns:
            DebateResult
        """
        t_total = time.perf_counter()
        rounds: List[DebateRound] = []
        current_context = topic

        for round_num in range(self.cfg.n_rounds):
            round_result = self._run_round(round_num, current_context)
            rounds.append(round_result)

            # 洗練案を次ラウンドのコンテキストに使用
            if round_result.refinements:
                best_refinement, _ = self._consensus_engine.score(
                    list(round_result.refinements.values())
                )
                if best_refinement:
                    current_context = f"{topic}\n\n前ラウンドの洗練案:\n{best_refinement}"

            if round_result.agreement_score >= self.cfg.consensus_threshold:
                rounds_used = round_num + 1
                early_stopped = True
                break
        else:
            rounds_used = self.cfg.n_rounds
            early_stopped = False

        # 最終合意を最終ラウンドの洗練案から生成
        last_round = rounds[-1]
        final_texts = list(last_round.refinements.values()) or list(last_round.proposals.values())
        consensus, agreement_score = self._consensus_engine.score(final_texts)

        round_scores = [r.agreement_score for r in rounds]
        confidence = self._consensus_engine.confidence(round_scores)

        total_ms = (time.perf_counter() - t_total) * 1000

        for agent in self._agents:
            agent.reset()

        return DebateResult(
            topic=topic,
            rounds=rounds,
            consensus=consensus,
            agreement_score=agreement_score,
            confidence=confidence,
            n_rounds_used=rounds_used,
            total_latency_ms=round(total_ms, 2),
            early_stopped=early_stopped,
        )

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int, context: str) -> DebateRound:
        """1討議ラウンドを実行する。"""
        t_start = time.perf_counter()
        round_result = DebateRound(round_num=round_num)

        # Phase 1: Propose — 全エージェントが並列で提案生成
        proposals = self._run_phase_parallel(
            context,
            prompt_prefix="提案:",
            phase_name="propose",
        )
        round_result.proposals = proposals

        # Phase 2: Critique — 他エージェントの提案を批評
        all_proposals_text = "\n".join(
            f"エージェント{i}の提案: {p}" for i, p in proposals.items()
        )
        critique_context = f"{context}\n\n{all_proposals_text}"
        critiques = self._run_phase_parallel(
            critique_context,
            prompt_prefix="批評:",
            phase_name="critique",
        )
        round_result.critiques = critiques

        # Phase 3: Refine — 批評を踏まえて提案を洗練
        all_critiques_text = "\n".join(
            f"エージェント{i}の批評: {c}" for i, c in critiques.items()
        )
        refine_context = f"{critique_context}\n\n{all_critiques_text}"
        refinements = self._run_phase_parallel(
            refine_context,
            prompt_prefix="洗練案:",
            phase_name="refine",
        )
        round_result.refinements = refinements

        # Phase 4: Consensus score 計算
        _, agreement_score = self._consensus_engine.score(list(refinements.values()))
        round_result.agreement_score = agreement_score
        round_result.latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return round_result

    def _run_phase_parallel(
        self,
        context: str,
        prompt_prefix: str,
        phase_name: str,
    ) -> Dict[int, str]:
        """全エージェントを並列でフェーズ実行し、結果を辞書で返す。"""
        future_to_id: Dict[Future, int] = {}
        for i, agent in enumerate(self._agents):
            prompt = f"{context}\n\n{prompt_prefix}"
            f = self._pool.submit(self._call_agent, agent, i, prompt)
            future_to_id[f] = i

        results: Dict[int, str] = {}
        for f in as_completed(future_to_id):
            agent_id = future_to_id[f]
            try:
                results[agent_id] = f.result()
            except Exception:  # noqa: BLE001
                results[agent_id] = ""
        return results

    @staticmethod
    def _call_agent(agent: MythosAgent, agent_id: int, prompt: str) -> str:
        """エージェントを呼び出し、テキストを返す。例外は空文字で握りつぶす。"""
        try:
            return agent.run(prompt)
        except Exception:  # noqa: BLE001
            return ""
