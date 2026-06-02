"""
SelfDistillLoop — 継続的自己蒸留 (Sprint 25 / P6パターン).

自分が生成した出力のうち高スコアのものを教師データとしてフィルタリングし
LoRA SFT で継続的にファインチューンするセルフプレイ型成長ループ。

設計:
    DistillSample     -- 1件の蒸留サンプル (入力・出力・スコア)
    DistillDataset    -- サンプル集合 + JSONL エクスポート
    OutputFilter      -- スコア閾値フィルタ + 多様性保証
    SelfDistillCollector -- 推論→スコア→保存パイプライン
    SelfDistillLoop   -- Collect→Filter→SFT→Eval を n_rounds 自律実行

使い方::

    from open_mythos.self_distill import SelfDistillLoop, SelfDistillConfig

    loop = SelfDistillLoop(SelfDistillConfig(n_rounds=3, score_threshold=0.6))
    result = loop.run(
        prompts=["SEO対策とは？", "LLMOスコアを上げるには？"],
    )
    print(result.mean_score_improvement, result.rounds_completed)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# DistillSample
# ---------------------------------------------------------------------------


@dataclass
class DistillSample:
    """
    1件の蒸留サンプル。

    Attributes
    ----------
    prompt     : 入力プロンプト
    output     : モデル出力
    score      : LLMO等のスコア (0.0〜1.0)
    round_num  : 収集ラウンド番号
    sample_id  : 一意識別子
    metadata   : 追加情報
    """

    prompt: str
    output: str
    score: float
    round_num: int = 0
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: Dict = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "output": self.output,
            "score": self.score,
            "round": self.round_num,
            "id": self.sample_id,
        }


# ---------------------------------------------------------------------------
# DistillDataset
# ---------------------------------------------------------------------------


class DistillDataset:
    """
    蒸留サンプルのコレクション。

    Args
    ----
    max_size : 保持する最大サンプル数
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._samples: List[DistillSample] = []
        self.max_size = max_size

    def add(self, sample: DistillSample) -> None:
        self._samples.append(sample)
        if len(self._samples) > self.max_size:
            # スコア低い順に削除
            self._samples.sort(key=lambda s: -s.score)
            self._samples = self._samples[: self.max_size]

    def add_batch(self, samples: List[DistillSample]) -> None:
        for s in samples:
            self.add(s)

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(s.to_jsonl_dict(), ensure_ascii=False)
            for s in self._samples
        )

    @property
    def mean_score(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.score for s in self._samples) / len(self._samples)

    @property
    def total(self) -> int:
        return len(self._samples)

    def __len__(self) -> int:
        return self.total

    def samples_above(self, threshold: float) -> List[DistillSample]:
        return [s for s in self._samples if s.score >= threshold]


# ---------------------------------------------------------------------------
# OutputFilter
# ---------------------------------------------------------------------------


class OutputFilter:
    """
    スコア閾値フィルタ + 多様性保証フィルタ。

    Args
    ----
    score_threshold    : 保持する最低スコア
    diversity_min_len  : 多様性確保のための最短出力長 (文字数)
    max_similar_ratio  : 同一プロンプトの重複を除く類似度閾値
    """

    def __init__(
        self,
        score_threshold: float = 0.6,
        diversity_min_len: int = 10,
        max_similar_ratio: float = 0.95,
    ) -> None:
        self.score_threshold = score_threshold
        self.diversity_min_len = diversity_min_len
        self.max_similar_ratio = max_similar_ratio

    def filter(self, samples: List[DistillSample]) -> List[DistillSample]:
        """
        スコア閾値・多様性フィルタを適用してサンプルを絞り込む。

        Args:
            samples: フィルタ対象サンプルリスト

        Returns:
            フィルタ通過サンプルリスト
        """
        # スコアフィルタ
        passed = [s for s in samples if s.score >= self.score_threshold]
        # 最短長フィルタ
        passed = [s for s in passed if len(s.output) >= self.diversity_min_len]
        # 重複除去 (同一プロンプトで出力が酷似するものを除く)
        passed = self._dedup(passed)
        return passed

    def _dedup(self, samples: List[DistillSample]) -> List[DistillSample]:
        """類似度が max_similar_ratio 以上の重複を除く。"""
        seen: List[DistillSample] = []
        for s in samples:
            if not any(self._similarity(s.output, prev.output) >= self.max_similar_ratio for prev in seen):
                seen.append(s)
        return seen

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """
        Jaccard 類似度でテキストを比較する。

        英語: 空白分割、日本語: 文字 bi-gram を使用。
        split() のみでは日本語テキストが全不一致になるバグを修正。
        """
        if not a or not b:
            return 0.0

        def _tokens(text: str) -> set:
            import re
            # 英語トークン
            words = re.findall(r"[a-zA-Z0-9]+", text.lower())
            # 日本語 bi-gram (空白分割では全トークン1つになり類似度が 0 または 1 になる)
            ja = re.findall(r"[^\x00-\x7F\s]", text)
            bigrams = [ja[i] + ja[i + 1] for i in range(len(ja) - 1)]
            combined = words + bigrams
            return set(combined) if combined else {text[:20]}  # フォールバック

        set_a = _tokens(a)
        set_b = _tokens(b)
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# SFTResult (LoRA SFT のシミュレート結果)
# ---------------------------------------------------------------------------


@dataclass
class SFTResult:
    """
    SFT 実行結果 (実際の GPU 訓練をシミュレート)。

    Attributes
    ----------
    n_samples      : 訓練サンプル数
    train_loss     : 訓練ロス (シミュレート)
    eval_score     : 評価スコア (シミュレート)
    round_num      : ラウンド番号
    duration_ms    : 実行時間
    """

    n_samples: int
    train_loss: float
    eval_score: float
    round_num: int
    duration_ms: float


# ---------------------------------------------------------------------------
# SelfDistillConfig
# ---------------------------------------------------------------------------


@dataclass
class SelfDistillConfig:
    """SelfDistillLoop の設定。"""

    n_rounds: int = 3
    score_threshold: float = 0.6
    samples_per_round: int = 10
    diversity_min_len: int = 10
    early_stop_score: float = 0.85
    sft_simulate: bool = True


# ---------------------------------------------------------------------------
# SelfDistillRoundResult
# ---------------------------------------------------------------------------


@dataclass
class SelfDistillRoundResult:
    """1ラウンドの実行結果。"""

    round_num: int
    collected: int
    filtered: int
    mean_score: float
    sft_result: Optional[SFTResult]
    latency_ms: float


# ---------------------------------------------------------------------------
# SelfDistillResult
# ---------------------------------------------------------------------------


@dataclass
class SelfDistillResult:
    """SelfDistillLoop.run() の全体結果。"""

    rounds_completed: int
    round_results: List[SelfDistillRoundResult]
    dataset: DistillDataset
    initial_mean_score: float
    final_mean_score: float
    mean_score_improvement: float
    early_stopped: bool
    total_latency_ms: float

    @property
    def total_samples(self) -> int:
        return self.dataset.total


# ---------------------------------------------------------------------------
# SelfDistillCollector
# ---------------------------------------------------------------------------


class SelfDistillCollector:
    """
    推論実行 → LLMO スコア計算 → サンプル保存パイプライン。

    Args
    ----
    generate_fn : プロンプトから出力を生成する関数 (prompt: str) -> str
    score_fn    : 出力のスコアを計算する関数 (text: str) -> float
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        score_fn: Callable[[str], float],
    ) -> None:
        self._generate = generate_fn
        self._score = score_fn

    def collect(
        self,
        prompts: List[str],
        round_num: int = 0,
    ) -> List[DistillSample]:
        """
        各プロンプトに対して推論→スコアリングを実行し DistillSample リストを返す。

        Args:
            prompts   : 推論対象プロンプトリスト
            round_num : ラウンド番号

        Returns:
            DistillSample のリスト
        """
        samples: List[DistillSample] = []
        for prompt in prompts:
            try:
                output = self._generate(prompt)
                score = self._score(output)
            except Exception:  # noqa: BLE001
                output, score = "", 0.0
            samples.append(DistillSample(
                prompt=prompt,
                output=output,
                score=score,
                round_num=round_num,
            ))
        return samples


# ---------------------------------------------------------------------------
# SelfDistillLoop
# ---------------------------------------------------------------------------


class SelfDistillLoop:
    """
    Collect → Filter → SFT → Eval サイクルを自律実行するセルフ蒸留ループ。

    Args
    ----
    cfg         : SelfDistillConfig
    generate_fn : 生成関数 (省略時はエコー関数)
    score_fn    : スコア関数 (省略時は LLMO スコア)
    """

    def __init__(
        self,
        cfg: Optional[SelfDistillConfig] = None,
        generate_fn: Optional[Callable[[str], str]] = None,
        score_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        self.cfg = cfg or SelfDistillConfig()
        self._generate = generate_fn or self._default_generate
        self._score = score_fn or self._default_score
        self._collector = SelfDistillCollector(self._generate, self._score)
        self._filter = OutputFilter(
            score_threshold=self.cfg.score_threshold,
            diversity_min_len=self.cfg.diversity_min_len,
        )
        self._dataset = DistillDataset()

    def run(self, prompts: List[str]) -> SelfDistillResult:
        """
        n_rounds 回の蒸留サイクルを実行する。

        Args:
            prompts: 蒸留に使うプロンプトリスト

        Returns:
            SelfDistillResult
        """
        t_total = time.perf_counter()
        round_results: List[SelfDistillRoundResult] = []
        initial_score = self._estimate_score(prompts)
        early_stopped = False

        for round_num in range(1, self.cfg.n_rounds + 1):
            t_round = time.perf_counter()

            # Collect
            samples = self._collector.collect(prompts, round_num=round_num)

            # Filter
            filtered = self._filter.filter(samples)

            # Dataset に追加
            self._dataset.add_batch(filtered)

            mean_score = (
                sum(s.score for s in filtered) / len(filtered)
                if filtered else 0.0
            )

            # SFT シミュレート
            sft_result = None
            if self.cfg.sft_simulate and filtered:
                sft_result = self._simulate_sft(filtered, round_num)

            round_ms = (time.perf_counter() - t_round) * 1000
            round_results.append(SelfDistillRoundResult(
                round_num=round_num,
                collected=len(samples),
                filtered=len(filtered),
                mean_score=round(mean_score, 4),
                sft_result=sft_result,
                latency_ms=round(round_ms, 2),
            ))

            if mean_score >= self.cfg.early_stop_score:
                early_stopped = True
                break

        final_score = self._dataset.mean_score
        improvement = final_score - initial_score
        total_ms = (time.perf_counter() - t_total) * 1000

        return SelfDistillResult(
            rounds_completed=len(round_results),
            round_results=round_results,
            dataset=self._dataset,
            initial_mean_score=round(initial_score, 4),
            final_mean_score=round(final_score, 4),
            mean_score_improvement=round(improvement, 4),
            early_stopped=early_stopped,
            total_latency_ms=round(total_ms, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_score(self, prompts: List[str]) -> float:
        """初期スコアを推定する (最初の N 件をサンプリング)。"""
        n = min(3, len(prompts))
        if n == 0:
            return 0.0
        scores = []
        for p in prompts[:n]:
            try:
                out = self._generate(p)
                scores.append(self._score(out))
            except Exception:  # noqa: BLE001
                scores.append(0.0)
        return sum(scores) / len(scores)

    @staticmethod
    def _simulate_sft(samples: List[DistillSample], round_num: int) -> SFTResult:
        """GPU SFT をシミュレートする (実訓練なし)。"""
        n = len(samples)
        base_loss = max(0.1, 1.0 - round_num * 0.15)
        mean_score = sum(s.score for s in samples) / n
        # スコア高いほど eval_score も高い
        eval_score = min(mean_score + round_num * 0.02, 1.0)
        return SFTResult(
            n_samples=n,
            train_loss=round(base_loss, 4),
            eval_score=round(eval_score, 4),
            round_num=round_num,
            duration_ms=round(n * 10.0, 1),  # ダミー実行時間
        )

    @staticmethod
    def _default_generate(prompt: str) -> str:
        """デフォルト生成関数: プロンプトに追記してエコー。"""
        return f"{prompt} — OpenMythosによる自動生成コンテンツ。SEO最適化済み。"

    @staticmethod
    def _default_score(text: str) -> float:
        """デフォルトスコア関数: LLMO スコアを使用。"""
        from open_mythos.llmo import LLMOScorer
        return LLMOScorer().score(text).llmo_total
