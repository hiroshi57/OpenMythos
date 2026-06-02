"""
PromptEvolution — 適応型プロンプト進化 (Sprint 28 / P9パターン).

遺伝的アルゴリズム (GA) でプロンプトを世代ごとに進化させ、
LLMO スコア (または任意フィットネス関数) を自動最大化する。

設計:
    PromptGene       -- 1つのプロンプト個体 (text + fitness)
    EvolutionConfig  -- 集団サイズ・突然変異率・世代数設定
    EvolutionRound   -- 1世代の進化結果スナップショット
    EvolutionResult  -- 全世代を通じた進化結果
    PromptEvolution  -- 遺伝的アルゴリズムエンジン

遺伝演算子:
    - Selection   : トーナメント選択 (tournament_size=3)
    - Crossover   : 文レベル交叉 (親 A / 親 B の文を交互に組み合わせ)
    - Mutation    : キーワード置換 / 文挿入 / 文削除

精度向上のポイント:
    - 初期集団にドメイン知識テンプレートを混入させる
    - MistakeGuard (Sprint 24) で生成した個体を事前フィルタリング
    - エリート保存 (最良 top_k を次世代に引き継ぐ)
    - 早期収束防止: 世代内 diversity が低い場合に突然変異率を増加

他スプリントとの連携:
    - LLMOScorer (Sprint 10/19): デフォルトフィットネス関数
    - EnsembleScorer (Sprint 27): より精度の高いフィットネスに差し替え可能
    - MistakeGuard (Sprint 24): 危険なプロンプト変異体を除去
    - SelfDistillLoop (Sprint 25): 進化済みプロンプトで蒸留データを生成

使い方::

    from open_mythos.prompt_evolution import PromptEvolution, EvolutionConfig

    config = EvolutionConfig(
        population_size=10,
        n_generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_size=2,
    )
    evo = PromptEvolution(config)
    result = evo.evolve(
        seed_prompt="SEO記事を書いてください",
        topic_keywords=["LLMO", "検索", "最適化"],
    )
    print(result.best_gene.text)
    print(f"fitness: {result.best_gene.fitness:.3f}")
"""

from __future__ import annotations

import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# デフォルトフィットネス関数 (LLMO ライト版)
# ---------------------------------------------------------------------------


def _default_fitness(text: str) -> float:
    """LLMO 簡易スコア: entity_density + directness + structure の平均。"""
    words = text.split()
    n_words = len(words) or 1
    # entity
    entities = len(re.findall(r"[A-Z][A-Z0-9]{1,}|\d+(?:\.\d+)?%?|[一-龯]{2,}", text))
    ed = min(1.0, entities / max(n_words * 0.1, 1))
    # directness: 最初の文が 5〜20 語
    first = re.split(r"[。.!?！？\n]", text.strip())[0].split()
    dr = min(1.0, len(first) / 15) if 3 <= len(first) <= 20 else 0.3
    # structure
    has_list = bool(re.search(r"[-•・\d]\s", text))
    struct = 0.7 if has_list else 0.3
    return (ed + dr + struct) / 3


# ---------------------------------------------------------------------------
# PromptGene
# ---------------------------------------------------------------------------


@dataclass
class PromptGene:
    """
    1つのプロンプト個体。

    Attributes
    ----------
    text       : プロンプト本文
    fitness    : フィットネス値 (高いほど良い)
    generation : 生成された世代番号
    gene_id    : 一意ID
    parents    : 親の gene_id リスト (交叉由来の場合)
    """

    text: str
    fitness: float = 0.0
    generation: int = 0
    gene_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parents: List[str] = field(default_factory=list)

    def __lt__(self, other: "PromptGene") -> bool:
        return self.fitness < other.fitness


# ---------------------------------------------------------------------------
# EvolutionConfig
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """
    遺伝的アルゴリズムの設定。

    Attributes
    ----------
    population_size    : 集団サイズ
    n_generations      : 世代数
    mutation_rate      : 突然変異率 (0〜1)
    crossover_rate     : 交叉率 (0〜1)
    elite_size         : 次世代に引き継ぐエリート数
    tournament_size    : トーナメント選択の参加数
    diversity_threshold: 世代内多様性がこれを下回ると突然変異率を増加
    early_stop_delta   : 連続 N 世代で改善量がこれ未満なら停止
    early_stop_patience: 改善なし世代の許容回数
    """

    population_size: int = 8
    n_generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    diversity_threshold: float = 0.3
    early_stop_delta: float = 0.005
    early_stop_patience: int = 3


# ---------------------------------------------------------------------------
# EvolutionRound
# ---------------------------------------------------------------------------


@dataclass
class EvolutionRound:
    """
    1世代の進化結果スナップショット。

    Attributes
    ----------
    generation   : 世代番号 (0=初期集団)
    best_fitness : 最良フィットネス
    mean_fitness : 平均フィットネス
    diversity    : 集団内のテキスト多様性 (0〜1)
    best_text    : 最良プロンプト本文
    population_size : 集団サイズ
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float
    best_text: str
    population_size: int


# ---------------------------------------------------------------------------
# EvolutionResult
# ---------------------------------------------------------------------------


@dataclass
class EvolutionResult:
    """
    全世代を通じた進化結果。

    Attributes
    ----------
    best_gene      : 全世代中の最良個体
    rounds         : 各世代のスナップショット
    n_generations_run: 実際に実行した世代数 (早期終了含む)
    fitness_history: 世代ごとの最良フィットネス推移
    improvement    : 初期集団 → 最終世代の改善量
    converged      : 早期収束フラグ
    """

    best_gene: PromptGene
    rounds: List[EvolutionRound]
    n_generations_run: int
    fitness_history: List[float]
    improvement: float
    converged: bool

    @property
    def best_prompt(self) -> str:
        return self.best_gene.text

    def summary(self) -> str:
        lines = [
            f"PromptEvolution: {self.n_generations_run} generations, "
            f"improvement={self.improvement:+.3f}, converged={self.converged}",
            f"best fitness: {self.best_gene.fitness:.3f}",
            f"best prompt: {self.best_gene.text[:120]}...",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PromptEvolution
# ---------------------------------------------------------------------------


class PromptEvolution:
    """
    遺伝的アルゴリズムでプロンプトを自動最適化するエンジン。

    Parameters
    ----------
    config       : EvolutionConfig
    fitness_fn   : (text: str) -> float の評価関数。
                   None の場合は内部 LLMO スコアを使用
    guard_fn     : (text: str) -> bool の安全フィルタ。
                   True を返す個体は除去 (MistakeGuard.check() を渡す等)
    seed         : 乱数シード (再現性のため)
    """

    # 変異テンプレート (カテゴリ: 追加語句)
    _MUTATION_POOL = {
        "quality": ["詳細に", "具体的に", "step-by-step で", "専門用語を用いて",
                    "例を挙げながら", "構造的に", "箇条書きで"],
        "audience": ["初心者向けに", "専門家向けに", "マーケター向けに",
                     "エンジニア向けに", "経営者向けに"],
        "format": ["見出しを付けて", "リスト形式で", "表形式で",
                   "FAQ形式で", "結論→理由→例の順で"],
        "length": ["簡潔に200字以内で", "500字以上で詳しく", "1000字程度で"],
    }

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_fn: Optional[Callable[[str], float]] = None,
        guard_fn: Optional[Callable[[str], bool]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self.fitness_fn = fitness_fn or _default_fitness
        self.guard_fn = guard_fn
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ evolve

    def evolve(
        self,
        seed_prompt: str,
        topic_keywords: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
    ) -> EvolutionResult:
        """
        プロンプトを進化させる。

        Args:
            seed_prompt     : 初期プロンプト (進化の出発点)
            topic_keywords  : ドメインキーワード (変異に利用)
            templates       : 追加初期個体となるテンプレート群

        Returns:
            EvolutionResult
        """
        cfg = self.config
        keywords = topic_keywords or []

        # ---- 初期集団生成 ----
        population = self._init_population(seed_prompt, templates, keywords)
        population = self._evaluate_population(population)

        rounds: List[EvolutionRound] = []
        fitness_history: List[float] = []
        global_best = max(population, key=lambda g: g.fitness)
        no_improve_count = 0
        converged = False

        # 世代 0 の記録
        rounds.append(self._snapshot(population, 0))
        fitness_history.append(rounds[0].best_fitness)

        for gen in range(1, cfg.n_generations + 1):
            # ---- 多様性チェック → 適応的突然変異 ----
            diversity = self._diversity(population)
            effective_mutation = cfg.mutation_rate
            if diversity < cfg.diversity_threshold:
                effective_mutation = min(0.8, cfg.mutation_rate * 2)

            # ---- エリート保存 ----
            elites = sorted(population, key=lambda g: g.fitness, reverse=True)[: cfg.elite_size]

            # ---- 次世代生成 ----
            next_pop: List[PromptGene] = list(elites)
            while len(next_pop) < cfg.population_size:
                p1 = self._tournament_select(population)
                if self._rng.random() < cfg.crossover_rate:
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2, gen)
                else:
                    child = PromptGene(text=p1.text, generation=gen, parents=[p1.gene_id])

                if self._rng.random() < effective_mutation:
                    child = self._mutate(child, keywords, gen)

                # 安全フィルタ
                if self.guard_fn and self.guard_fn(child.text):
                    continue  # 危険な個体を除去
                next_pop.append(child)

            population = self._evaluate_population(next_pop)
            gen_best = max(population, key=lambda g: g.fitness)

            # グローバル最良更新
            if gen_best.fitness > global_best.fitness + cfg.early_stop_delta:
                global_best = gen_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            snap = self._snapshot(population, gen)
            rounds.append(snap)
            fitness_history.append(snap.best_fitness)

            # 早期終了
            if no_improve_count >= cfg.early_stop_patience:
                converged = True
                break

        initial_fitness = fitness_history[0]
        improvement = global_best.fitness - initial_fitness

        return EvolutionResult(
            best_gene=global_best,
            rounds=rounds,
            n_generations_run=len(rounds) - 1,
            fitness_history=fitness_history,
            improvement=improvement,
            converged=converged,
        )

    # ----------------------------------------------------------------- private

    def _init_population(
        self,
        seed: str,
        templates: Optional[List[str]],
        keywords: List[str],
    ) -> List[PromptGene]:
        """初期集団を生成する。seed + テンプレート + ランダム変異。"""
        pop: List[PromptGene] = [PromptGene(text=seed, generation=0)]

        # テンプレート追加
        if templates:
            for t in templates[: self.config.population_size - 1]:
                pop.append(PromptGene(text=t, generation=0))

        # 足りない分は seed を変異させて補充
        while len(pop) < self.config.population_size:
            g = PromptGene(text=seed, generation=0)
            mutated = self._mutate(g, keywords, 0)
            pop.append(mutated)

        return pop

    def _evaluate_population(self, population: List[PromptGene]) -> List[PromptGene]:
        for g in population:
            g.fitness = round(self.fitness_fn(g.text), 4)
        return population

    def _tournament_select(self, population: List[PromptGene]) -> PromptGene:
        """トーナメント選択。"""
        k = min(self.config.tournament_size, len(population))
        contestants = self._rng.sample(population, k)
        return max(contestants, key=lambda g: g.fitness)

    def _crossover(
        self, parent_a: PromptGene, parent_b: PromptGene, gen: int
    ) -> PromptGene:
        """
        文レベル交叉: 親 A と親 B の文を交互に組み合わせる。
        """
        sents_a = [s.strip() for s in re.split(r"[。.!?！？\n]", parent_a.text) if s.strip()]
        sents_b = [s.strip() for s in re.split(r"[。.!?！？\n]", parent_b.text) if s.strip()]

        if not sents_a:
            return PromptGene(text=parent_b.text, generation=gen,
                              parents=[parent_a.gene_id, parent_b.gene_id])
        if not sents_b:
            return PromptGene(text=parent_a.text, generation=gen,
                              parents=[parent_a.gene_id, parent_b.gene_id])

        # 交叉点を選択
        cut_a = self._rng.randint(1, max(1, len(sents_a)))
        cut_b = self._rng.randint(0, max(1, len(sents_b) - 1))
        child_sents = sents_a[:cut_a] + sents_b[cut_b:]
        child_text = "。".join(child_sents).strip()
        if not child_text:
            child_text = parent_a.text

        return PromptGene(
            text=child_text,
            generation=gen,
            parents=[parent_a.gene_id, parent_b.gene_id],
        )

    def _mutate(
        self, gene: PromptGene, keywords: List[str], gen: int
    ) -> PromptGene:
        """
        突然変異: キーワード追加 / 修飾語挿入 / 文削除。
        """
        text = gene.text
        op = self._rng.choice(["insert_modifier", "add_keyword", "delete_sentence",
                                "rephrase_start"])
        if op == "insert_modifier":
            category = self._rng.choice(list(self._MUTATION_POOL.keys()))
            modifier = self._rng.choice(self._MUTATION_POOL[category])
            text = modifier + "、" + text
        elif op == "add_keyword" and keywords:
            kw = self._rng.choice(keywords)
            if kw not in text:
                text = text + f"（特に「{kw}」について）"
        elif op == "delete_sentence":
            sents = [s.strip() for s in re.split(r"[。.!?！？\n]", text) if s.strip()]
            if len(sents) > 1:
                del_idx = self._rng.randint(0, len(sents) - 1)
                sents.pop(del_idx)
                text = "。".join(sents)
        elif op == "rephrase_start":
            starters = ["まず", "次に", "重要なのは", "具体的には", "まとめると"]
            starter = self._rng.choice(starters)
            text = starter + "、" + text.lstrip("まず次に重要具体、")

        return PromptGene(
            text=text or gene.text,
            generation=gen,
            parents=[gene.gene_id],
        )

    def _diversity(self, population: List[PromptGene]) -> float:
        """
        集団内の平均ペアワイズ Jaccard 距離 (1-Jaccard = diversity)。
        """
        if len(population) < 2:
            return 1.0
        texts = [g.text for g in population]
        pairs = 0
        total_sim = 0.0
        limit = min(len(texts), 8)
        for i in range(limit):
            for j in range(i + 1, limit):
                sa = set(re.findall(r"[a-zA-Z0-9一-龯ぁ-ん]+", texts[i].lower()))
                sb = set(re.findall(r"[a-zA-Z0-9一-龯ぁ-ん]+", texts[j].lower()))
                if sa | sb:
                    sim = len(sa & sb) / len(sa | sb)
                else:
                    sim = 1.0
                total_sim += sim
                pairs += 1
        if pairs == 0:
            return 1.0
        avg_sim = total_sim / pairs
        return round(1.0 - avg_sim, 4)

    def _snapshot(self, population: List[PromptGene], gen: int) -> EvolutionRound:
        fitnesses = [g.fitness for g in population]
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        div = self._diversity(population)
        best_text = max(population, key=lambda g: g.fitness).text
        return EvolutionRound(
            generation=gen,
            best_fitness=round(best, 4),
            mean_fitness=round(mean, 4),
            diversity=div,
            best_text=best_text,
            population_size=len(population),
        )
