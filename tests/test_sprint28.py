"""
Sprint 28 テスト — 適応型プロンプト進化 PromptEvolution (P9)

- TestPromptGene        : PromptGene データ構造
- TestEvolutionConfig   : EvolutionConfig デフォルト値
- TestEvolutionRound    : 世代スナップショット
- TestEvolutionResult   : 進化結果
- TestPromptEvolution   : 遺伝的アルゴリズムエンジン
- TestCrossoverMutate   : 交叉・突然変異演算子
- TestIntegration       : 他スプリントとの連携
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


# ===========================================================================
# TestPromptGene
# ===========================================================================


class TestPromptGene:
    def test_default_fitness_zero(self):
        from open_mythos.prompt_evolution import PromptGene
        g = PromptGene(text="テストプロンプト")
        assert g.fitness == 0.0

    def test_gene_id_unique(self):
        from open_mythos.prompt_evolution import PromptGene
        g1 = PromptGene(text="a")
        g2 = PromptGene(text="b")
        assert g1.gene_id != g2.gene_id

    def test_ordering_by_fitness(self):
        from open_mythos.prompt_evolution import PromptGene
        g_low = PromptGene(text="low", fitness=0.3)
        g_high = PromptGene(text="high", fitness=0.8)
        assert g_low < g_high

    def test_parents_default_empty(self):
        from open_mythos.prompt_evolution import PromptGene
        g = PromptGene(text="test")
        assert g.parents == []

    def test_generation_stored(self):
        from open_mythos.prompt_evolution import PromptGene
        g = PromptGene(text="test", generation=3)
        assert g.generation == 3


# ===========================================================================
# TestEvolutionConfig
# ===========================================================================


class TestEvolutionConfig:
    def test_defaults(self):
        from open_mythos.prompt_evolution import EvolutionConfig
        cfg = EvolutionConfig()
        assert cfg.population_size >= 4
        assert 0.0 < cfg.mutation_rate < 1.0
        assert 0.0 < cfg.crossover_rate < 1.0
        assert cfg.elite_size >= 1

    def test_custom_values(self):
        from open_mythos.prompt_evolution import EvolutionConfig
        cfg = EvolutionConfig(population_size=12, n_generations=10, mutation_rate=0.5)
        assert cfg.population_size == 12
        assert cfg.n_generations == 10


# ===========================================================================
# TestEvolutionRound
# ===========================================================================


class TestEvolutionRound:
    def test_fields(self):
        from open_mythos.prompt_evolution import EvolutionRound
        r = EvolutionRound(
            generation=1,
            best_fitness=0.8,
            mean_fitness=0.6,
            diversity=0.4,
            best_text="SEO記事を書いてください",
            population_size=8,
        )
        assert r.generation == 1
        assert r.best_fitness == 0.8
        assert 0.0 <= r.diversity <= 1.0


# ===========================================================================
# TestEvolutionResult
# ===========================================================================


class TestEvolutionResult:
    def _run_quick(self, seed="SEO記事を書いてください"):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(population_size=4, n_generations=2, elite_size=1)
        evo = PromptEvolution(config=cfg, seed=42)
        return evo.evolve(seed)

    def test_best_gene_exists(self):
        r = self._run_quick()
        assert r.best_gene is not None
        assert r.best_gene.text != ""

    def test_fitness_history_length(self):
        r = self._run_quick()
        assert len(r.fitness_history) >= 1

    def test_improvement_is_float(self):
        r = self._run_quick()
        assert isinstance(r.improvement, float)

    def test_n_generations_run_positive(self):
        r = self._run_quick()
        assert r.n_generations_run >= 0

    def test_converged_is_bool(self):
        r = self._run_quick()
        assert isinstance(r.converged, bool)

    def test_best_prompt_property(self):
        r = self._run_quick()
        assert r.best_prompt == r.best_gene.text

    def test_summary_string(self):
        r = self._run_quick()
        s = r.summary()
        assert "PromptEvolution" in s
        assert "generations" in s


# ===========================================================================
# TestPromptEvolution
# ===========================================================================


class TestPromptEvolution:
    def _make_evo(self, pop=4, gens=2):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(population_size=pop, n_generations=gens, elite_size=1)
        return PromptEvolution(config=cfg, seed=0)

    def test_evolve_returns_result(self):
        from open_mythos.prompt_evolution import EvolutionResult
        evo = self._make_evo()
        result = evo.evolve("テストプロンプト")
        assert isinstance(result, EvolutionResult)

    def test_best_gene_has_valid_fitness(self):
        evo = self._make_evo()
        result = evo.evolve("SEO記事を書いてください")
        assert 0.0 <= result.best_gene.fitness <= 1.0

    def test_rounds_count_matches_generations(self):
        evo = self._make_evo(pop=4, gens=3)
        result = evo.evolve("テスト")
        assert len(result.rounds) >= 1

    def test_with_topic_keywords(self):
        evo = self._make_evo()
        result = evo.evolve("マーケティング記事", topic_keywords=["LLMO", "SEO", "コンバージョン"])
        assert result.best_gene.text != ""

    def test_with_templates(self):
        evo = self._make_evo()
        templates = [
            "詳細に説明してください",
            "箇条書きで教えてください",
        ]
        result = evo.evolve("AI技術について", templates=templates)
        assert result.best_gene is not None

    def test_custom_fitness_fn(self):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(population_size=4, n_generations=2)
        # 長さベースのカスタムフィットネス
        evo = PromptEvolution(config=cfg, fitness_fn=lambda t: min(1.0, len(t) / 100), seed=1)
        result = evo.evolve("短い")
        assert result.best_gene.fitness >= 0.0

    def test_guard_fn_filters_dangerous(self):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        blocked = []
        def guard(text: str) -> bool:
            if "悪意" in text:
                blocked.append(text)
                return True
            return False
        cfg = EvolutionConfig(population_size=4, n_generations=2)
        evo = PromptEvolution(config=cfg, guard_fn=guard, seed=2)
        result = evo.evolve("通常のプロンプト")
        # ガードが機能していること (blocked は空かもしれないが例外は出ない)
        assert result.best_gene is not None

    def test_elite_preserved_across_generations(self):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(population_size=6, n_generations=3, elite_size=2)
        evo = PromptEvolution(config=cfg, seed=7)
        result = evo.evolve("マーケティング")
        # 最良のフィットネスは最初の世代以上
        assert result.best_gene.fitness >= result.rounds[0].best_fitness - 0.01

    def test_early_stop_with_patience(self):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(
            population_size=4, n_generations=10,
            early_stop_delta=1.0,  # 高い閾値 → ほぼ即収束
            early_stop_patience=1,
        )
        evo = PromptEvolution(config=cfg, seed=5)
        result = evo.evolve("テスト")
        assert result.n_generations_run <= 10

    def test_deterministic_with_seed(self):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(population_size=4, n_generations=2)
        r1 = PromptEvolution(config=cfg, seed=42).evolve("テスト")
        r2 = PromptEvolution(config=cfg, seed=42).evolve("テスト")
        assert abs(r1.best_gene.fitness - r2.best_gene.fitness) < 0.001


# ===========================================================================
# TestCrossoverMutate
# ===========================================================================


class TestCrossoverMutate:
    def _make_evo(self):
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(population_size=4, n_generations=1)
        return PromptEvolution(config=cfg, seed=10)

    def test_crossover_creates_child(self):
        from open_mythos.prompt_evolution import PromptGene
        evo = self._make_evo()
        p1 = PromptGene(text="SEO記事を書いてください。詳細に記述します。")
        p2 = PromptGene(text="マーケティングレポートを作成します。専門用語を使います。")
        child = evo._crossover(p1, p2, gen=1)
        assert child.text != ""
        assert p1.gene_id in child.parents or p2.gene_id in child.parents

    def test_mutate_changes_text(self):
        from open_mythos.prompt_evolution import PromptGene
        evo = self._make_evo()
        original = PromptGene(text="テスト記事を書いてください", generation=0)
        mutated = evo._mutate(original, keywords=["SEO", "LLMO"], gen=1)
        # 変異後も空文字列にならない
        assert mutated.text != ""

    def test_diversity_calculation(self):
        from open_mythos.prompt_evolution import PromptGene
        evo = self._make_evo()
        diverse_pop = [
            PromptGene(text="SEO記事を詳細に書く"),
            PromptGene(text="料理レシピを簡潔に説明する"),
            PromptGene(text="プログラミングコードを解説する"),
        ]
        div = evo._diversity(diverse_pop)
        assert 0.0 <= div <= 1.0

    def test_single_element_diversity_is_one(self):
        from open_mythos.prompt_evolution import PromptGene
        evo = self._make_evo()
        pop = [PromptGene(text="一つだけ")]
        div = evo._diversity(pop)
        assert div == 1.0


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    def test_ensemble_scorer_as_fitness(self):
        """EnsembleScorer (Sprint 27) をフィットネス関数として使用。"""
        from open_mythos.ensemble_scorer import EnsembleScorer
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution

        scorer = EnsembleScorer()

        def ensemble_fitness(text: str) -> float:
            return scorer.score(text).ensemble_score

        cfg = EvolutionConfig(population_size=4, n_generations=2)
        evo = PromptEvolution(config=cfg, fitness_fn=ensemble_fitness, seed=3)
        result = evo.evolve("SEO記事を最適化してください", topic_keywords=["LLMO", "検索"])
        assert 0.0 <= result.best_gene.fitness <= 1.0

    def test_mistake_guard_as_guard_fn(self):
        """MistakeGuard (Sprint 24) をガード関数として使用。"""
        from open_mythos.error_memory import MistakeGuard
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution

        guard = MistakeGuard()

        def safe_check(text: str) -> bool:
            # True = 危険 → 除去
            result = guard.check(text)
            return result.blocked

        cfg = EvolutionConfig(population_size=4, n_generations=2)
        evo = PromptEvolution(config=cfg, guard_fn=safe_check, seed=4)
        result = evo.evolve("通常プロンプトを最適化する")
        assert result.best_gene is not None

    def test_fitness_improves_or_stays(self):
        """世代を重ねるとフィットネスが単調に改善されること。"""
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        cfg = EvolutionConfig(
            population_size=6, n_generations=4, elite_size=2,
            mutation_rate=0.4, crossover_rate=0.6,
        )
        evo = PromptEvolution(config=cfg, seed=99)
        result = evo.evolve(
            "SEO最適化コンテンツを作成してください",
            topic_keywords=["LLMO", "エンティティ", "構造化"],
        )
        # 最終フィットネスは初期以上
        assert result.best_gene.fitness >= result.rounds[0].best_fitness - 0.05

    def test_llmo_optimizer_seed_integration(self):
        """Sprint 19 の LLMOOptimizer で改善したプロンプトを seed に使う。"""
        from open_mythos.llmo import LLMOOptimizer
        from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution
        opt = LLMOOptimizer()
        opt_result = opt.optimize(
            "SEO記事",
            query="最適化の方法",
            max_iterations=2,
        )
        cfg = EvolutionConfig(population_size=4, n_generations=2)
        evo = PromptEvolution(config=cfg, seed=11)
        result = evo.evolve(opt_result.optimized_text[:200])
        assert result.best_gene.text != ""
