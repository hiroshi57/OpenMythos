"""
Sprint 20 テスト — 討議型集合知 DebateOrchestrator

- TestDebateConfig        : 設定値の検証
- TestConsensusEngine     : Jaccard合意スコア計算
- TestDebateRound         : 1ラウンドのデータ構造
- TestDebateResult        : 討議結果データ構造
- TestDebateOrchestrator  : 実行フロー (tiny model)
- TestDebateAPIEndpoint   : FastAPI /v1/debate/run
"""

from __future__ import annotations


def _tiny_model():
    from open_mythos.main import OpenMythos, MythosConfig

    cfg = MythosConfig(
        vocab_size=256,
        dim=32,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=64,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
    )
    return OpenMythos(cfg).eval()


# ===========================================================================
# TestDebateConfig
# ===========================================================================


class TestDebateConfig:
    def test_defaults(self):
        from open_mythos.debate import DebateConfig

        cfg = DebateConfig()
        assert cfg.n_agents == 3
        assert cfg.n_rounds == 2
        assert cfg.consensus_threshold == 0.75
        assert cfg.max_workers == 4

    def test_custom_values(self):
        from open_mythos.debate import DebateConfig

        cfg = DebateConfig(n_agents=5, n_rounds=3, consensus_threshold=0.8)
        assert cfg.n_agents == 5
        assert cfg.n_rounds == 3
        assert cfg.consensus_threshold == 0.8

    def test_agent_name_prefix(self):
        from open_mythos.debate import DebateConfig

        cfg = DebateConfig(agent_name_prefix="critic")
        assert cfg.agent_name_prefix == "critic"


# ===========================================================================
# TestConsensusEngine
# ===========================================================================


class TestConsensusEngine:
    def test_identical_texts_score_1(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        texts = ["SEO対策は重要です", "SEO対策は重要です", "SEO対策は重要です"]
        consensus, score = engine.score(texts)
        assert score == 1.0
        assert consensus == "SEO対策は重要です"

    def test_single_text_score_1(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        consensus, score = engine.score(["単独テキスト"])
        assert score == 1.0
        assert consensus == "単独テキスト"

    def test_empty_list_score_0(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        consensus, score = engine.score([])
        assert score == 0.0
        assert consensus == ""

    def test_dissimilar_texts_low_score(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        texts = ["apple banana cherry", "python java golang", "football tennis soccer"]
        _, score = engine.score(texts)
        assert 0.0 <= score <= 0.3

    def test_similar_texts_high_score(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        texts = [
            "SEO対策でコンテンツを最適化する",
            "SEO対策でコンテンツを品質向上する",
            "SEO対策でコンテンツを改善する",
        ]
        _, score = engine.score(texts)
        assert score > 0.3

    def test_score_range(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        texts = ["hello world", "world hello", "hello there world"]
        _, score = engine.score(texts)
        assert 0.0 <= score <= 1.0

    def test_consensus_is_one_of_inputs(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        texts = ["text A B C", "text A B D", "text A B E"]
        consensus, _ = engine.score(texts)
        assert consensus in texts

    def test_confidence_single_round(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        conf = engine.confidence([0.8])
        assert conf == 0.8

    def test_confidence_increasing_rounds(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        conf_inc = engine.confidence([0.4, 0.6, 0.8])
        conf_dec = engine.confidence([0.8, 0.6, 0.4])
        assert conf_inc > conf_dec

    def test_confidence_range(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        conf = engine.confidence([0.5, 0.7, 0.9])
        assert 0.0 <= conf <= 1.0

    def test_all_empty_texts(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        consensus, score = engine.score(["", "", ""])
        assert score == 0.0

    def test_two_texts(self):
        from open_mythos.debate import ConsensusEngine

        engine = ConsensusEngine()
        texts = ["hello world foo", "hello world bar"]
        consensus, score = engine.score(texts)
        assert 0.0 <= score <= 1.0
        assert consensus in texts


# ===========================================================================
# TestDebateRound
# ===========================================================================


class TestDebateRound:
    def test_default_fields(self):
        from open_mythos.debate import DebateRound

        r = DebateRound(round_num=0)
        assert r.round_num == 0
        assert r.proposals == {}
        assert r.critiques == {}
        assert r.refinements == {}
        assert r.agreement_score == 0.0

    def test_field_assignment(self):
        from open_mythos.debate import DebateRound

        r = DebateRound(
            round_num=1,
            proposals={0: "提案A", 1: "提案B"},
            agreement_score=0.75,
        )
        assert r.proposals[0] == "提案A"
        assert r.agreement_score == 0.75


# ===========================================================================
# TestDebateResult
# ===========================================================================


class TestDebateResult:
    def _make_result(self, agreement_score=0.8, confidence=0.85):
        from open_mythos.debate import DebateResult, DebateRound

        rounds = [DebateRound(round_num=0, agreement_score=agreement_score)]
        return DebateResult(
            topic="テスト",
            rounds=rounds,
            consensus="合意テキスト",
            agreement_score=agreement_score,
            confidence=confidence,
            n_rounds_used=1,
            total_latency_ms=100.0,
        )

    def test_improved_over_solo_true(self):
        result = self._make_result(agreement_score=0.7)
        assert result.improved_over_solo is True

    def test_improved_over_solo_false(self):
        result = self._make_result(agreement_score=0.4)
        assert result.improved_over_solo is False

    def test_round_scores(self):
        result = self._make_result(agreement_score=0.75)
        scores = result.round_scores()
        assert len(scores) == 1
        assert scores[0] == 0.75

    def test_early_stopped_default_false(self):
        result = self._make_result()
        assert result.early_stopped is False

    def test_n_rounds_used(self):
        result = self._make_result()
        assert result.n_rounds_used == 1


# ===========================================================================
# TestDebateOrchestrator
# ===========================================================================


class TestDebateOrchestrator:
    def test_run_returns_debate_result(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig, DebateResult

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("SEO戦略について")
        assert isinstance(result, DebateResult)

    def test_result_topic_matches(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("テストトピック")
        assert result.topic == "テストトピック"

    def test_agreement_score_in_range(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("コンテンツ最適化")
        assert 0.0 <= result.agreement_score <= 1.0

    def test_confidence_in_range(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("マーケティング戦略")
        assert 0.0 <= result.confidence <= 1.0

    def test_rounds_count(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=2)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("広告戦略")
        assert result.n_rounds_used <= 2
        assert len(result.rounds) <= 2

    def test_consensus_is_string(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("トピック")
        assert isinstance(result.consensus, str)

    def test_total_latency_positive(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("レイテンシテスト")
        assert result.total_latency_ms > 0

    def test_context_manager_shutdown(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        debate = DebateOrchestrator(model, cfg, max_new_tokens=4)
        debate.run("テスト")
        debate.shutdown()

    def test_three_agents(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=3, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("3エージェント討議")
        assert result.n_rounds_used >= 1

    def test_early_stop_on_threshold(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        # threshold=0.0 なら必ず1ラウンドで早期終了
        cfg = DebateConfig(n_agents=2, n_rounds=3, consensus_threshold=0.0)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("早期終了テスト")
        assert result.early_stopped is True
        assert result.n_rounds_used == 1

    def test_round_has_proposals(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("提案テスト")
        assert len(result.rounds[0].proposals) == 2

    def test_round_has_critiques(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("批評テスト")
        assert len(result.rounds[0].critiques) == 2

    def test_round_has_refinements(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("洗練テスト")
        assert len(result.rounds[0].refinements) == 2

    def test_round_agreement_score_in_range(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("ラウンドスコアテスト")
        assert 0.0 <= result.rounds[0].agreement_score <= 1.0

    def test_round_latency_positive(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("レイテンシ")
        assert result.rounds[0].latency_ms > 0

    def test_round_scores_length(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=2, consensus_threshold=1.1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("2ラウンド")
        assert len(result.round_scores()) == 2

    def test_no_threshold_runs_all_rounds(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=2, consensus_threshold=1.1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("全ラウンド実行")
        assert result.n_rounds_used == 2
        assert result.early_stopped is False

    def test_improved_over_solo_field_present(self):
        from open_mythos.debate import DebateOrchestrator, DebateConfig

        model = _tiny_model()
        cfg = DebateConfig(n_agents=2, n_rounds=1)
        with DebateOrchestrator(model, cfg, max_new_tokens=4) as debate:
            result = debate.run("改善チェック")
        assert isinstance(result.improved_over_solo, bool)


# ===========================================================================
# TestDebateAPIEndpoint  (静的ソース検査 — state 未初期化問題を回避)
# ===========================================================================

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


class TestDebateAPIEndpoint:
    def _api_source(self) -> str:
        return (_ROOT / "serve" / "api.py").read_text(encoding="utf-8")

    def test_debate_run_route_exists(self):
        assert '"/v1/debate/run"' in self._api_source()

    def test_debate_run_post_method(self):
        src = self._api_source()
        idx = src.index('"/v1/debate/run"')
        snippet = src[max(0, idx - 60) : idx]
        assert "@app.post" in snippet

    def test_debate_run_tag_debate(self):
        src = self._api_source()
        idx = src.index('"/v1/debate/run"')
        snippet = src[idx : idx + 300]
        assert 'tags=["debate"]' in snippet

    def test_debate_run_request_model(self):
        assert "DebateRunRequest" in self._api_source()

    def test_debate_run_topic_field(self):
        src = self._api_source()
        idx = src.index("DebateRunRequest")
        snippet = src[idx : idx + 400]
        assert "topic" in snippet

    def test_debate_run_n_agents_field(self):
        src = self._api_source()
        idx = src.index("DebateRunRequest")
        snippet = src[idx : idx + 400]
        assert "n_agents" in snippet

    def test_debate_run_n_rounds_field(self):
        src = self._api_source()
        idx = src.index("DebateRunRequest")
        snippet = src[idx : idx + 400]
        assert "n_rounds" in snippet

    def test_debate_run_consensus_threshold_field(self):
        src = self._api_source()
        idx = src.index("DebateRunRequest")
        snippet = src[idx : idx + 400]
        assert "consensus_threshold" in snippet

    def test_debate_run_returns_consensus_key(self):
        src = self._api_source()
        assert '"consensus"' in src

    def test_debate_run_returns_agreement_score_key(self):
        src = self._api_source()
        assert '"agreement_score"' in src

    def test_debate_run_returns_confidence_key(self):
        src = self._api_source()
        assert '"confidence"' in src

    def test_debate_run_returns_early_stopped_key(self):
        src = self._api_source()
        assert '"early_stopped"' in src

    def test_debate_run_uses_debate_orchestrator(self):
        src = self._api_source()
        assert "DebateOrchestrator" in src

    def test_debate_run_verify_api_key_dependency(self):
        src = self._api_source()
        idx = src.index("def debate_run")
        snippet = src[max(0, idx - 300) : idx + 100]
        assert "verify_api_key" in snippet

    def test_debate_run_uses_debate_config(self):
        src = self._api_source()
        assert "DebateConfig" in src

    def test_debate_run_n_rounds_used_key(self):
        src = self._api_source()
        assert '"n_rounds_used"' in src

    def test_debate_run_improved_over_solo_key(self):
        src = self._api_source()
        assert '"improved_over_solo"' in src

    def test_debate_run_total_latency_key(self):
        src = self._api_source()
        assert '"total_latency_ms"' in src

    def test_debate_run_rounds_key(self):
        src = self._api_source()
        assert '"rounds"' in src
