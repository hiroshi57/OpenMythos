import pytest

# ---------------------------------------------------------------------------
# serve.ab_router
# ---------------------------------------------------------------------------


class TestABRoute:
    def test_deterministic_same_user(self):
        from serve.ab_router import _route

        r1 = _route("user-abc-123")
        r2 = _route("user-abc-123")
        assert r1 == r2, "同じ user_id は同じグループになること"

    def test_returns_valid_group(self):
        from serve.ab_router import _route

        for uid in ["alice", "bob", "charlie", "dave", "eve"]:
            assert _route(uid) in {"openmythos", "existing_ml"}

    def test_traffic_split_approx(self):
        """100ユーザーでおおよそ OPENMYTHOS_TRAFFIC_PCT% が openmythos グループ。"""
        from serve.ab_router import _route, OPENMYTHOS_TRAFFIC_PCT

        groups = [_route(f"user-{i:04d}") for i in range(200)]
        openmythos_pct = groups.count("openmythos") / len(groups) * 100
        # 余裕を持たせて ±15%
        assert abs(openmythos_pct - OPENMYTHOS_TRAFFIC_PCT) < 15

    def test_different_users_may_differ(self):
        from serve.ab_router import _route

        results = {_route(f"uid-{i}") for i in range(50)}
        assert len(results) == 2, "50ユーザーでは両グループに割り振られること"

    def test_stats_init(self):
        from serve.ab_router import _Stats

        s = _Stats()
        assert s.counts["openmythos"] == 0
        assert s.latencies["existing_ml"] == []

    def test_stats_accumulate(self):
        from serve.ab_router import _Stats

        s = _Stats()
        s.counts["openmythos"] += 1
        s.latencies["openmythos"].append(42.0)
        s.scores["openmythos"].append(0.8)
        assert s.counts["openmythos"] == 1
        assert s.latencies["openmythos"] == [42.0]


# ---------------------------------------------------------------------------
# serve.sla_router
# ---------------------------------------------------------------------------


class TestSLALoopResolution:
    def test_fraud_detect_accurate_has_most_loops(self):
        from serve.sla_router import _resolve_loops_and_budget

        loops, budget = _resolve_loops_and_budget("fraud_detect", "accurate")
        assert loops >= 12, "詐欺検知・accurate は最大ループ数であること"
        assert budget >= 1000

    def test_ad_performance_fast_has_low_loops(self):
        from serve.sla_router import _resolve_loops_and_budget

        loops, budget = _resolve_loops_and_budget("ad_performance", "fast")
        assert loops <= 4, "広告・fast は低ループ数であること"
        assert budget <= 500

    def test_general_task_fallback(self):
        from serve.sla_router import _resolve_loops_and_budget

        loops, budget = _resolve_loops_and_budget("unknown_task", "balanced")
        assert loops > 0
        assert budget > 0

    def test_all_tasks_all_modes(self):
        from serve.sla_router import _resolve_loops_and_budget, DEFAULT_SLA

        for task in DEFAULT_SLA:
            for mode in ("fast", "balanced", "accurate"):
                loops, budget = _resolve_loops_and_budget(task, mode)
                assert 1 <= loops <= 16
                assert budget > 0

    def test_accurate_loops_gte_balanced(self):
        from serve.sla_router import _resolve_loops_and_budget, DEFAULT_SLA

        for task in DEFAULT_SLA:
            loops_bal, _ = _resolve_loops_and_budget(task, "balanced")
            loops_acc, _ = _resolve_loops_and_budget(task, "accurate")
            assert loops_acc >= loops_bal, f"{task}: accurate >= balanced"

    def test_config_update(self):
        from serve.sla_router import _sla_config

        original = _sla_config.get("general", {}).get("balanced", (4, 800))
        _sla_config.setdefault("general", {})["balanced"] = (99, 9999)
        loops, budget = _sla_config["general"]["balanced"]
        assert loops == 99
        assert budget == 9999
        # 元に戻す
        _sla_config["general"]["balanced"] = original

    def test_health_response_has_tasks(self):
        from serve.sla_router import _sla_config

        assert "general" in _sla_config
        assert "fraud_detect" in _sla_config


# ---------------------------------------------------------------------------
# serve.monitor
# ---------------------------------------------------------------------------


class TestPSI:
    def test_identical_distributions_psi_near_zero(self):
        from serve.monitor import _psi

        data = [i / 10 for i in range(10)]
        result = _psi(data, data)
        assert result < 0.05, "同一分布の PSI ≈ 0"

    def test_opposite_distributions_psi_high(self):
        from serve.monitor import _psi

        low = [0.1] * 20
        high = [0.9] * 20
        result = _psi(low, high)
        assert result > 0.1, "逆分布は PSI が大きいこと"

    def test_empty_inputs_return_zero(self):
        from serve.monitor import _psi

        assert _psi([], []) == 0.0
        assert _psi([0.5], []) == 0.0

    def test_psi_non_negative(self):
        from serve.monitor import _psi

        import random

        rng = random.Random(0)
        a = [rng.random() for _ in range(50)]
        b = [rng.random() for _ in range(50)]
        assert _psi(a, b) >= 0.0


class TestMonitorDB:
    @pytest.fixture
    def tmp_db(self, tmp_path, monkeypatch):
        """一時ディレクトリに monitor DB を設置してテスト分離する。"""
        import serve.monitor as m

        db = tmp_path / "monitor.db"
        monkeypatch.setattr(m, "DB_PATH", db)
        m._init_db()
        return db

    def test_log_inference_creates_row(self, tmp_db):
        from serve.monitor import log_inference
        import sqlite3

        log_inference(
            model_id="test",
            task="general",
            score=0.75,
            label=1,
            latency_ms=12.3,
            n_loops=4,
            ground_truth=1,
        )
        con = sqlite3.connect(tmp_db)
        rows = con.execute("SELECT * FROM inference_log").fetchall()
        con.close()
        assert len(rows) == 1

    def test_log_inference_multiple(self, tmp_db):
        from serve.monitor import log_inference
        import sqlite3

        for i in range(5):
            log_inference("m", "ad_performance", 0.5 + i * 0.05, 1, 10.0)

        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM inference_log").fetchone()[0]
        con.close()
        assert count == 5

    def test_check_drift_no_data(self, tmp_db):
        from serve.monitor import check_drift

        result = check_drift("nonexistent_task")
        assert result["status"] == "no_data"

    def test_set_baseline_and_check_drift(self, tmp_db):
        from serve.monitor import log_inference, set_baseline, check_drift

        set_baseline(
            "general", accuracy=0.9, avg_score=0.8, score_p25=0.6, score_p75=0.95
        )
        for i in range(10):
            log_inference("m", "general", 0.7 + i * 0.01, 1, 5.0)

        result = check_drift("general")
        assert result["status"] in {"ok", "drift_detected", "no_data", "alert"}
        assert "task" in result

    def test_is_correct_set_when_ground_truth_matches(self, tmp_db):
        from serve.monitor import log_inference
        import sqlite3

        log_inference("m", "general", 0.8, 1, 5.0, ground_truth=1)
        log_inference("m", "general", 0.2, 0, 5.0, ground_truth=1)

        con = sqlite3.connect(tmp_db)
        rows = con.execute(
            "SELECT is_correct FROM inference_log ORDER BY id"
        ).fetchall()
        con.close()
        assert rows[0][0] == 1  # 正解
        assert rows[1][0] == 0  # 不正解


# ---------------------------------------------------------------------------
# serve.api — config / schema
# ---------------------------------------------------------------------------


class TestAPIConfig:
    def test_build_config_returns_mythos_config(self):
        from serve.api import _build_config
        from open_mythos.main import MythosConfig

        cfg = _build_config(128, "gqa")
        assert isinstance(cfg, MythosConfig)
        assert cfg.dim == 128

    def test_build_config_mla(self):
        from serve.api import _build_config

        cfg = _build_config(256, "mla")
        assert cfg.attn_type == "mla"

    def test_task_loops_covers_all_task_types(self):
        from serve.api import TASK_LOOPS

        required = {
            "ad_performance",
            "content_quality",
            "persona_segment",
            "market_research",
            "identity_verify",
            "fraud_detect",
            "general",
        }
        assert required.issubset(set(TASK_LOOPS.keys()))

    def test_task_loops_values_within_max(self):
        from serve.api import TASK_LOOPS, MAX_LOOPS

        for task, loops in TASK_LOOPS.items():
            assert 1 <= loops <= MAX_LOOPS, f"{task}: loops={loops}"

    def test_fraud_detect_has_highest_loops(self):
        from serve.api import TASK_LOOPS

        assert TASK_LOOPS["fraud_detect"] == max(TASK_LOOPS.values())


class TestAPISchemas:
    def test_infer_request_default(self):
        from serve.api import InferRequest

        req = InferRequest(text="hello")
        assert req.task == "general"
        assert req.max_new_tokens is None

    def test_infer_request_validation(self):
        from serve.api import InferRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InferRequest(text="x", loops=0)  # loops < 1

    def test_ab_request_default(self):
        from serve.ab_router import ABRequest

        req = ABRequest(user_id="u1", text="hello")
        assert req.task == "general"
        assert req.ground_truth is None

    def test_sla_request_default(self):
        from serve.sla_router import SLARequest

        req = SLARequest(text="hello")
        assert req.sla_mode == "balanced"
        assert req.loops_override is None
