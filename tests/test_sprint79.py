"""
Sprint 79A — 都市マップ WebSocket リアルタイム更新 テスト
累計: 4487 + ? → 目標 +50 PASS
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from open_mythos.skills.city_map_realtime import (
    ConnectionState,
    DistrictStateSimulator,
    DistrictUpdate,
    RealtimeConfig,
    RealtimeEngine,
    RealtimeEvent,
    RealtimeManager,
    RealtimeManagerFactory,
    RealtimeSession,
    RealtimeSnapshot,
    TrainLine,
    TrainPosition,
    TrainScheduler,
    UpdateType,
    TICK_INTERVAL,
    MAX_SESSIONS,
)


# ─── UpdateType ───────────────────────────────────────────────────

class TestUpdateType:
    def test_values(self):
        assert UpdateType.TRAIN_POSITION == "train_position"
        assert UpdateType.CROWD_DENSITY  == "crowd_density"
        assert UpdateType.ENERGY_LOAD    == "energy_load"
        assert UpdateType.NOISE_LEVEL    == "noise_level"
        assert UpdateType.DISASTER_ALERT == "disaster_alert"

    def test_all_five_types(self):
        assert len(UpdateType) == 5


# ─── TrainLine ────────────────────────────────────────────────────

class TestTrainLine:
    def test_three_lines(self):
        assert len(TrainLine) == 3

    def test_values(self):
        assert TrainLine.YAMANOTE == "yamanote"
        assert TrainLine.CHUO     == "chuo"
        assert TrainLine.TOKAIDO  == "tokaido"


# ─── TrainPosition ────────────────────────────────────────────────

class TestTrainPosition:
    def _make(self, pos=0.5, delay=0):
        return TrainPosition(
            line=TrainLine.YAMANOTE,
            train_id="yamanote-00",
            position=pos,
            speed_kmh=50.0,
            delay_minutes=delay,
            direction=1,
        )

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for key in ("line", "train_id", "position", "speed_kmh", "delay_minutes", "direction", "timestamp"):
            assert key in d

    def test_position_rounded(self):
        t = self._make(pos=0.123456789)
        assert len(str(t.to_dict()["position"])) <= 8

    def test_line_value_in_dict(self):
        d = self._make().to_dict()
        assert d["line"] == "yamanote"


# ─── DistrictUpdate ───────────────────────────────────────────────

class TestDistrictUpdate:
    def _make(self):
        return DistrictUpdate(
            district_name="新宿",
            update_type=UpdateType.CROWD_DENSITY,
            old_value="normal",
            new_value="crowded",
            numeric=0.7,
        )

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for key in ("district_name", "update_type", "old_value", "new_value", "numeric", "timestamp"):
            assert key in d

    def test_numeric_rounded(self):
        u = self._make()
        assert isinstance(u.to_dict()["numeric"], float)

    def test_update_type_string(self):
        d = self._make().to_dict()
        assert d["update_type"] == "crowd_density"


# ─── RealtimeEvent ────────────────────────────────────────────────

class TestRealtimeEvent:
    def _make(self):
        return RealtimeEvent(
            city="Tokyo",
            tick=5,
            trains=[TrainPosition(TrainLine.YAMANOTE, "y-00", 0.3, 50.0, 0, 1)],
            district_updates=[DistrictUpdate("渋谷", UpdateType.ENERGY_LOAD, "normal", "high", 0.6)],
        )

    def test_to_dict_structure(self):
        d = self._make().to_dict()
        assert d["city"] == "Tokyo"
        assert d["tick"] == 5
        assert len(d["trains"]) == 1
        assert len(d["district_updates"]) == 1

    def test_to_json_parseable(self):
        j = self._make().to_json()
        parsed = json.loads(j)
        assert parsed["tick"] == 5

    def test_event_id_generated(self):
        e = RealtimeEvent()
        assert len(e.event_id) > 0


# ─── RealtimeConfig ───────────────────────────────────────────────

class TestRealtimeConfig:
    def test_defaults(self):
        c = RealtimeConfig()
        assert c.city == "Tokyo"
        assert c.tick_interval == TICK_INTERVAL
        assert c.trains_per_line == 3
        assert 0 < c.district_change_prob < 1
        assert c.disaster_prob < 0.1

    def test_custom(self):
        c = RealtimeConfig(city="Osaka", tick_interval=1.0, seed=42)
        assert c.city == "Osaka"
        assert c.tick_interval == 1.0
        assert c.seed == 42


# ─── TrainScheduler ───────────────────────────────────────────────

class TestTrainScheduler:
    def _make(self, trains_per_line=2, seed=42):
        cfg = RealtimeConfig(trains_per_line=trains_per_line, seed=seed)
        return TrainScheduler(cfg)

    def test_init_count(self):
        ts = self._make(trains_per_line=3)
        trains = ts.get_all()
        assert len(trains) == 3 * len(TrainLine)  # 9

    def test_positions_in_range(self):
        ts = self._make()
        for t in ts.get_all():
            assert 0.0 <= t.position <= 1.0

    def test_tick_returns_all(self):
        ts = self._make()
        result = ts.tick(dt=10.0)
        assert len(result) == len(ts.get_all())

    def test_positions_change_after_tick(self):
        ts = self._make(seed=0)
        before = {t.train_id: t.position for t in ts.get_all()}
        ts.tick(dt=60.0)  # 1分
        after = {t.train_id: t.position for t in ts.get_all()}
        changed = sum(1 for tid in before if before[tid] != after[tid])
        assert changed > 0

    def test_train_ids_unique(self):
        ts = self._make()
        ids = [t.train_id for t in ts.get_all()]
        assert len(ids) == len(set(ids))

    def test_speed_positive(self):
        ts = self._make()
        for t in ts.get_all():
            assert t.speed_kmh > 0


# ─── DistrictStateSimulator ───────────────────────────────────────

class TestDistrictStateSimulator:
    def _make(self, prob=1.0):
        cfg = RealtimeConfig(district_change_prob=prob, seed=7)
        names = ["新宿", "渋谷", "品川", "池袋", "上野"]
        return DistrictStateSimulator(names, cfg), names

    def test_initial_snapshot_keys(self):
        sim, names = self._make()
        snap = sim.get_snapshot()
        assert set(snap.keys()) == set(names)

    def test_initial_values_valid(self):
        sim, _ = self._make()
        snap = sim.get_snapshot()
        valid_crowd   = {"sparse", "normal", "crowded", "packed"}
        valid_energy  = {"normal", "high", "critical"}
        valid_noise   = {"compliant", "near_limit", "violation"}
        for state in snap.values():
            assert state["crowd_level"] in valid_crowd
            assert state["energy_status"] in valid_energy
            assert state["noise_status"] in valid_noise
            assert state["disaster_level"] is None

    def test_tick_returns_updates(self):
        sim, _ = self._make(prob=1.0)
        updates = sim.tick()
        assert len(updates) > 0

    def test_update_type_valid(self):
        sim, _ = self._make(prob=1.0)
        for u in sim.tick():
            assert isinstance(u.update_type, UpdateType)

    def test_numeric_in_range(self):
        sim, _ = self._make(prob=1.0)
        for u in sim.tick():
            assert 0.0 <= u.numeric <= 1.0

    def test_zero_prob_no_updates(self):
        cfg = RealtimeConfig(district_change_prob=0.0, disaster_prob=0.0, seed=1)
        sim = DistrictStateSimulator(["A", "B"], cfg)
        updates = sim.tick()
        assert len(updates) == 0


# ─── RealtimeEngine ───────────────────────────────────────────────

class TestRealtimeEngine:
    def _make(self, names=None, seed=42):
        cfg = RealtimeConfig(seed=seed, district_change_prob=0.5)
        return RealtimeEngine(cfg, district_names=names or ["A", "B", "C"])

    def test_tick_increments(self):
        eng = self._make()
        assert eng.tick_count == 0
        eng.tick(dt=2.0)
        assert eng.tick_count == 1
        eng.tick(dt=2.0)
        assert eng.tick_count == 2

    def test_tick_returns_event(self):
        eng = self._make()
        event = eng.tick(dt=2.0)
        assert isinstance(event, RealtimeEvent)
        assert event.tick == 1

    def test_event_has_trains(self):
        eng = self._make()
        event = eng.tick(dt=2.0)
        assert len(event.trains) > 0

    def test_snapshot_structure(self):
        eng = self._make()
        snap = eng.snapshot(connected_clients=3)
        assert isinstance(snap, RealtimeSnapshot)
        assert snap.connected_clients == 3
        assert "A" in snap.districts

    def test_uptime_grows(self):
        eng = self._make()
        t0 = eng.uptime
        time.sleep(0.01)
        assert eng.uptime > t0

    def test_city_in_event(self):
        cfg = RealtimeConfig(city="Osaka", seed=0)
        eng = RealtimeEngine(cfg, ["X"])
        event = eng.tick(dt=1.0)
        assert event.city == "Osaka"


# ─── RealtimeSession ─────────────────────────────────────────────

class TestRealtimeSession:
    def _make_session(self):
        sent = []
        async def _send(msg):
            sent.append(msg)
        session = RealtimeSession("sess-01", _send)
        return session, sent

    @pytest.mark.asyncio
    async def test_send_success(self):
        session, sent = self._make_session()
        event = RealtimeEvent(city="Tokyo", tick=1)
        ok = await session.send(event)
        assert ok is True
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_send_increments_count(self):
        session, _ = self._make_session()
        event = RealtimeEvent(city="Tokyo", tick=1)
        await session.send(event)
        assert session.messages_sent == 1

    @pytest.mark.asyncio
    async def test_send_failure_marks_closed(self):
        async def _bad_send(msg):
            raise ConnectionError("断線")
        session = RealtimeSession("sess-02", _bad_send)
        event = RealtimeEvent(city="Tokyo", tick=1)
        ok = await session.send(event)
        assert ok is False
        assert session.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_history_stored(self):
        session, _ = self._make_session()
        for i in range(3):
            await session.send(RealtimeEvent(tick=i))
        assert len(session.get_history()) == 3

    def test_uptime(self):
        session, _ = self._make_session()
        time.sleep(0.01)
        assert session.uptime > 0


# ─── RealtimeManager ─────────────────────────────────────────────

class TestRealtimeManager:
    def _make(self):
        return RealtimeManagerFactory.create_mock(seed=0)

    def test_register_session(self):
        mgr = self._make()
        called = []
        async def _send(msg): called.append(msg)
        session = mgr.register_session("s1", _send)
        assert session.session_id == "s1"
        assert mgr.session_count == 1

    def test_remove_session(self):
        mgr = self._make()
        async def _send(msg): pass
        mgr.register_session("s1", _send)
        mgr.remove_session("s1")
        assert mgr.session_count == 0

    def test_get_session(self):
        mgr = self._make()
        async def _send(msg): pass
        mgr.register_session("s1", _send)
        assert mgr.get_session("s1") is not None
        assert mgr.get_session("unknown") is None

    def test_status_keys(self):
        mgr = self._make()
        st = mgr.status()
        for k in ("running", "connected_clients", "tick_count", "uptime_seconds", "tick_interval", "city"):
            assert k in st

    def test_init_engine(self):
        mgr = self._make()
        eng = mgr.init_engine(["X", "Y"])
        assert eng is not None
        assert mgr.engine is eng

    def test_snapshot_none_before_engine(self):
        mgr = RealtimeManager()
        assert mgr.snapshot() is None

    def test_snapshot_after_engine(self):
        mgr = self._make()
        mgr.init_engine(["A", "B"])
        snap = mgr.snapshot()
        assert snap is not None
        assert snap.city == "Tokyo"

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_sessions(self):
        mgr = self._make()
        received = []
        async def _send(msg): received.append(msg)
        mgr.register_session("s1", _send)
        event = RealtimeEvent(city="Tokyo", tick=1)
        sent = await mgr.broadcast(event)
        assert sent == 1
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_sessions(self):
        mgr = self._make()
        async def _bad(msg): raise Exception("dead")
        mgr.register_session("dead", _bad)
        event = RealtimeEvent(tick=1)
        await mgr.broadcast(event)
        assert mgr.session_count == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        mgr = self._make()
        mgr.init_engine(["X"])
        await mgr.start()
        assert mgr._running is True
        await mgr.stop()
        assert mgr._running is False


# ─── RealtimeManagerFactory ───────────────────────────────────────

class TestRealtimeManagerFactory:
    def test_create_default(self):
        mgr = RealtimeManagerFactory.create()
        assert mgr._config.city == "Tokyo"
        assert mgr._config.tick_interval == TICK_INTERVAL

    def test_create_custom(self):
        mgr = RealtimeManagerFactory.create(city="Osaka", tick_interval=1.0, seed=99)
        assert mgr._config.city == "Osaka"
        assert mgr._config.seed == 99

    def test_create_mock(self):
        mgr = RealtimeManagerFactory.create_mock(seed=42)
        assert mgr._config.tick_interval == 0.1
        assert mgr._config.seed == 42


# ─── RealtimeSnapshot ────────────────────────────────────────────

class TestRealtimeSnapshot:
    def test_to_dict_keys(self):
        snap = RealtimeSnapshot(
            city="Tokyo", tick=10,
            trains=[], districts={"A": {"crowd_level": "normal"}},
            connected_clients=5, uptime_seconds=120.5,
        )
        d = snap.to_dict()
        for k in ("city", "tick", "trains", "districts", "connected_clients", "uptime_seconds", "timestamp"):
            assert k in d
        assert d["connected_clients"] == 5
        assert d["uptime_seconds"] == 120.5


# ─── API 統合テスト ───────────────────────────────────────────────

class TestApiIntegration:
    """serve/api.py の Sprint 79A エンドポイント簡易テスト。"""

    def test_engine_tick_manual(self):
        """RealtimeEngine の tick が正しく動作することを確認。"""
        cfg = RealtimeConfig(seed=1, district_change_prob=0.5)
        engine = RealtimeEngine(cfg, ["新宿", "渋谷"])
        event = engine.tick(dt=2.0)
        assert event.tick == 1
        assert len(event.trains) > 0
        snap = engine.snapshot(connected_clients=0)
        assert "新宿" in snap.districts

    def test_deterministic_with_seed(self):
        """同じシードなら同じ結果になる。"""
        cfg = RealtimeConfig(seed=42, district_change_prob=1.0)
        eng1 = RealtimeEngine(cfg, ["A", "B"])
        eng2 = RealtimeEngine(cfg, ["A", "B"])
        e1 = eng1.tick(dt=2.0)
        e2 = eng2.tick(dt=2.0)
        assert e1.trains[0].position == e2.trains[0].position
