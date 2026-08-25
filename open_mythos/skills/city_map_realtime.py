"""
Sprint 79A — 都市マップ WebSocket リアルタイム更新

Sprint 78 の 3D 都市マップ (city_map_viz.py) に
WebSocket Push チャンネルを追加する。

アーキテクチャ:
    クライアント ←── ws:// ──► RealtimeManager ◄── RealtimeEngine (tick)
                                      │
                               RealtimeSession × N

イベント種別 (UpdateType):
    TRAIN_POSITION  — 電車位置 (route, t: 0.0–1.0, speed, delay)
    CROWD_DENSITY   — 地区混雑度の変化
    ENERGY_LOAD     — エネルギー負荷変化
    NOISE_LEVEL     — 騒音レベル変化
    DISASTER_ALERT  — 災害アラート発令/解除
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ─── 定数 ──────────────────────────────────────────────────────────

TICK_INTERVAL = 2.0          # 秒：デフォルト更新間隔
MAX_SESSIONS  = 100          # 最大同時接続数
HISTORY_SIZE  = 50           # 1セッションあたりのイベント履歴


# ─── Enums ────────────────────────────────────────────────────────

class UpdateType(str, Enum):
    TRAIN_POSITION = "train_position"
    CROWD_DENSITY  = "crowd_density"
    ENERGY_LOAD    = "energy_load"
    NOISE_LEVEL    = "noise_level"
    DISASTER_ALERT = "disaster_alert"


class TrainLine(str, Enum):
    YAMANOTE  = "yamanote"    # 山手線（環状）
    CHUO      = "chuo"        # 中央線
    TOKAIDO   = "tokaido"     # 東海道線


class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED  = "connected"
    CLOSED     = "closed"


# ─── Data Classes ─────────────────────────────────────────────────

@dataclass
class TrainPosition:
    """電車の現在位置。"""
    line:          TrainLine
    train_id:      str
    position:      float      # 0.0–1.0 (路線上の進行率)
    speed_kmh:     float      # km/h
    delay_minutes: int        # 遅延分数
    direction:     int        # +1 / -1
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "line": self.line.value,
            "train_id": self.train_id,
            "position": round(self.position, 4),
            "speed_kmh": round(self.speed_kmh, 1),
            "delay_minutes": self.delay_minutes,
            "direction": self.direction,
            "timestamp": self.timestamp,
        }


@dataclass
class DistrictUpdate:
    """地区レベルの値変化イベント。"""
    district_name: str
    update_type:   UpdateType
    old_value:     Optional[str]
    new_value:     str
    numeric:       float         # 0.0–1.0 の正規化値（グラフ用）
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "district_name": self.district_name,
            "update_type": self.update_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "numeric": round(self.numeric, 3),
            "timestamp": self.timestamp,
        }


@dataclass
class RealtimeEvent:
    """1 tick で生成される複合イベント。"""
    event_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    city:           str = "Tokyo"
    tick:           int = 0
    trains:         List[TrainPosition] = field(default_factory=list)
    district_updates: List[DistrictUpdate] = field(default_factory=list)
    timestamp:      float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "city": self.city,
            "tick": self.tick,
            "trains": [t.to_dict() for t in self.trains],
            "district_updates": [d.to_dict() for d in self.district_updates],
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RealtimeSnapshot:
    """現在の全状態スナップショット。"""
    city:      str
    tick:      int
    trains:    List[TrainPosition]
    districts: Dict[str, Dict[str, Any]]   # district_name → 現在値
    connected_clients: int
    uptime_seconds: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "tick": self.tick,
            "trains": [t.to_dict() for t in self.trains],
            "districts": self.districts,
            "connected_clients": self.connected_clients,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class RealtimeConfig:
    """リアルタイムエンジンの設定。"""
    city:           str   = "Tokyo"
    tick_interval:  float = TICK_INTERVAL    # 秒
    trains_per_line: int  = 3                # 路線ごとの電車数
    district_change_prob: float = 0.15       # 1 tick での地区値変化確率
    disaster_prob:  float = 0.01             # 1 tick での災害アラート確率
    seed:           Optional[int] = None


# ─── 電車スケジューラー ────────────────────────────────────────────

_TRAIN_SPEEDS = {
    TrainLine.YAMANOTE: 50.0,
    TrainLine.CHUO:     80.0,
    TrainLine.TOKAIDO:  110.0,
}

_TRAIN_DELAYS = [0, 0, 0, 0, 1, 2, 3, 5]   # 遅延分数の重み付き候補


class TrainScheduler:
    """路線ごとの電車位置をシミュレーション。"""

    def __init__(self, config: RealtimeConfig):
        self._config = config
        rng_seed = config.seed if config.seed is not None else 42
        self._rng = random.Random(rng_seed)
        self._trains: Dict[str, TrainPosition] = {}
        self._init_trains()

    def _init_trains(self) -> None:
        for line in TrainLine:
            base_speed = _TRAIN_SPEEDS[line]
            for i in range(self._config.trains_per_line):
                tid = f"{line.value}-{i:02d}"
                pos = i / self._config.trains_per_line  # 均等配置
                direction = 1 if i % 2 == 0 else -1
                self._trains[tid] = TrainPosition(
                    line=line,
                    train_id=tid,
                    position=pos % 1.0,
                    speed_kmh=base_speed + self._rng.uniform(-5, 5),
                    delay_minutes=self._rng.choice(_TRAIN_DELAYS),
                    direction=direction,
                )

    def tick(self, dt: float) -> List[TrainPosition]:
        """dt 秒分だけ電車を進め、現在位置リストを返す。"""
        updated: List[TrainPosition] = []
        for tid, train in self._trains.items():
            base_speed = _TRAIN_SPEEDS[train.line]
            # 確率的に速度ゆらぎ
            if self._rng.random() < 0.1:
                train.speed_kmh = base_speed + self._rng.uniform(-10, 10)
                train.speed_kmh = max(10.0, min(train.speed_kmh, base_speed * 1.3))
            # 遅延変化（稀に）
            if self._rng.random() < 0.03:
                train.delay_minutes = self._rng.choice(_TRAIN_DELAYS)
            # 位置更新（km/h → 位置単位/秒; 路線全長を 100km と仮定）
            advance = (train.speed_kmh / 100.0) * (dt / 3600.0)
            train.position = (train.position + advance * train.direction) % 1.0
            train.timestamp = time.time()
            updated.append(TrainPosition(**train.__dict__))
        return updated

    def get_all(self) -> List[TrainPosition]:
        return list(self._trains.values())


# ─── 地区状態シミュレーター ────────────────────────────────────────

_CROWD_LEVELS   = ["sparse", "normal", "crowded", "packed"]
_ENERGY_LEVELS  = ["normal", "high", "critical"]
_NOISE_LEVELS   = ["compliant", "near_limit", "violation"]
_DISASTER_LEVELS = ["info", "watch", "warning", "critical"]

# 値 → numeric (0.0–1.0) マッピング
_CROWD_NUMERIC   = {"sparse": 0.1, "normal": 0.4, "crowded": 0.7, "packed": 1.0}
_ENERGY_NUMERIC  = {"normal": 0.2, "high": 0.6, "critical": 1.0}
_NOISE_NUMERIC   = {"compliant": 0.1, "near_limit": 0.6, "violation": 1.0}
_DISASTER_NUMERIC = {"info": 0.25, "watch": 0.5, "warning": 0.75, "critical": 1.0}


class DistrictStateSimulator:
    """地区ごとの crowd / energy / noise / disaster をシミュレーション。"""

    def __init__(self, district_names: List[str], config: RealtimeConfig):
        rng_seed = (config.seed or 0) + 1
        self._rng = random.Random(rng_seed)
        self._config = config
        # 初期状態
        self._state: Dict[str, Dict[str, Optional[str]]] = {}
        for name in district_names:
            self._state[name] = {
                "crowd_level":    self._rng.choice(_CROWD_LEVELS),
                "energy_status":  self._rng.choice(_ENERGY_LEVELS),
                "noise_status":   self._rng.choice(_NOISE_LEVELS),
                "disaster_level": None,
            }

    def tick(self) -> List[DistrictUpdate]:
        updates: List[DistrictUpdate] = []
        p = self._config.district_change_prob
        dp = self._config.disaster_prob

        for name, state in self._state.items():
            # crowd_level
            if self._rng.random() < p:
                new = self._rng.choice(_CROWD_LEVELS)
                if new != state["crowd_level"]:
                    updates.append(DistrictUpdate(
                        district_name=name,
                        update_type=UpdateType.CROWD_DENSITY,
                        old_value=state["crowd_level"],
                        new_value=new,
                        numeric=_CROWD_NUMERIC[new],
                    ))
                    state["crowd_level"] = new

            # energy_status
            if self._rng.random() < p:
                new = self._rng.choice(_ENERGY_LEVELS)
                if new != state["energy_status"]:
                    updates.append(DistrictUpdate(
                        district_name=name,
                        update_type=UpdateType.ENERGY_LOAD,
                        old_value=state["energy_status"],
                        new_value=new,
                        numeric=_ENERGY_NUMERIC[new],
                    ))
                    state["energy_status"] = new

            # noise_status
            if self._rng.random() < p * 0.5:
                new = self._rng.choice(_NOISE_LEVELS)
                if new != state["noise_status"]:
                    updates.append(DistrictUpdate(
                        district_name=name,
                        update_type=UpdateType.NOISE_LEVEL,
                        old_value=state["noise_status"],
                        new_value=new,
                        numeric=_NOISE_NUMERIC[new],
                    ))
                    state["noise_status"] = new

            # disaster_alert (稀に発令・解除)
            if self._rng.random() < dp:
                if state["disaster_level"] is None:
                    new = self._rng.choice(_DISASTER_LEVELS[:2])  # info/watch のみ
                    updates.append(DistrictUpdate(
                        district_name=name,
                        update_type=UpdateType.DISASTER_ALERT,
                        old_value=None,
                        new_value=new,
                        numeric=_DISASTER_NUMERIC[new],
                    ))
                    state["disaster_level"] = new
                else:
                    # 解除
                    updates.append(DistrictUpdate(
                        district_name=name,
                        update_type=UpdateType.DISASTER_ALERT,
                        old_value=state["disaster_level"],
                        new_value="none",
                        numeric=0.0,
                    ))
                    state["disaster_level"] = None

        return updates

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(state) for name, state in self._state.items()}


# ─── リアルタイムセッション ────────────────────────────────────────

class RealtimeSession:
    """1 クライアントとの WebSocket セッション。"""

    def __init__(self, session_id: str, send_fn: Callable[[str], Any]):
        self.session_id = session_id
        self._send = send_fn
        self.state = ConnectionState.CONNECTING
        self.connected_at = time.time()
        self.messages_sent = 0
        self._history: List[RealtimeEvent] = []

    async def send(self, event: RealtimeEvent) -> bool:
        """イベントを送信。失敗時は False を返す。"""
        try:
            await self._send(event.to_json())
            self.messages_sent += 1
            self._history.append(event)
            if len(self._history) > HISTORY_SIZE:
                self._history.pop(0)
            return True
        except Exception:
            self.state = ConnectionState.CLOSED
            return False

    def get_history(self) -> List[RealtimeEvent]:
        return list(self._history)

    @property
    def uptime(self) -> float:
        return time.time() - self.connected_at


# ─── リアルタイムエンジン ─────────────────────────────────────────

class RealtimeEngine:
    """
    tick() を呼ぶたびに RealtimeEvent を生成するステートマシン。
    asyncio に依存しない純粋 Python — テストしやすい設計。
    """

    def __init__(self, config: RealtimeConfig, district_names: Optional[List[str]] = None):
        self._config = config
        if district_names is None:
            district_names = [f"District-{i}" for i in range(5)]
        self._district_names = district_names
        self._train_scheduler = TrainScheduler(config)
        self._district_sim = DistrictStateSimulator(district_names, config)
        self._tick_count = 0
        self._last_tick_time = time.time()
        self._start_time = time.time()

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    def tick(self, dt: Optional[float] = None) -> RealtimeEvent:
        """1 tick を実行してイベントを返す。dt=None で自動計算。"""
        now = time.time()
        if dt is None:
            dt = now - self._last_tick_time
        self._last_tick_time = now
        self._tick_count += 1

        trains = self._train_scheduler.tick(dt)
        district_updates = self._district_sim.tick()

        return RealtimeEvent(
            city=self._config.city,
            tick=self._tick_count,
            trains=trains,
            district_updates=district_updates,
        )

    def snapshot(self, connected_clients: int = 0) -> RealtimeSnapshot:
        """現在の全状態スナップショット。"""
        return RealtimeSnapshot(
            city=self._config.city,
            tick=self._tick_count,
            trains=self._train_scheduler.get_all(),
            districts=self._district_sim.get_snapshot(),
            connected_clients=connected_clients,
            uptime_seconds=self.uptime,
        )

    def get_config(self) -> RealtimeConfig:
        return self._config


# ─── リアルタイムマネージャー ─────────────────────────────────────

class RealtimeManager:
    """
    WebSocket セッション管理 + エンジン駆動の中央コーディネーター。
    FastAPI の WebSocket ハンドラから呼び出す。
    """

    def __init__(self, config: Optional[RealtimeConfig] = None):
        if config is None:
            config = RealtimeConfig()
        self._config = config
        self._engine: Optional[RealtimeEngine] = None
        self._sessions: Dict[str, RealtimeSession] = {}
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    # ── セッション管理 ──────────────────────────────────────────────

    def register_session(
        self, session_id: str, send_fn: Callable[[str], Any]
    ) -> RealtimeSession:
        if len(self._sessions) >= MAX_SESSIONS:
            raise RuntimeError(f"最大接続数 ({MAX_SESSIONS}) に達しました")
        session = RealtimeSession(session_id, send_fn)
        session.state = ConnectionState.CONNECTED
        self._sessions[session_id] = session
        return session

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def get_session(self, session_id: str) -> Optional[RealtimeSession]:
        return self._sessions.get(session_id)

    # ── エンジン管理 ────────────────────────────────────────────────

    def init_engine(
        self,
        district_names: Optional[List[str]] = None,
        config: Optional[RealtimeConfig] = None,
    ) -> RealtimeEngine:
        if config is not None:
            self._config = config
        self._engine = RealtimeEngine(self._config, district_names)
        return self._engine

    @property
    def engine(self) -> Optional[RealtimeEngine]:
        return self._engine

    # ── ブロードキャスト ─────────────────────────────────────────────

    async def broadcast(self, event: RealtimeEvent) -> int:
        """全セッションにイベントを送信。送信成功数を返す。"""
        dead: Set[str] = set()
        sent = 0
        for sid, session in self._sessions.items():
            ok = await session.send(event)
            if ok:
                sent += 1
            else:
                dead.add(sid)
        for sid in dead:
            self.remove_session(sid)
        return sent

    # ── バックグラウンドループ ───────────────────────────────────────

    async def _run_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._config.tick_interval)
            if self._engine is None or not self._sessions:
                continue
            event = self._engine.tick()
            await self.broadcast(event)

    async def start(self) -> None:
        if self._running:
            return
        if self._engine is None:
            self.init_engine()
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ── スナップショット ─────────────────────────────────────────────

    def snapshot(self) -> Optional[RealtimeSnapshot]:
        if self._engine is None:
            return None
        return self._engine.snapshot(connected_clients=self.session_count)

    # ── ステータス ───────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running": self._running,
            "connected_clients": self.session_count,
            "tick_count": self._engine.tick_count if self._engine else 0,
            "uptime_seconds": self._engine.uptime if self._engine else 0.0,
            "tick_interval": self._config.tick_interval,
            "city": self._config.city,
        }


# ─── ファクトリー ──────────────────────────────────────────────────

class RealtimeManagerFactory:
    """テスト / 本番で差し替えやすいファクトリー。"""

    @staticmethod
    def create(
        city: str = "Tokyo",
        tick_interval: float = TICK_INTERVAL,
        trains_per_line: int = 3,
        district_change_prob: float = 0.15,
        seed: Optional[int] = None,
    ) -> RealtimeManager:
        config = RealtimeConfig(
            city=city,
            tick_interval=tick_interval,
            trains_per_line=trains_per_line,
            district_change_prob=district_change_prob,
            seed=seed,
        )
        return RealtimeManager(config=config)

    @staticmethod
    def create_mock(seed: int = 0) -> RealtimeManager:
        """テスト用: 決定論的なシード固定マネージャー。"""
        return RealtimeManagerFactory.create(
            tick_interval=0.1,
            trains_per_line=2,
            district_change_prob=0.5,
            seed=seed,
        )
