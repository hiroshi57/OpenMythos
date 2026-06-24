"""
Sprint 73B — 経路探索エンジン (乗換案内)

2駅間の最短経路をグラフ探索 (Dijkstra) で求める。
外部ライブラリなし。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from open_mythos.skills.city_map import (
    CityName, MetroLine, Station, CityMapStore,
)


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class RouteEdge:
    """路線グラフの辺 (隣接駅間)"""
    from_id: str
    to_id: str
    line_id: str
    cost: float          # 所要時間 (分) or 距離コスト

    def to_dict(self) -> dict:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "line_id": self.line_id,
            "cost": self.cost,
        }


@dataclass
class RouteStep:
    """経路の 1 ステップ"""
    station_id: str
    station_name: str
    line_id: str
    cumulative_cost: float
    transfer: bool = False    # この駅で乗換あり

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "line_id": self.line_id,
            "cumulative_cost": round(self.cumulative_cost, 2),
            "transfer": self.transfer,
        }


@dataclass
class RouteResult:
    """経路探索結果"""
    from_id: str
    to_id: str
    found: bool
    steps: List[RouteStep]
    total_cost: float
    transfer_count: int
    message: str

    def to_dict(self) -> dict:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "found": self.found,
            "steps": [s.to_dict() for s in self.steps],
            "total_cost": round(self.total_cost, 2),
            "transfer_count": self.transfer_count,
            "message": self.message,
        }


# ─── RouteGraph ───────────────────────────────────────────────────


class RouteGraph:
    """路線グラフ (隣接リスト)"""

    def __init__(self) -> None:
        # station_id → [(neighbor_id, line_id, cost)]
        self._adj: Dict[str, List[Tuple[str, str, float]]] = {}
        self._stations: Dict[str, Station] = {}

    def add_station(self, station: Station) -> None:
        self._stations[station.id] = station
        if station.id not in self._adj:
            self._adj[station.id] = []

    def add_edge(self, from_id: str, to_id: str, line_id: str, cost: float) -> None:
        if from_id not in self._adj:
            self._adj[from_id] = []
        if to_id not in self._adj:
            self._adj[to_id] = []
        self._adj[from_id].append((to_id, line_id, cost))
        self._adj[to_id].append((from_id, line_id, cost))  # 双方向

    def neighbors(self, station_id: str) -> List[Tuple[str, str, float]]:
        return self._adj.get(station_id, [])

    def get_station(self, station_id: str) -> Optional[Station]:
        return self._stations.get(station_id)

    def station_ids(self) -> List[str]:
        return list(self._adj.keys())

    def edge_count(self) -> int:
        return sum(len(v) for v in self._adj.values()) // 2

    def has_station(self, station_id: str) -> bool:
        return station_id in self._adj


# ─── GraphBuilder ─────────────────────────────────────────────────


class RouteGraphBuilder:
    """CityMapStore から RouteGraph を構築するビルダー"""

    TRANSFER_COST: float = 3.0    # 乗換コスト (分相当)
    DEFAULT_EDGE_COST: float = 2.0  # 隣接駅間のデフォルトコスト (分)

    @classmethod
    def build(cls, store: CityMapStore, city: Optional[CityName] = None) -> RouteGraph:
        graph = RouteGraph()

        if city is not None:
            lines = store.lines.list_by_city(city)
            stations_iter = store.stations.list_by_city(city)
        else:
            lines = store.lines.all()
            stations_iter = store.stations.all()

        # 全駅をグラフに登録
        for st in stations_iter:
            graph.add_station(st)

        # 路線ごとに隣接辺を追加
        for line in lines:
            ids = line.station_ids
            for i in range(len(ids) - 1):
                a, b = ids[i], ids[i + 1]
                # 両端駅がグラフに存在する場合のみ追加
                if graph.has_station(a) and graph.has_station(b):
                    graph.add_edge(a, b, line.id, cls.DEFAULT_EDGE_COST)
                elif graph.has_station(a) or graph.has_station(b):
                    # 片方だけ存在する場合もエッジは追加しない
                    pass

        return graph


# ─── RouteFinder ──────────────────────────────────────────────────


class RouteFinder:
    """Dijkstra 法で最短経路を探索するクラス"""

    TRANSFER_COST: float = RouteGraphBuilder.TRANSFER_COST

    def __init__(self, graph: RouteGraph) -> None:
        self._graph = graph

    def find(self, from_id: str, to_id: str) -> RouteResult:
        """from_id → to_id の最短経路を返す"""
        if not self._graph.has_station(from_id):
            return RouteResult(from_id, to_id, False, [], 0.0, 0,
                               f"出発駅が見つかりません: {from_id}")
        if not self._graph.has_station(to_id):
            return RouteResult(from_id, to_id, False, [], 0.0, 0,
                               f"到着駅が見つかりません: {to_id}")
        if from_id == to_id:
            st = self._graph.get_station(from_id)
            name = st.name if st else from_id
            step = RouteStep(from_id, name, "", 0.0)
            return RouteResult(from_id, to_id, True, [step], 0.0, 0, "同一駅です。")

        # Dijkstra
        # heap: (cost, station_id, current_line_id, path)
        heap: List[Tuple[float, str, str, List[Tuple[str, str, str]]]] = [
            (0.0, from_id, "", [(from_id, "", 0.0)])
        ]
        visited: Set[Tuple[str, str]] = set()  # (station_id, line_id)

        while heap:
            cost, curr_id, curr_line, path = heapq.heappop(heap)

            state = (curr_id, curr_line)
            if state in visited:
                continue
            visited.add(state)

            if curr_id == to_id:
                return self._build_result(from_id, to_id, path, cost)

            for next_id, line_id, edge_cost in self._graph.neighbors(curr_id):
                transfer_penalty = self.TRANSFER_COST if (curr_line and line_id != curr_line) else 0.0
                new_cost = cost + edge_cost + transfer_penalty
                new_path = path + [(next_id, line_id, new_cost)]
                next_state = (next_id, line_id)
                if next_state not in visited:
                    heapq.heappush(heap, (new_cost, next_id, line_id, new_path))

        return RouteResult(from_id, to_id, False, [], 0.0, 0, "経路が見つかりません。")

    def _build_result(
        self,
        from_id: str,
        to_id: str,
        path: List[Tuple[str, str, float]],
        total_cost: float,
    ) -> RouteResult:
        steps: List[RouteStep] = []
        transfer_count = 0
        prev_line = ""

        for i, (sid, line_id, cum_cost) in enumerate(path):
            st = self._graph.get_station(sid)
            name = st.name if st else sid
            is_transfer = (i > 0 and prev_line and line_id and line_id != prev_line)
            if is_transfer:
                transfer_count += 1
            steps.append(RouteStep(
                station_id=sid,
                station_name=name,
                line_id=line_id,
                cumulative_cost=cum_cost,
                transfer=is_transfer,
            ))
            if line_id:
                prev_line = line_id

        msg = f"{len(steps)} 駅、乗換 {transfer_count} 回、約 {total_cost:.0f} 分"
        return RouteResult(from_id, to_id, True, steps, total_cost, transfer_count, msg)
