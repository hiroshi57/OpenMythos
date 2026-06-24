"""
Sprint 72B — インタラクティブ路線データ編集

路線・駅・地質層を CRUD で管理する MapEditor。
CityMapStore をラップして変更履歴を保持する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from open_mythos.skills.city_map import (
    CityName, LineType, GeologyLayerType,
    GeoCoord, Station, MetroLine, GeologyLayer, CityMapStore,
)


# ─── EditAction ───────────────────────────────────────────────────


class EditAction(str, Enum):
    ADD_LINE = "add_line"
    UPDATE_LINE = "update_line"
    DELETE_LINE = "delete_line"
    ADD_STATION = "add_station"
    UPDATE_STATION = "update_station"
    DELETE_STATION = "delete_station"
    ADD_GEOLOGY = "add_geology"
    UPDATE_GEOLOGY = "update_geology"
    DELETE_GEOLOGY = "delete_geology"


@dataclass
class EditRecord:
    """変更履歴の 1 件"""
    action: EditAction
    target_id: str
    snapshot_before: Optional[dict]
    snapshot_after: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "target_id": self.target_id,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
        }


@dataclass
class EditResult:
    """CRUD 操作の結果"""
    success: bool
    action: EditAction
    target_id: str
    message: str
    data: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "action": self.action.value,
            "target_id": self.target_id,
            "message": self.message,
            "data": self.data,
        }


# ─── MapEditor ────────────────────────────────────────────────────


class MapEditor:
    """CityMapStore に対して CRUD 操作を行い変更履歴を管理するエディタ"""

    def __init__(self, store: CityMapStore) -> None:
        self._store = store
        self._history: List[EditRecord] = []

    @property
    def history(self) -> List[EditRecord]:
        return list(self._history)

    def history_dicts(self) -> List[dict]:
        return [r.to_dict() for r in self._history]

    # ── MetroLine CRUD ────────────────────────────────────────────

    def add_line(self, line: MetroLine) -> EditResult:
        if self._store.lines.get(line.id) is not None:
            return EditResult(False, EditAction.ADD_LINE, line.id,
                              f"Line already exists: {line.id}")
        self._store.lines.add(line)
        rec = EditRecord(EditAction.ADD_LINE, line.id, None, line.to_dict())
        self._history.append(rec)
        return EditResult(True, EditAction.ADD_LINE, line.id,
                          "Line added.", data=line.to_dict())

    def update_line(self, line_id: str, **kwargs) -> EditResult:
        line = self._store.lines.get(line_id)
        if line is None:
            return EditResult(False, EditAction.UPDATE_LINE, line_id,
                              f"Line not found: {line_id}")
        before = line.to_dict()
        allowed = {"name", "name_en", "color", "total_length_km",
                   "opened_year", "station_ids", "line_type"}
        for k, v in kwargs.items():
            if k in allowed:
                if k == "line_type":
                    v = LineType(v)
                setattr(line, k, v)
        rec = EditRecord(EditAction.UPDATE_LINE, line_id, before, line.to_dict())
        self._history.append(rec)
        return EditResult(True, EditAction.UPDATE_LINE, line_id,
                          "Line updated.", data=line.to_dict())

    def delete_line(self, line_id: str) -> EditResult:
        line = self._store.lines.get(line_id)
        if line is None:
            return EditResult(False, EditAction.DELETE_LINE, line_id,
                              f"Line not found: {line_id}")
        before = line.to_dict()
        self._store.lines._data.pop(line_id)
        rec = EditRecord(EditAction.DELETE_LINE, line_id, before, None)
        self._history.append(rec)
        return EditResult(True, EditAction.DELETE_LINE, line_id,
                          "Line deleted.")

    # ── Station CRUD ──────────────────────────────────────────────

    def add_station(self, station: Station) -> EditResult:
        if self._store.stations.get(station.id) is not None:
            return EditResult(False, EditAction.ADD_STATION, station.id,
                              f"Station already exists: {station.id}")
        self._store.stations.add(station)
        rec = EditRecord(EditAction.ADD_STATION, station.id, None, station.to_dict())
        self._history.append(rec)
        return EditResult(True, EditAction.ADD_STATION, station.id,
                          "Station added.", data=station.to_dict())

    def update_station(self, station_id: str, **kwargs) -> EditResult:
        station = self._store.stations.get(station_id)
        if station is None:
            return EditResult(False, EditAction.UPDATE_STATION, station_id,
                              f"Station not found: {station_id}")
        before = station.to_dict()
        allowed = {"name", "name_en", "depth_m", "platform_count", "opened_year"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(station, k, v)
        rec = EditRecord(EditAction.UPDATE_STATION, station_id, before, station.to_dict())
        self._history.append(rec)
        return EditResult(True, EditAction.UPDATE_STATION, station_id,
                          "Station updated.", data=station.to_dict())

    def delete_station(self, station_id: str) -> EditResult:
        station = self._store.stations.get(station_id)
        if station is None:
            return EditResult(False, EditAction.DELETE_STATION, station_id,
                              f"Station not found: {station_id}")
        before = station.to_dict()
        self._store.stations._data.pop(station_id)
        rec = EditRecord(EditAction.DELETE_STATION, station_id, before, None)
        self._history.append(rec)
        return EditResult(True, EditAction.DELETE_STATION, station_id,
                          "Station deleted.")

    # ── GeologyLayer CRUD ─────────────────────────────────────────

    def add_geology(self, layer: GeologyLayer) -> EditResult:
        if self._store.geology.get(layer.id) is not None:
            return EditResult(False, EditAction.ADD_GEOLOGY, layer.id,
                              f"GeologyLayer already exists: {layer.id}")
        self._store.geology.add(layer)
        rec = EditRecord(EditAction.ADD_GEOLOGY, layer.id, None, layer.to_dict())
        self._history.append(rec)
        return EditResult(True, EditAction.ADD_GEOLOGY, layer.id,
                          "GeologyLayer added.", data=layer.to_dict())

    def update_geology(self, layer_id: str, **kwargs) -> EditResult:
        layer = self._store.geology.get(layer_id)
        if layer is None:
            return EditResult(False, EditAction.UPDATE_GEOLOGY, layer_id,
                              f"GeologyLayer not found: {layer_id}")
        before = layer.to_dict()
        allowed = {"name", "depth_from_m", "depth_to_m", "color", "n_value"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(layer, k, v)
        rec = EditRecord(EditAction.UPDATE_GEOLOGY, layer_id, before, layer.to_dict())
        self._history.append(rec)
        return EditResult(True, EditAction.UPDATE_GEOLOGY, layer_id,
                          "GeologyLayer updated.", data=layer.to_dict())

    def delete_geology(self, layer_id: str) -> EditResult:
        layer = self._store.geology.get(layer_id)
        if layer is None:
            return EditResult(False, EditAction.DELETE_GEOLOGY, layer_id,
                              f"GeologyLayer not found: {layer_id}")
        before = layer.to_dict()
        self._store.geology._data.pop(layer_id)
        rec = EditRecord(EditAction.DELETE_GEOLOGY, layer_id, before, None)
        self._history.append(rec)
        return EditResult(True, EditAction.DELETE_GEOLOGY, layer_id,
                          "GeologyLayer deleted.")

    # ── Utility ───────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "lines": len(self._store.lines),
            "stations": len(self._store.stations),
            "geology_layers": len(self._store.geology),
            "history_count": len(self._history),
        }
