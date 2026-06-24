"""
Sprint 73C — 地図データインポート (CSV/GeoJSON)

外部データを CityMapStore に一括インポートする。
外部ライブラリなし (csv / json モジュールのみ)。
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from open_mythos.skills.city_map import (
    CityName, LineType, GeologyLayerType,
    GeoCoord, Station, MetroLine, GeologyLayer, CityMapStore,
)


# ─── Import Result ────────────────────────────────────────────────


@dataclass
class ImportError:
    row: int
    message: str
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        return {"row": self.row, "message": self.message, "raw": self.raw}


@dataclass
class ImportResult:
    """インポート処理の結果"""
    source_type: str          # "csv_stations" / "csv_lines" / "csv_geology" / "geojson"
    total_rows: int
    imported: int
    skipped: int
    errors: List[ImportError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "total_rows": self.total_rows,
            "imported": self.imported,
            "skipped": self.skipped,
            "errors": [e.to_dict() for e in self.errors],
            "success": self.success,
        }


# ─── CSV Parsers ──────────────────────────────────────────────────


class StationCSVImporter:
    """
    駅 CSV インポーター

    ヘッダー: id,name,name_en,line_id,city,lat,lon,depth_m,platform_count,opened_year
    """

    REQUIRED = {"id", "name", "name_en", "line_id", "city", "lat", "lon", "depth_m"}

    def __init__(self, store: CityMapStore) -> None:
        self._store = store

    def import_csv(self, csv_text: str) -> ImportResult:
        reader = csv.DictReader(io.StringIO(csv_text))
        result = ImportResult("csv_stations", 0, 0, 0)

        missing = self.REQUIRED - set(reader.fieldnames or [])
        if missing:
            result.errors.append(ImportError(0, f"必須カラム不足: {missing}"))
            return result

        for i, row in enumerate(reader, start=1):
            result.total_rows += 1
            try:
                city = CityName(row["city"].strip())
                station = Station(
                    id=row["id"].strip(),
                    name=row["name"].strip(),
                    name_en=row["name_en"].strip(),
                    line_id=row["line_id"].strip(),
                    city=city,
                    coord=GeoCoord(float(row["lat"]), float(row["lon"])),
                    depth_m=float(row["depth_m"]),
                    platform_count=int(row.get("platform_count") or 2),
                    opened_year=int(row["opened_year"]) if row.get("opened_year") else None,
                )
                if self._store.stations.get(station.id) is not None:
                    result.skipped += 1
                    continue
                self._store.stations.add(station)
                result.imported += 1
            except (ValueError, KeyError) as e:
                result.errors.append(ImportError(i, str(e), str(row)))

        return result


class LineCSVImporter:
    """
    路線 CSV インポーター

    ヘッダー: id,name,name_en,city,line_type,color,total_length_km,opened_year
    """

    REQUIRED = {"id", "name", "name_en", "city", "line_type", "color"}

    def __init__(self, store: CityMapStore) -> None:
        self._store = store

    def import_csv(self, csv_text: str) -> ImportResult:
        reader = csv.DictReader(io.StringIO(csv_text))
        result = ImportResult("csv_lines", 0, 0, 0)

        missing = self.REQUIRED - set(reader.fieldnames or [])
        if missing:
            result.errors.append(ImportError(0, f"必須カラム不足: {missing}"))
            return result

        for i, row in enumerate(reader, start=1):
            result.total_rows += 1
            try:
                city = CityName(row["city"].strip())
                line_type = LineType(row["line_type"].strip())
                line = MetroLine(
                    id=row["id"].strip(),
                    name=row["name"].strip(),
                    name_en=row["name_en"].strip(),
                    city=city,
                    line_type=line_type,
                    color=row["color"].strip(),
                    total_length_km=float(row.get("total_length_km") or 0.0),
                    opened_year=int(row["opened_year"]) if row.get("opened_year") else None,
                )
                if self._store.lines.get(line.id) is not None:
                    result.skipped += 1
                    continue
                self._store.lines.add(line)
                result.imported += 1
            except (ValueError, KeyError) as e:
                result.errors.append(ImportError(i, str(e), str(row)))

        return result


class GeologyCSVImporter:
    """
    地質層 CSV インポーター

    ヘッダー: id,city,layer_type,name,depth_from_m,depth_to_m,color,n_value
    """

    REQUIRED = {"id", "city", "layer_type", "name", "depth_from_m", "depth_to_m", "color"}

    def __init__(self, store: CityMapStore) -> None:
        self._store = store

    def import_csv(self, csv_text: str) -> ImportResult:
        reader = csv.DictReader(io.StringIO(csv_text))
        result = ImportResult("csv_geology", 0, 0, 0)

        missing = self.REQUIRED - set(reader.fieldnames or [])
        if missing:
            result.errors.append(ImportError(0, f"必須カラム不足: {missing}"))
            return result

        for i, row in enumerate(reader, start=1):
            result.total_rows += 1
            try:
                city = CityName(row["city"].strip())
                layer_type = GeologyLayerType(row["layer_type"].strip())
                n_val_raw = row.get("n_value", "").strip()
                layer = GeologyLayer(
                    id=row["id"].strip(),
                    city=city,
                    layer_type=layer_type,
                    name=row["name"].strip(),
                    depth_from_m=float(row["depth_from_m"]),
                    depth_to_m=float(row["depth_to_m"]),
                    color=row["color"].strip(),
                    n_value=float(n_val_raw) if n_val_raw else None,
                )
                if self._store.geology.get(layer.id) is not None:
                    result.skipped += 1
                    continue
                self._store.geology.add(layer)
                result.imported += 1
            except (ValueError, KeyError) as e:
                result.errors.append(ImportError(i, str(e), str(row)))

        return result


# ─── GeoJSON Importer ─────────────────────────────────────────────


class GeoJSONImporter:
    """
    GeoJSON FeatureCollection インポーター

    Feature の geometry.type == "Point" のみ対応 (駅データ)。
    properties に id/name/name_en/line_id/city/depth_m が必要。
    """

    REQUIRED_PROPS = {"id", "name", "name_en", "line_id", "city", "depth_m"}

    def __init__(self, store: CityMapStore) -> None:
        self._store = store

    def import_geojson(self, geojson_text: str) -> ImportResult:
        result = ImportResult("geojson", 0, 0, 0)
        try:
            data = json.loads(geojson_text)
        except json.JSONDecodeError as e:
            result.errors.append(ImportError(0, f"JSON parse error: {e}"))
            return result

        if data.get("type") != "FeatureCollection":
            result.errors.append(ImportError(0, "type が FeatureCollection ではありません"))
            return result

        features = data.get("features", [])
        for i, feat in enumerate(features, start=1):
            result.total_rows += 1
            try:
                if feat.get("type") != "Feature":
                    result.skipped += 1
                    continue
                geom = feat.get("geometry", {})
                if geom.get("type") != "Point":
                    result.skipped += 1
                    continue
                coords = geom.get("coordinates", [])
                if len(coords) < 2:
                    result.errors.append(ImportError(i, "coordinates が不足しています"))
                    continue
                lon, lat = float(coords[0]), float(coords[1])

                props = feat.get("properties", {})
                missing = self.REQUIRED_PROPS - set(props.keys())
                if missing:
                    result.errors.append(ImportError(i, f"properties 不足: {missing}"))
                    continue

                city = CityName(str(props["city"]).strip())
                station = Station(
                    id=str(props["id"]).strip(),
                    name=str(props["name"]).strip(),
                    name_en=str(props["name_en"]).strip(),
                    line_id=str(props["line_id"]).strip(),
                    city=city,
                    coord=GeoCoord(lat, lon),
                    depth_m=float(props["depth_m"]),
                    platform_count=int(props.get("platform_count", 2)),
                    opened_year=int(props["opened_year"]) if props.get("opened_year") else None,
                )
                if self._store.stations.get(station.id) is not None:
                    result.skipped += 1
                    continue
                self._store.stations.add(station)
                result.imported += 1
            except (ValueError, KeyError, TypeError) as e:
                result.errors.append(ImportError(i, str(e)))

        return result


# ─── MapImporter (facade) ─────────────────────────────────────────


class MapImporter:
    """CSV/GeoJSON インポーターの統合ファサード"""

    def __init__(self, store: CityMapStore) -> None:
        self._store = store
        self._station_importer = StationCSVImporter(store)
        self._line_importer = LineCSVImporter(store)
        self._geology_importer = GeologyCSVImporter(store)
        self._geojson_importer = GeoJSONImporter(store)

    def import_stations_csv(self, csv_text: str) -> ImportResult:
        return self._station_importer.import_csv(csv_text)

    def import_lines_csv(self, csv_text: str) -> ImportResult:
        return self._line_importer.import_csv(csv_text)

    def import_geology_csv(self, csv_text: str) -> ImportResult:
        return self._geology_importer.import_csv(csv_text)

    def import_geojson(self, geojson_text: str) -> ImportResult:
        return self._geojson_importer.import_geojson(geojson_text)

    def summary(self) -> dict:
        return {
            "lines": len(self._store.lines),
            "stations": len(self._store.stations),
            "geology_layers": len(self._store.geology),
        }
