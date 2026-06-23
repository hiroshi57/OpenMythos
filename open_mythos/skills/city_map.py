"""
Sprint 71 — 主要都市地図ビジュアライザ (候補A: メトロ断面図データ)

参照: https://tokyo-danmenzu.pages.dev/ （東京地下断面図・3D地質断面ビューア / @chizutodesign）

主要都市（政令指定都市＋首都圏主要市・13都市）の地下鉄路線・駅・地質層データを
GeoJSON 互換のオブジェクトモデルとして提供し、路線ごとの「地下断面 (CrossSection)」を構築する。

ユースケース:
  - 都市の地下鉄路線・駅の一覧取得
  - 路線を横から見た地下断面（駅深度 × 地質層）の生成
  - GTFS (公共交通オープンデータ) からの路線・駅取り込み（フォールバック付き）

オブジェクト:
  GeoPoint              : 緯度経度
  Station               : 駅 (路線内順序・地下深度付き)
  GeologyLayer          : 地質層 (沖積層 / 武蔵野礫層 / 東京層 等)
  Line                  : 路線 (順序付き駅リスト)
  City                  : 都市 (路線リスト + 地質プロファイル)
  CrossSection          : 路線の地下断面 (駅列 + 地質層 + 距離 + 最大深度)
  BaseCityDataSource    : 都市データ源 抽象基底
  SampleCityDataSource  : 同梱サンプル (13都市・外部依存ゼロ・デフォルト)
  GTFSCityDataSource    : GTFS zip から取り込み (失敗時 Sample へフォールバック)
  GeologyModel          : 地質層・地下深度のルールベース推定 (GTFS に深度情報が無いため)
  CrossSectionBuilder   : データ源 + 地質モデル → CrossSection 統合層
  CityMapStore          : CrossSection インメモリストア
  CityMapFactory        : from_sample / from_gtfs / available_cities

設計方針:
  - GTFS が取得できない環境（オフライン・テスト）では SampleCityDataSource に自動フォールバック
  - 地下深度・地質層は GTFS に含まれないため GeologyModel が都市ごとにルールベース推定
  - 外部依存ゼロ（標準ライブラリ urllib / zipfile / csv のみ）。time_series.py の
    「外部依存が無い環境へのフォールバック」哲学に合わせる
"""
from __future__ import annotations

import copy
import csv
import io
import math
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GeoPoint:
    """緯度経度の1点"""
    lat: float
    lon: float

    def to_dict(self) -> Dict[str, Any]:
        return {"lat": round(self.lat, 6), "lon": round(self.lon, 6)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeoPoint":
        return cls(lat=float(d["lat"]), lon=float(d["lon"]))


@dataclass
class Station:
    """
    駅。

    Attributes:
        station_id : 一意ID
        name       : 駅名
        geo        : 緯度経度
        line_id    : 所属路線ID
        order      : 路線内の順序 (0-based, 起点 = 0)
        depth_m    : 地下深度 [m] (正値 = 地下。GTFS に無いため GeologyModel が推定)
        name_kana  : 読み仮名 (任意)
    """
    station_id: str
    name:       str
    geo:        GeoPoint
    line_id:    str
    order:      int
    depth_m:    Optional[float] = None
    name_kana:  Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "name":       self.name,
            "name_kana":  self.name_kana,
            "geo":        self.geo.to_dict(),
            "line_id":    self.line_id,
            "order":      self.order,
            "depth_m":    round(self.depth_m, 2) if self.depth_m is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Station":
        return cls(
            station_id=d["station_id"],
            name=d["name"],
            geo=GeoPoint.from_dict(d["geo"]),
            line_id=d["line_id"],
            order=int(d["order"]),
            depth_m=d.get("depth_m"),
            name_kana=d.get("name_kana"),
        )


@dataclass
class GeologyLayer:
    """
    地質層。

    Attributes:
        layer_id  : 一意ID
        name      : 層名 (例: 沖積層 / 武蔵野礫層 / 東京層)
        top_m     : 上端深度 [m] (地表 = 0、下に行くほど大)
        bottom_m  : 下端深度 [m] (top_m < bottom_m)
        color     : SVG 用カラー (#hex)
        soil_type : 土質区分 (clay / sand / gravel / loam / bedrock)
    """
    layer_id:  str
    name:      str
    top_m:     float
    bottom_m:  float
    color:     str
    soil_type: str

    @property
    def thickness_m(self) -> float:
        return self.bottom_m - self.top_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id":    self.layer_id,
            "name":        self.name,
            "top_m":       round(self.top_m, 2),
            "bottom_m":    round(self.bottom_m, 2),
            "thickness_m": round(self.thickness_m, 2),
            "color":       self.color,
            "soil_type":   self.soil_type,
        }


@dataclass
class Line:
    """路線 (順序付き駅リスト)"""
    line_id:  str
    name:     str
    city_id:  str
    color:    str
    stations: List[Station] = field(default_factory=list)
    operator: Optional[str] = None

    @property
    def station_count(self) -> int:
        return len(self.stations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id":       self.line_id,
            "name":          self.name,
            "city_id":       self.city_id,
            "color":         self.color,
            "operator":      self.operator,
            "station_count": self.station_count,
            "stations":      [s.to_dict() for s in self.stations],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Line":
        return cls(
            line_id=d["line_id"],
            name=d["name"],
            city_id=d["city_id"],
            color=d["color"],
            operator=d.get("operator"),
            stations=[Station.from_dict(s) for s in d.get("stations", [])],
        )


@dataclass
class City:
    """都市 (路線リスト + 地質プロファイル)"""
    city_id:  str
    name:     str
    name_en:  str
    center:   GeoPoint
    lines:    List[Line] = field(default_factory=list)
    geology:  List[GeologyLayer] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def get_line(self, line_id: str) -> Optional[Line]:
        for ln in self.lines:
            if ln.line_id == line_id:
                return ln
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city_id":    self.city_id,
            "name":       self.name,
            "name_en":    self.name_en,
            "center":     self.center.to_dict(),
            "line_count": self.line_count,
            "lines":      [ln.to_dict() for ln in self.lines],
            "geology":    [g.to_dict() for g in self.geology],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "City":
        return cls(
            city_id=d["city_id"],
            name=d["name"],
            name_en=d["name_en"],
            center=GeoPoint.from_dict(d["center"]),
            lines=[Line.from_dict(ln) for ln in d.get("lines", [])],
            geology=[
                GeologyLayer(
                    layer_id=g["layer_id"], name=g["name"], top_m=g["top_m"],
                    bottom_m=g["bottom_m"], color=g["color"], soil_type=g["soil_type"],
                )
                for g in d.get("geology", [])
            ],
        )


@dataclass
class CrossSection:
    """
    路線の地下断面。

    Attributes:
        city_id         : 都市ID
        line_id         : 路線ID
        line_name       : 路線名
        stations        : 駅列 (深度付き)
        layers          : 地質層
        total_distance_m: 起点〜終点の累積水平距離 [m]
        max_depth_m     : 断面の最大深度 [m]
        generated_by    : データ生成元 ("sample" / "gtfs" / "sample(fallback)")
    """
    city_id:          str
    line_id:          str
    line_name:        str
    stations:         List[Station]
    layers:           List[GeologyLayer]
    total_distance_m: float
    max_depth_m:      float
    generated_by:     str = "sample"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city_id":          self.city_id,
            "line_id":          self.line_id,
            "line_name":        self.line_name,
            "station_count":    len(self.stations),
            "stations":         [s.to_dict() for s in self.stations],
            "layers":           [g.to_dict() for g in self.layers],
            "total_distance_m": round(self.total_distance_m, 2),
            "max_depth_m":      round(self.max_depth_m, 2),
            "generated_by":     self.generated_by,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 地理ユーティリティ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(a: GeoPoint, b: GeoPoint) -> float:
    """2点間の大円距離 [m] (Haversine)。"""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# サンプルデータ (13都市・外部依存ゼロ)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 対象 = 政令指定都市 + 首都圏主要市 (人口100万人超を中心に13都市)。
# 各都市の代表1路線 (起点となる主要地下鉄/メトロ) と数駅を最小サンプルとして同梱。
# 駅の緯度経度は概略値。深度・地質層は GeologyModel が後段で付与する。

def _stations(line_id: str, items: List[Any]) -> List[Station]:
    """[(id, name, lat, lon), ...] → List[Station] (order を自動採番)。"""
    out: List[Station] = []
    for i, (sid, name, lat, lon) in enumerate(items):
        out.append(Station(
            station_id=sid, name=name, geo=GeoPoint(lat, lon),
            line_id=line_id, order=i,
        ))
    return out


# 都市メタ: city_id -> (name, name_en, center_lat, center_lon)
_CITY_META: Dict[str, Any] = {
    "tokyo":    ("東京",     "Tokyo",    35.6812, 139.7671),
    "yokohama": ("横浜",     "Yokohama", 35.4660, 139.6222),
    "osaka":    ("大阪",     "Osaka",    34.6937, 135.5023),
    "nagoya":   ("名古屋",   "Nagoya",   35.1709, 136.8815),
    "sapporo":  ("札幌",     "Sapporo",  43.0686, 141.3508),
    "fukuoka":  ("福岡",     "Fukuoka",  33.5902, 130.4207),
    "kobe":     ("神戸",     "Kobe",     34.6901, 135.1955),
    "kawasaki": ("川崎",     "Kawasaki", 35.5308, 139.7029),
    "kyoto":    ("京都",     "Kyoto",    35.0116, 135.7681),
    "saitama":  ("さいたま", "Saitama",  35.8617, 139.6455),
    "hiroshima":("広島",     "Hiroshima",34.3853, 132.4553),
    "sendai":   ("仙台",     "Sendai",   38.2602, 140.8821),
    "chiba":    ("千葉",     "Chiba",    35.6073, 140.1063),
}

# 都市 -> 代表路線リスト [(line_id, name, color, operator, [(sid,name,lat,lon),...]), ...]
_CITY_LINES: Dict[str, List[Any]] = {
    "tokyo": [
        ("marunouchi", "丸ノ内線", "#F62E36", "東京メトロ", [
            ("M01", "荻窪",       35.7045, 139.6203),
            ("M07", "新宿",       35.6909, 139.7003),
            ("M12", "赤坂見附",   35.6772, 139.7370),
            ("M15", "東京",       35.6812, 139.7671),
            ("M18", "御茶ノ水",   35.6993, 139.7654),
            ("M25", "池袋",       35.7295, 139.7109),
        ]),
        ("ginza", "銀座線", "#FF9500", "東京メトロ", [
            ("G01", "渋谷",       35.6580, 139.7016),
            ("G09", "銀座",       35.6717, 139.7640),
            ("G14", "上野",       35.7141, 139.7774),
            ("G19", "浅草",       35.7100, 139.7967),
        ]),
    ],
    "yokohama": [
        ("blue", "ブルーライン", "#0072BC", "横浜市営地下鉄", [
            ("B01", "あざみ野",   35.5747, 139.5547),
            ("B10", "新横浜",     35.5072, 139.6172),
            ("B18", "横浜",       35.4660, 139.6222),
            ("B25", "上大岡",     35.4072, 139.5972),
            ("B32", "湘南台",     35.3960, 139.4660),
        ]),
    ],
    "osaka": [
        ("midosuji", "御堂筋線", "#E5171F", "Osaka Metro", [
            ("M11", "新大阪",     34.7335, 135.5003),
            ("M16", "梅田",       34.7055, 135.4983),
            ("M18", "本町",       34.6829, 135.4994),
            ("M20", "なんば",     34.6659, 135.5012),
            ("M23", "天王寺",     34.6463, 135.5142),
        ]),
    ],
    "nagoya": [
        ("higashiyama", "東山線", "#EDAA00", "名古屋市営地下鉄", [
            ("H08", "名古屋",     35.1709, 136.8815),
            ("H10", "伏見",       35.1690, 136.8970),
            ("H11", "栄",         35.1709, 136.9081),
            ("H14", "今池",       35.1721, 136.9320),
        ]),
    ],
    "sapporo": [
        ("namboku", "南北線", "#00913A", "札幌市営地下鉄", [
            ("N06", "麻生",       43.1165, 141.3408),
            ("N08", "北24条",     43.0966, 141.3406),
            ("N06b", "さっぽろ",   43.0686, 141.3508),
            ("N12", "大通",       43.0606, 141.3469),
            ("N16", "真駒内",     43.0029, 141.3398),
        ]),
    ],
    "fukuoka": [
        ("kuko", "空港線", "#E5006D", "福岡市地下鉄", [
            ("K11", "姪浜",       33.5847, 130.3320),
            ("K06", "天神",       33.5902, 130.3990),
            ("K08", "博多",       33.5902, 130.4207),
            ("K13", "福岡空港",   33.5953, 130.4503),
        ]),
    ],
    "kobe": [
        ("seishin", "西神・山手線", "#1A805F", "神戸市営地下鉄", [
            ("S01", "新神戸",     34.7062, 135.2080),
            ("S02", "三宮",       34.6938, 135.1955),
            ("S07", "名谷",       34.6660, 135.1010),
            ("S09", "西神中央",   34.6790, 135.0470),
        ]),
    ],
    "kawasaki": [
        ("jr-nambu", "南武線", "#FFD400", "JR東日本", [
            ("KW1", "川崎",       35.5308, 139.7029),
            ("KW2", "武蔵小杉",   35.5762, 139.6595),
            ("KW3", "登戸",       35.6203, 139.5700),
            ("KW4", "立川",       35.6979, 139.4138),
        ]),
    ],
    "kyoto": [
        ("karasuma", "烏丸線", "#00A040", "京都市営地下鉄", [
            ("K04", "国際会館",   35.0610, 135.7850),
            ("K08", "今出川",     35.0300, 135.7590),
            ("K11", "京都",       35.0116, 135.7681),
            ("K15", "竹田",       34.9560, 135.7480),
        ]),
    ],
    "saitama": [
        ("jr-saikyo", "埼京線", "#00B261", "JR東日本", [
            ("ST1", "大宮",       35.9065, 139.6238),
            ("ST2", "武蔵浦和",   35.8540, 139.6450),
            ("ST3", "赤羽",       35.7780, 139.7210),
            ("ST4", "池袋",       35.7295, 139.7109),
        ]),
    ],
    "hiroshima": [
        ("astram", "アストラムライン", "#0099D9", "広島高速交通", [
            ("HR1", "本通",       34.3920, 132.4570),
            ("HR2", "県庁前",     34.3960, 132.4590),
            ("HR3", "白島",       34.4060, 132.4640),
            ("HR4", "広域公園前", 34.4490, 132.4060),
        ]),
    ],
    "sendai": [
        ("namboku", "南北線", "#009944", "仙台市地下鉄", [
            ("SN1", "泉中央",     38.3270, 140.8810),
            ("SN2", "北仙台",     38.2820, 140.8730),
            ("SN3", "仙台",       38.2602, 140.8821),
            ("SN4", "長町",       38.2270, 140.8830),
            ("SN5", "富沢",       38.2050, 140.8740),
        ]),
    ],
    "chiba": [
        ("monorail", "千葉都市モノレール1号線", "#E60012", "千葉都市モノレール", [
            ("CH1", "千葉",       35.6073, 140.1063),
            ("CH2", "栄町",       35.6120, 140.1140),
            ("CH3", "千葉公園",   35.6190, 140.1190),
            ("CH4", "県庁前",     35.6010, 140.1190),
        ]),
    ],
}


def _build_sample_cities() -> Dict[str, City]:
    """_CITY_META / _CITY_LINES からサンプル都市群を構築する。"""
    cities: Dict[str, City] = {}
    for cid, (name, name_en, lat, lon) in _CITY_META.items():
        lines: List[Line] = []
        for (line_id, lname, color, operator, st_items) in _CITY_LINES.get(cid, []):
            lines.append(Line(
                line_id=line_id, name=lname, city_id=cid, color=color,
                operator=operator, stations=_stations(line_id, st_items),
            ))
        cities[cid] = City(
            city_id=cid, name=name, name_en=name_en,
            center=GeoPoint(lat, lon), lines=lines, geology=[],
        )
    return cities


_SAMPLE_CITIES: Dict[str, City] = _build_sample_cities()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データソース (抽象基底 + Sample + GTFS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseCityDataSource(ABC):
    """都市データ源の抽象基底クラス。"""

    #: データ生成元の識別子 ("sample" / "gtfs" / "sample(fallback)")
    source_kind: str = "base"

    @abstractmethod
    def load_cities(self) -> List[City]:
        """全都市を返す。"""
        ...

    @abstractmethod
    def load_lines(self, city_id: str) -> List[Line]:
        """指定都市の路線一覧を返す。未知の都市は空リスト。"""
        ...

    def load_city(self, city_id: str) -> Optional[City]:
        for c in self.load_cities():
            if c.city_id == city_id:
                return c
        return None


class SampleCityDataSource(BaseCityDataSource):
    """
    同梱サンプルデータ源 (13都市)。
    外部依存ゼロ。テスト・オフラインのデフォルト。
    """
    source_kind = "sample"

    def load_cities(self) -> List[City]:
        # グローバル定数を変異させないようディープコピーを返す
        return [copy.deepcopy(c) for c in _SAMPLE_CITIES.values()]

    def load_lines(self, city_id: str) -> List[Line]:
        city = _SAMPLE_CITIES.get(city_id)
        return [copy.deepcopy(ln) for ln in city.lines] if city else []


class GTFSCityDataSource(BaseCityDataSource):
    """
    GTFS (公共交通オープンデータ) zip から路線・駅を取り込むデータ源。

    必須列のみに依存する:
      stops.txt  : stop_id, stop_name, stop_lat, stop_lon
      routes.txt : route_id, route_short_name / route_long_name, route_color?
    取得・パースに失敗した場合は SampleCityDataSource に自動フォールバックする
    (source_kind = "sample(fallback)")。

    注意: GTFS には地下深度・地質情報は含まれない。深度は GeologyModel が推定する。
    """
    source_kind = "gtfs"

    def __init__(
        self,
        city_id: str,
        gtfs_url: Optional[str] = None,
        *,
        timeout: float = 15.0,
        _zip_bytes: Optional[bytes] = None,
    ) -> None:
        self.city_id = city_id
        self.gtfs_url = gtfs_url
        self.timeout = timeout
        self._zip_bytes = _zip_bytes       # テスト用に zip を直接注入可能
        self._fallback = SampleCityDataSource()
        self._loaded: Optional[List[Line]] = None

    # -- 取得 ------------------------------------------------------------
    def _fetch_bytes(self) -> Optional[bytes]:
        if self._zip_bytes is not None:
            return self._zip_bytes
        if not self.gtfs_url:
            return None
        try:
            with urllib.request.urlopen(self.gtfs_url, timeout=self.timeout) as resp:
                return resp.read()
        except Exception:
            return None

    @staticmethod
    def _parse_gtfs(zip_bytes: bytes, city_id: str) -> List[Line]:
        """GTFS zip バイト列を Line 群にパースする。失敗時は例外。"""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            if "stops.txt" not in names or "routes.txt" not in names:
                raise ValueError("GTFS zip に stops.txt / routes.txt がありません")

            def _read(fname: str) -> List[Dict[str, str]]:
                with zf.open(fname) as fh:
                    text = io.TextIOWrapper(fh, encoding="utf-8-sig")
                    return list(csv.DictReader(text))

            stops_rows = _read("stops.txt")
            routes_rows = _read("routes.txt")

        # ルート構築
        lines: List[Line] = []
        for r in routes_rows:
            rid = r.get("route_id") or ""
            if not rid:
                continue
            rname = r.get("route_short_name") or r.get("route_long_name") or rid
            color = r.get("route_color") or ""
            color = ("#" + color) if color and not color.startswith("#") else (color or "#888888")
            lines.append(Line(
                line_id=rid, name=rname, city_id=city_id,
                color=color, operator=r.get("agency_id") or None, stations=[],
            ))

        # 駅 (stops) を全路線に共通の順序付きで割り当てる
        # (GTFS の stop_times を辿らず、簡易に stops を取り込む。順序は出現順)
        stations: List[Station] = []
        for i, s in enumerate(stops_rows):
            sid = s.get("stop_id") or ""
            try:
                lat = float(s.get("stop_lat") or "")
                lon = float(s.get("stop_lon") or "")
            except ValueError:
                continue
            if not sid:
                continue
            stations.append(Station(
                station_id=sid, name=s.get("stop_name") or sid,
                geo=GeoPoint(lat, lon),
                line_id=lines[0].line_id if lines else city_id, order=i,
            ))

        if not lines or not stations:
            raise ValueError("GTFS から有効な路線/駅を取得できませんでした")

        # 駅を先頭路線にひも付け (簡易モデル)
        lines[0].stations = stations
        return lines

    # -- API -------------------------------------------------------------
    def load_lines(self, city_id: str) -> List[Line]:
        if city_id != self.city_id:
            return self._fallback.load_lines(city_id)
        if self._loaded is not None:
            return self._loaded
        zip_bytes = self._fetch_bytes()
        if zip_bytes is not None:
            try:
                self._loaded = self._parse_gtfs(zip_bytes, city_id)
                self.source_kind = "gtfs"
                return self._loaded
            except Exception:
                pass  # フォールバックへ
        # フォールバック
        self.source_kind = "sample(fallback)"
        self._loaded = self._fallback.load_lines(city_id)
        return self._loaded

    def load_cities(self) -> List[City]:
        # 都市メタはサンプルを流用し、対象都市のみ路線を差し替える
        cities = self._fallback.load_cities()
        for c in cities:
            if c.city_id == self.city_id:
                c.lines = self.load_lines(self.city_id)
        return cities


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GeologyModel — 地質層・地下深度のルールベース推定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# GTFS には地下深度・地質情報が含まれないため、都市ごとの地質プロファイルと
# 路線の駅深度をルールベースで推定する。絶対値の精度は保証しないが、
# 「層が深度順」「深度が正値」といった整合性は満たす。

# 都市 -> 地質層プロファイル [(name, thickness_m, color, soil_type), ...]
_GEOLOGY_PROFILES: Dict[str, List[Any]] = {
    # 東京低地 (沖積層が厚い)
    "tokyo": [
        ("沖積層",       12.0, "#D7C49E", "clay"),
        ("武蔵野礫層",   8.0,  "#C9A66B", "gravel"),
        ("東京層",       15.0, "#B08D57", "sand"),
        ("上総層群",     25.0, "#8C7853", "bedrock"),
    ],
    # 関東ローム + 沖積
    "_kanto": [
        ("関東ローム層", 6.0,  "#A0522D", "loam"),
        ("沖積層",       10.0, "#D7C49E", "clay"),
        ("洪積砂礫層",   14.0, "#C9A66B", "gravel"),
        ("基盤岩",       25.0, "#8C7853", "bedrock"),
    ],
    # 大阪平野
    "osaka": [
        ("沖積層",       10.0, "#D7C49E", "clay"),
        ("天満層",       12.0, "#C9A66B", "sand"),
        ("大阪層群",     30.0, "#8C7853", "bedrock"),
    ],
    # 名古屋・濃尾平野
    "nagoya": [
        ("沖積層",       9.0,  "#D7C49E", "clay"),
        ("熱田層",       11.0, "#C9A66B", "sand"),
        ("東海層群",     28.0, "#8C7853", "bedrock"),
    ],
    # 火山灰系 (札幌・仙台)
    "_volcanic": [
        ("火山灰質土",   7.0,  "#9E8B6B", "loam"),
        ("沖積砂礫層",   12.0, "#C9A66B", "gravel"),
        ("第三系基盤",   26.0, "#8C7853", "bedrock"),
    ],
    # デフォルト
    "_default": [
        ("表土層",       6.0,  "#A0826D", "loam"),
        ("沖積層",       12.0, "#D7C49E", "clay"),
        ("洪積層",       16.0, "#C9A66B", "gravel"),
        ("基盤岩",       24.0, "#8C7853", "bedrock"),
    ],
}

# プロファイルキーへのマッピング
_GEOLOGY_KEY: Dict[str, str] = {
    "tokyo": "tokyo",
    "osaka": "osaka",
    "nagoya": "nagoya",
    "yokohama": "_kanto",
    "kawasaki": "_kanto",
    "saitama": "_kanto",
    "chiba": "_kanto",
    "sapporo": "_volcanic",
    "sendai": "_volcanic",
}


class GeologyModel:
    """地質層・地下深度のルールベース推定モデル。"""

    def estimate_layers(self, city_id: str, center: Optional[GeoPoint] = None) -> List[GeologyLayer]:
        """都市の地質層プロファイルを返す (上端=0 から積層)。"""
        key = _GEOLOGY_KEY.get(city_id, "_default")
        profile = _GEOLOGY_PROFILES[key]
        layers: List[GeologyLayer] = []
        top = 0.0
        for i, (name, thickness, color, soil) in enumerate(profile):
            bottom = top + thickness
            layers.append(GeologyLayer(
                layer_id=city_id + "-L" + str(i),
                name=name, top_m=top, bottom_m=bottom,
                color=color, soil_type=soil,
            ))
            top = bottom
        return layers

    def estimate_depth(self, station: Station, line: Line) -> float:
        """
        駅の地下深度 [m] を推定する。
        路線種別 (地下鉄=深い / モノレール=地上) と駅順から起伏を付ける。
        """
        name = (line.name or "") + (line.line_id or "")
        is_underground = not any(k in name for k in ("モノレール", "monorail", "astram", "アストラム"))
        base = 18.0 if is_underground else -8.0  # 地上路線は負 (高架)
        # 駅順に応じて緩やかな起伏 (正弦波)
        n = max(1, line.station_count)
        wave = math.sin((station.order / n) * math.pi) * 6.0
        depth = base + wave
        return round(depth, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CrossSectionBuilder — データ源 + 地質モデル → CrossSection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CrossSectionBuilder:
    """データ源と地質モデルを統合して CrossSection を構築する。"""

    def __init__(self, geology_model: Optional[GeologyModel] = None) -> None:
        self.geology_model = geology_model or GeologyModel()

    def build(
        self,
        city_id: str,
        line_id: str,
        source: BaseCityDataSource,
    ) -> CrossSection:
        """指定都市・路線の地下断面を構築する。"""
        lines = source.load_lines(city_id)
        line = next((ln for ln in lines if ln.line_id == line_id), None)
        if line is None:
            raise ValueError(
                "未知の都市/路線です: city_id=" + city_id + ", line_id=" + line_id
            )

        city = source.load_city(city_id)
        center = city.center if city else None
        layers = self.geology_model.estimate_layers(city_id, center)

        # 駅深度を推定し、累積距離を計算
        stations: List[Station] = []
        total_dist = 0.0
        prev: Optional[Station] = None
        for st in line.stations:
            depth = self.geology_model.estimate_depth(st, line)
            new_st = Station(
                station_id=st.station_id, name=st.name, geo=st.geo,
                line_id=st.line_id, order=st.order, depth_m=depth,
                name_kana=st.name_kana,
            )
            if prev is not None:
                total_dist += haversine_m(prev.geo, new_st.geo)
            stations.append(new_st)
            prev = new_st

        max_depth = max(
            [s.depth_m for s in stations if s.depth_m is not None] +
            [layers[-1].bottom_m if layers else 0.0]
        )
        return CrossSection(
            city_id=city_id, line_id=line_id, line_name=line.name,
            stations=stations, layers=layers,
            total_distance_m=total_dist, max_depth_m=max_depth,
            generated_by=source.source_kind,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CityMapStore — CrossSection インメモリストア
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CityMapStore:
    """構築済み CrossSection のインメモリストア。"""

    def __init__(self) -> None:
        self._store: Dict[str, CrossSection] = {}

    @staticmethod
    def _key(city_id: str, line_id: str) -> str:
        return city_id + ":" + line_id

    def save(self, cs: CrossSection) -> None:
        self._store[self._key(cs.city_id, cs.line_id)] = cs

    def get(self, city_id: str, line_id: str) -> Optional[CrossSection]:
        return self._store.get(self._key(city_id, line_id))

    def list_by_city(self, city_id: str) -> List[CrossSection]:
        return [cs for cs in self._store.values() if cs.city_id == city_id]

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CityMapFactory — データ源ファクトリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CityMapFactory:
    """都市データ源のファクトリ。"""

    @staticmethod
    def from_sample() -> SampleCityDataSource:
        """同梱サンプル (13都市) を返す。デフォルト。"""
        return SampleCityDataSource()

    @staticmethod
    def from_gtfs(
        city_id: str,
        gtfs_url: Optional[str] = None,
        *,
        _zip_bytes: Optional[bytes] = None,
    ) -> GTFSCityDataSource:
        """GTFS データ源を返す (取得失敗時はサンプルへフォールバック)。"""
        return GTFSCityDataSource(city_id, gtfs_url, _zip_bytes=_zip_bytes)

    @staticmethod
    def available_cities() -> List[Dict[str, Any]]:
        """対応都市メタの一覧 (city_id / name / name_en / line_count)。"""
        out: List[Dict[str, Any]] = []
        for c in _SAMPLE_CITIES.values():
            out.append({
                "city_id":    c.city_id,
                "name":       c.name,
                "name_en":    c.name_en,
                "center":     c.center.to_dict(),
                "line_count": c.line_count,
            })
        return out
