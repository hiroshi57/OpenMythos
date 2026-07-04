"""
Sprint 71A — 主要都市メトロ断面図データ

対象都市: 東京・大阪・名古屋・横浜・福岡
データ: 地下鉄路線 GeoJSON + 地質層データ
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────


class CityName(str, Enum):
    TOKYO = "tokyo"
    OSAKA = "osaka"
    NAGOYA = "nagoya"
    YOKOHAMA = "yokohama"
    FUKUOKA = "fukuoka"


class LineType(str, Enum):
    SUBWAY = "subway"
    JR = "jr"
    PRIVATE = "private"


class GeologyLayerType(str, Enum):
    FILL = "fill"           # 盛土・埋立
    ALLUVIUM = "alluvium"   # 沖積層
    DILUVIUM = "diluvium"   # 洪積層
    CLAY = "clay"           # 粘土層
    SAND = "sand"           # 砂層
    GRAVEL = "gravel"       # 砂礫層
    BEDROCK = "bedrock"     # 岩盤


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class GeoCoord:
    """緯度経度座標"""
    lat: float
    lon: float

    def to_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon}


@dataclass
class Station:
    """地下鉄駅データ"""
    id: str
    name: str
    name_en: str
    line_id: str
    city: CityName
    coord: GeoCoord
    depth_m: float          # 地下深度 (m)
    platform_count: int = 2
    opened_year: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "name_en": self.name_en,
            "line_id": self.line_id,
            "city": self.city.value,
            "coord": self.coord.to_dict(),
            "depth_m": self.depth_m,
            "platform_count": self.platform_count,
            "opened_year": self.opened_year,
        }


@dataclass
class MetroLine:
    """地下鉄路線データ"""
    id: str
    name: str
    name_en: str
    city: CityName
    line_type: LineType
    color: str              # HEX カラーコード
    station_ids: List[str] = field(default_factory=list)
    total_length_km: float = 0.0
    opened_year: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "name_en": self.name_en,
            "city": self.city.value,
            "line_type": self.line_type.value,
            "color": self.color,
            "station_ids": self.station_ids,
            "total_length_km": self.total_length_km,
            "opened_year": self.opened_year,
        }


@dataclass
class GeologyLayer:
    """地質層データ"""
    id: str
    city: CityName
    layer_type: GeologyLayerType
    name: str
    depth_from_m: float     # 層上端の深度 (m)
    depth_to_m: float       # 層下端の深度 (m)
    color: str              # SVG 表示色
    n_value: Optional[float] = None  # 標準貫入試験 N 値

    @property
    def thickness_m(self) -> float:
        return self.depth_to_m - self.depth_from_m

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "city": self.city.value,
            "layer_type": self.layer_type.value,
            "name": self.name,
            "depth_from_m": self.depth_from_m,
            "depth_to_m": self.depth_to_m,
            "thickness_m": self.thickness_m,
            "color": self.color,
            "n_value": self.n_value,
        }


@dataclass
class CityMapData:
    """都市の地図データセット"""
    city: CityName
    lines: List[MetroLine]
    stations: List[Station]
    geology_layers: List[GeologyLayer]

    def to_dict(self) -> dict:
        return {
            "city": self.city.value,
            "lines": [ln.to_dict() for ln in self.lines],
            "stations": [st.to_dict() for st in self.stations],
            "geology_layers": [gl.to_dict() for gl in self.geology_layers],
        }

    def to_geojson(self) -> dict:
        """GeoJSON FeatureCollection 形式で返す"""
        features = []
        for st in self.stations:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [st.coord.lon, st.coord.lat],
                },
                "properties": {
                    "id": st.id,
                    "name": st.name,
                    "name_en": st.name_en,
                    "line_id": st.line_id,
                    "depth_m": st.depth_m,
                    "platform_count": st.platform_count,
                },
            })
        return {
            "type": "FeatureCollection",
            "properties": {"city": self.city.value},
            "features": features,
        }


# ─── Stores ────────────────────────────────────────────────────────


class StationStore:
    """駅データストア"""

    def __init__(self) -> None:
        self._data: Dict[str, Station] = {}

    def add(self, station: Station) -> None:
        self._data[station.id] = station

    def get(self, station_id: str) -> Optional[Station]:
        return self._data.get(station_id)

    def list_by_city(self, city: CityName) -> List[Station]:
        return [s for s in self._data.values() if s.city == city]

    def list_by_line(self, line_id: str) -> List[Station]:
        return [s for s in self._data.values() if s.line_id == line_id]

    def all(self) -> List[Station]:
        return list(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


class MetroLineStore:
    """路線データストア"""

    def __init__(self) -> None:
        self._data: Dict[str, MetroLine] = {}

    def add(self, line: MetroLine) -> None:
        self._data[line.id] = line

    def get(self, line_id: str) -> Optional[MetroLine]:
        return self._data.get(line_id)

    def list_by_city(self, city: CityName) -> List[MetroLine]:
        return [ln for ln in self._data.values() if ln.city == city]

    def all(self) -> List[MetroLine]:
        return list(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


class GeologyStore:
    """地質層データストア"""

    def __init__(self) -> None:
        self._data: Dict[str, GeologyLayer] = {}

    def add(self, layer: GeologyLayer) -> None:
        self._data[layer.id] = layer

    def get(self, layer_id: str) -> Optional[GeologyLayer]:
        return self._data.get(layer_id)

    def list_by_city(self, city: CityName) -> List[GeologyLayer]:
        layers = [gl for gl in self._data.values() if gl.city == city]
        return sorted(layers, key=lambda x: x.depth_from_m)

    def all(self) -> List[GeologyLayer]:
        return list(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


class CityMapStore:
    """都市地図データの統合ストア"""

    def __init__(self) -> None:
        self.stations = StationStore()
        self.lines = MetroLineStore()
        self.geology = GeologyStore()

    def get_city_data(self, city: CityName) -> CityMapData:
        return CityMapData(
            city=city,
            lines=self.lines.list_by_city(city),
            stations=self.stations.list_by_city(city),
            geology_layers=self.geology.list_by_city(city),
        )

    def cities(self) -> List[str]:
        """データが存在する都市一覧"""
        city_set = set()
        for st in self.stations.all():
            city_set.add(st.city.value)
        return sorted(city_set)


# ─── Static Dataset ────────────────────────────────────────────────


class CityMapDataset:
    """主要都市の地下鉄・地質データセット (静的プリセット)"""

    @classmethod
    def build(cls) -> CityMapStore:
        store = CityMapStore()
        cls._load_tokyo(store)
        cls._load_osaka(store)
        cls._load_nagoya(store)
        cls._load_yokohama(store)
        cls._load_fukuoka(store)
        return store

    # ── 東京 ──────────────────────────────────────────────────────

    @classmethod
    def _load_tokyo(cls, store: CityMapStore) -> None:
        # 丸ノ内線
        m_line = MetroLine(
            id="tokyo-marunouchi", name="丸ノ内線", name_en="Marunouchi Line",
            city=CityName.TOKYO, line_type=LineType.SUBWAY,
            color="#E60012", station_ids=[
                "tokyo-ogikubo", "tokyo-nakano", "tokyo-koenji",
                "tokyo-shinjuku", "tokyo-yotsuya", "tokyo-akasaka-mitsuke",
                "tokyo-kasumigaseki", "tokyo-ginza", "tokyo-tokyo",
                "tokyo-otemachi",
            ],
            total_length_km=27.4, opened_year=1954,
        )
        store.lines.add(m_line)

        stations_tokyo = [
            Station("tokyo-ogikubo", "荻窪", "Ogikubo", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.7060, 139.6241), 12.0, 2, 1962),
            Station("tokyo-nakano", "中野", "Nakano", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.7078, 139.6654), 14.5, 2, 1962),
            Station("tokyo-koenji", "高円寺", "Koenji", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.7054, 139.6494), 13.2, 2, 1962),
            Station("tokyo-shinjuku", "新宿", "Shinjuku", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6896, 139.6995), 18.0, 4, 1959),
            Station("tokyo-yotsuya", "四ツ谷", "Yotsuya", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6862, 139.7298), 16.5, 2, 1959),
            Station("tokyo-akasaka-mitsuke", "赤坂見附", "Akasaka-mitsuke", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6791, 139.7377), 19.3, 2, 1959),
            Station("tokyo-kasumigaseki", "霞ケ関", "Kasumigaseki", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6740, 139.7503), 22.0, 2, 1957),
            Station("tokyo-ginza", "銀座", "Ginza", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6714, 139.7652), 24.5, 2, 1957),
            Station("tokyo-tokyo", "東京", "Tokyo", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6812, 139.7671), 20.1, 2, 1956),
            Station("tokyo-otemachi", "大手町", "Otemachi", "tokyo-marunouchi",
                    CityName.TOKYO, GeoCoord(35.6864, 139.7631), 21.8, 2, 1956),
        ]
        for st in stations_tokyo:
            store.stations.add(st)

        # 東京の地質層
        geology_tokyo = [
            GeologyLayer("tky-gl-1", CityName.TOKYO, GeologyLayerType.FILL,
                         "盛土・造成土", 0.0, 3.0, "#D2B48C", n_value=2.0),
            GeologyLayer("tky-gl-2", CityName.TOKYO, GeologyLayerType.ALLUVIUM,
                         "沖積粘土層 (軟弱)", 3.0, 12.0, "#90EE90", n_value=3.0),
            GeologyLayer("tky-gl-3", CityName.TOKYO, GeologyLayerType.SAND,
                         "沖積砂層", 12.0, 20.0, "#F4D03F", n_value=15.0),
            GeologyLayer("tky-gl-4", CityName.TOKYO, GeologyLayerType.DILUVIUM,
                         "洪積粘土層 (東京層)", 20.0, 35.0, "#85C1E9", n_value=20.0),
            GeologyLayer("tky-gl-5", CityName.TOKYO, GeologyLayerType.GRAVEL,
                         "洪積砂礫層 (東京礫層)", 35.0, 55.0, "#F0A500", n_value=50.0),
            GeologyLayer("tky-gl-6", CityName.TOKYO, GeologyLayerType.BEDROCK,
                         "凝灰岩・砂岩 (上総層群)", 55.0, 100.0, "#AAB7B8", n_value=None),
        ]
        for gl in geology_tokyo:
            store.geology.add(gl)

    # ── 大阪 ──────────────────────────────────────────────────────

    @classmethod
    def _load_osaka(cls, store: CityMapStore) -> None:
        o_line = MetroLine(
            id="osaka-midosuji", name="御堂筋線", name_en="Midosuji Line",
            city=CityName.OSAKA, line_type=LineType.SUBWAY,
            color="#E8380D", station_ids=[
                "osaka-senri-chuo", "osaka-shinsaibashi", "osaka-namba",
                "osaka-tengachaya", "osaka-nakamozu",
            ],
            total_length_km=24.5, opened_year=1933,
        )
        store.lines.add(o_line)

        stations_osaka = [
            Station("osaka-senri-chuo", "千里中央", "Senri-Chuo", "osaka-midosuji",
                    CityName.OSAKA, GeoCoord(34.8116, 135.4967), 8.0, 2, 1970),
            Station("osaka-shinsaibashi", "心斎橋", "Shinsaibashi", "osaka-midosuji",
                    CityName.OSAKA, GeoCoord(34.6722, 135.4998), 19.5, 2, 1938),
            Station("osaka-namba", "なんば", "Namba", "osaka-midosuji",
                    CityName.OSAKA, GeoCoord(34.6654, 135.5010), 16.8, 4, 1938),
            Station("osaka-tengachaya", "天下茶屋", "Tengachaya", "osaka-midosuji",
                    CityName.OSAKA, GeoCoord(34.6391, 135.5071), 14.2, 2, 1987),
            Station("osaka-nakamozu", "中百舌鳥", "Nakamozu", "osaka-midosuji",
                    CityName.OSAKA, GeoCoord(34.5585, 135.4897), 9.5, 2, 1987),
        ]
        for st in stations_osaka:
            store.stations.add(st)

        geology_osaka = [
            GeologyLayer("osk-gl-1", CityName.OSAKA, GeologyLayerType.FILL,
                         "盛土・埋立土", 0.0, 4.0, "#D2B48C", n_value=1.5),
            GeologyLayer("osk-gl-2", CityName.OSAKA, GeologyLayerType.ALLUVIUM,
                         "沖積粘土層 (軟弱・大阪低地)", 4.0, 18.0, "#90EE90", n_value=2.5),
            GeologyLayer("osk-gl-3", CityName.OSAKA, GeologyLayerType.SAND,
                         "沖積砂層 (難波砂層)", 18.0, 28.0, "#F4D03F", n_value=18.0),
            GeologyLayer("osk-gl-4", CityName.OSAKA, GeologyLayerType.CLAY,
                         "大阪粘土層群 (Ma12〜Ma0)", 28.0, 60.0, "#7FB3D3", n_value=8.0),
            GeologyLayer("osk-gl-5", CityName.OSAKA, GeologyLayerType.GRAVEL,
                         "大阪礫層群", 60.0, 80.0, "#F0A500", n_value=60.0),
            GeologyLayer("osk-gl-6", CityName.OSAKA, GeologyLayerType.BEDROCK,
                         "花崗岩・結晶片岩", 80.0, 120.0, "#AAB7B8", n_value=None),
        ]
        for gl in geology_osaka:
            store.geology.add(gl)

    # ── 名古屋 ─────────────────────────────────────────────────────

    @classmethod
    def _load_nagoya(cls, store: CityMapStore) -> None:
        n_line = MetroLine(
            id="nagoya-higashiyama", name="東山線", name_en="Higashiyama Line",
            city=CityName.NAGOYA, line_type=LineType.SUBWAY,
            color="#FFD700", station_ids=[
                "nagoya-takabata", "nagoya-nagoya", "nagoya-sakae",
                "nagoya-fujigaoka",
            ],
            total_length_km=20.6, opened_year=1957,
        )
        store.lines.add(n_line)

        stations_nagoya = [
            Station("nagoya-takabata", "高畑", "Takabata", "nagoya-higashiyama",
                    CityName.NAGOYA, GeoCoord(35.1587, 136.8451), 10.5, 2, 1965),
            Station("nagoya-nagoya", "名古屋", "Nagoya", "nagoya-higashiyama",
                    CityName.NAGOYA, GeoCoord(35.1709, 136.8815), 15.3, 4, 1957),
            Station("nagoya-sakae", "栄", "Sakae", "nagoya-higashiyama",
                    CityName.NAGOYA, GeoCoord(35.1689, 136.9065), 16.0, 2, 1957),
            Station("nagoya-fujigaoka", "藤が丘", "Fujigaoka", "nagoya-higashiyama",
                    CityName.NAGOYA, GeoCoord(35.1716, 137.0513), 7.5, 2, 1969),
        ]
        for st in stations_nagoya:
            store.stations.add(st)

        geology_nagoya = [
            GeologyLayer("ngy-gl-1", CityName.NAGOYA, GeologyLayerType.FILL,
                         "盛土", 0.0, 2.5, "#D2B48C", n_value=3.0),
            GeologyLayer("ngy-gl-2", CityName.NAGOYA, GeologyLayerType.ALLUVIUM,
                         "沖積粘土・砂 (熱田低地)", 2.5, 15.0, "#90EE90", n_value=5.0),
            GeologyLayer("ngy-gl-3", CityName.NAGOYA, GeologyLayerType.DILUVIUM,
                         "洪積砂礫層 (名古屋礫層)", 15.0, 30.0, "#F4D03F", n_value=40.0),
            GeologyLayer("ngy-gl-4", CityName.NAGOYA, GeologyLayerType.CLAY,
                         "熱田粘土層", 30.0, 50.0, "#7FB3D3", n_value=12.0),
            GeologyLayer("ngy-gl-5", CityName.NAGOYA, GeologyLayerType.BEDROCK,
                         "花崗岩 (東海層群基盤)", 50.0, 90.0, "#AAB7B8", n_value=None),
        ]
        for gl in geology_nagoya:
            store.geology.add(gl)

    # ── 横浜 ──────────────────────────────────────────────────────

    @classmethod
    def _load_yokohama(cls, store: CityMapStore) -> None:
        y_line = MetroLine(
            id="yokohama-blue", name="ブルーライン", name_en="Blue Line",
            city=CityName.YOKOHAMA, line_type=LineType.SUBWAY,
            color="#0099D4", station_ids=[
                "yokohama-shonandai", "yokohama-totsuka", "yokohama-kami-ohoka",
                "yokohama-yokohama", "yokohama-azamino",
            ],
            total_length_km=40.4, opened_year=1972,
        )
        store.lines.add(y_line)

        stations_yokohama = [
            Station("yokohama-shonandai", "湘南台", "Shonandai", "yokohama-blue",
                    CityName.YOKOHAMA, GeoCoord(35.3874, 139.4879), 6.5, 2, 1999),
            Station("yokohama-totsuka", "戸塚", "Totsuka", "yokohama-blue",
                    CityName.YOKOHAMA, GeoCoord(35.3999, 139.5337), 11.8, 2, 1976),
            Station("yokohama-kami-ohoka", "上大岡", "Kami-Ohoka", "yokohama-blue",
                    CityName.YOKOHAMA, GeoCoord(35.3960, 139.5897), 13.5, 2, 1976),
            Station("yokohama-yokohama", "横浜", "Yokohama", "yokohama-blue",
                    CityName.YOKOHAMA, GeoCoord(35.4660, 139.6222), 17.2, 4, 1976),
            Station("yokohama-azamino", "あざみ野", "Azamino", "yokohama-blue",
                    CityName.YOKOHAMA, GeoCoord(35.5627, 139.5934), 9.0, 2, 1993),
        ]
        for st in stations_yokohama:
            store.stations.add(st)

        geology_yokohama = [
            GeologyLayer("yok-gl-1", CityName.YOKOHAMA, GeologyLayerType.FILL,
                         "埋立・盛土", 0.0, 3.5, "#D2B48C", n_value=2.0),
            GeologyLayer("yok-gl-2", CityName.YOKOHAMA, GeologyLayerType.ALLUVIUM,
                         "沖積砂泥層 (横浜港埋立)", 3.5, 14.0, "#90EE90", n_value=4.0),
            GeologyLayer("yok-gl-3", CityName.YOKOHAMA, GeologyLayerType.DILUVIUM,
                         "洪積砂礫層 (下末吉面)", 14.0, 28.0, "#F4D03F", n_value=35.0),
            GeologyLayer("yok-gl-4", CityName.YOKOHAMA, GeologyLayerType.CLAY,
                         "下末吉粘土層", 28.0, 42.0, "#7FB3D3", n_value=15.0),
            GeologyLayer("yok-gl-5", CityName.YOKOHAMA, GeologyLayerType.BEDROCK,
                         "泥岩・砂岩 (三浦層群)", 42.0, 80.0, "#AAB7B8", n_value=None),
        ]
        for gl in geology_yokohama:
            store.geology.add(gl)

    # ── 福岡 ──────────────────────────────────────────────────────

    @classmethod
    def _load_fukuoka(cls, store: CityMapStore) -> None:
        f_line = MetroLine(
            id="fukuoka-kuko", name="空港線", name_en="Airport Line",
            city=CityName.FUKUOKA, line_type=LineType.SUBWAY,
            color="#E60012", station_ids=[
                "fukuoka-airport", "fukuoka-hakata", "fukuoka-tenjin",
                "fukuoka-meinohama",
            ],
            total_length_km=13.1, opened_year=1981,
        )
        store.lines.add(f_line)

        stations_fukuoka = [
            Station("fukuoka-airport", "福岡空港", "Fukuoka Airport", "fukuoka-kuko",
                    CityName.FUKUOKA, GeoCoord(33.5897, 130.4508), 5.5, 2, 1993),
            Station("fukuoka-hakata", "博多", "Hakata", "fukuoka-kuko",
                    CityName.FUKUOKA, GeoCoord(33.5890, 130.4208), 12.8, 4, 1981),
            Station("fukuoka-tenjin", "天神", "Tenjin", "fukuoka-kuko",
                    CityName.FUKUOKA, GeoCoord(33.5904, 130.3985), 14.5, 4, 1981),
            Station("fukuoka-meinohama", "姪浜", "Meinohama", "fukuoka-kuko",
                    CityName.FUKUOKA, GeoCoord(33.5931, 130.3294), 8.2, 2, 1981),
        ]
        for st in stations_fukuoka:
            store.stations.add(st)

        geology_fukuoka = [
            GeologyLayer("fuk-gl-1", CityName.FUKUOKA, GeologyLayerType.FILL,
                         "盛土・埋立", 0.0, 3.0, "#D2B48C", n_value=2.0),
            GeologyLayer("fuk-gl-2", CityName.FUKUOKA, GeologyLayerType.ALLUVIUM,
                         "沖積砂泥層 (博多湾岸)", 3.0, 12.0, "#90EE90", n_value=5.0),
            GeologyLayer("fuk-gl-3", CityName.FUKUOKA, GeologyLayerType.SAND,
                         "砂礫層 (那珂川低地)", 12.0, 22.0, "#F4D03F", n_value=25.0),
            GeologyLayer("fuk-gl-4", CityName.FUKUOKA, GeologyLayerType.CLAY,
                         "海成粘土層", 22.0, 35.0, "#7FB3D3", n_value=10.0),
            GeologyLayer("fuk-gl-5", CityName.FUKUOKA, GeologyLayerType.BEDROCK,
                         "花崗岩 (福岡基盤)", 35.0, 70.0, "#AAB7B8", n_value=None),
        ]
        for gl in geology_fukuoka:
            store.geology.add(gl)
