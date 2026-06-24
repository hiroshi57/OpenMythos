"""
Sprint 72C — 地図レポート生成 (Markdown)

都市の地下鉄プロファイルを Markdown レポートで出力する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from open_mythos.skills.city_map import (
    CityName, CityMapData, CityMapStore,
)


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class ReportSection:
    """Markdown レポートのセクション"""
    title: str
    content: str

    def to_markdown(self) -> str:
        return f"## {self.title}\n\n{self.content}\n"


@dataclass
class CityMapReport:
    """都市地図レポート"""
    city: str
    title: str
    sections: List[ReportSection] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(section.to_markdown())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "title": self.title,
            "sections": [
                {"title": s.title, "content": s.content}
                for s in self.sections
            ],
            "markdown": self.to_markdown(),
        }


@dataclass
class MultiCityReport:
    """複数都市の比較レポート"""
    title: str
    cities: List[str]
    sections: List[ReportSection] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(section.to_markdown())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "cities": self.cities,
            "sections": [
                {"title": s.title, "content": s.content}
                for s in self.sections
            ],
            "markdown": self.to_markdown(),
        }


# ─── MapReportEngine ──────────────────────────────────────────────


class MapReportEngine:
    """CityMapStore からレポートを生成するエンジン"""

    def __init__(self, store: CityMapStore) -> None:
        self._store = store

    # ── 単都市レポート ─────────────────────────────────────────────

    def generate_city_report(self, city: CityName) -> CityMapReport:
        data = self._store.get_city_data(city)
        title = f"{city.value.upper()} 地下鉄プロファイルレポート"
        sections = [
            self._overview_section(data),
            self._lines_section(data),
            self._stations_section(data),
            self._geology_section(data),
            self._depth_stats_section(data),
        ]
        return CityMapReport(city=city.value, title=title, sections=sections)

    def _overview_section(self, data: CityMapData) -> ReportSection:
        city = data.city.value.upper()
        n_lines = len(data.lines)
        n_stations = len(data.stations)
        n_layers = len(data.geology_layers)
        content = (
            f"**都市**: {city}  \n"
            f"**路線数**: {n_lines}  \n"
            f"**駅数**: {n_stations}  \n"
            f"**地質層数**: {n_layers}  \n"
        )
        return ReportSection("概要", content)

    def _lines_section(self, data: CityMapData) -> ReportSection:
        if not data.lines:
            return ReportSection("路線一覧", "_データなし_")
        rows = ["| 路線名 | 種別 | 総延長 (km) | 開業年 |",
                "|--------|------|------------|--------|"]
        for ln in data.lines:
            opened = str(ln.opened_year) if ln.opened_year else "不明"
            rows.append(f"| {ln.name} ({ln.name_en}) | {ln.line_type.value} "
                        f"| {ln.total_length_km} | {opened} |")
        return ReportSection("路線一覧", "\n".join(rows))

    def _stations_section(self, data: CityMapData) -> ReportSection:
        if not data.stations:
            return ReportSection("駅一覧", "_データなし_")
        sorted_st = sorted(data.stations, key=lambda s: s.depth_m, reverse=True)
        rows = ["| 駅名 | 深度 (m) | プラットフォーム数 | 開業年 |",
                "|------|---------|-----------------|--------|"]
        for st in sorted_st:
            opened = str(st.opened_year) if st.opened_year else "不明"
            rows.append(f"| {st.name} ({st.name_en}) | {st.depth_m} "
                        f"| {st.platform_count} | {opened} |")
        return ReportSection("駅一覧 (深度順)", "\n".join(rows))

    def _geology_section(self, data: CityMapData) -> ReportSection:
        if not data.geology_layers:
            return ReportSection("地質層", "_データなし_")
        rows = ["| 層名 | 種別 | 深度 (m) | 厚さ (m) | N値 |",
                "|------|------|---------|---------|-----|"]
        for gl in data.geology_layers:
            n_val = f"{gl.n_value}" if gl.n_value is not None else "—"
            rows.append(
                f"| {gl.name} | {gl.layer_type.value} "
                f"| {gl.depth_from_m}〜{gl.depth_to_m} "
                f"| {gl.thickness_m:.1f} | {n_val} |"
            )
        return ReportSection("地質層プロファイル", "\n".join(rows))

    def _depth_stats_section(self, data: CityMapData) -> ReportSection:
        if not data.stations:
            return ReportSection("深度統計", "_駅データなし_")
        depths = [s.depth_m for s in data.stations]
        avg = sum(depths) / len(depths)
        deepest = max(data.stations, key=lambda s: s.depth_m)
        shallowest = min(data.stations, key=lambda s: s.depth_m)
        content = (
            f"- **最深駅**: {deepest.name} ({deepest.depth_m} m)  \n"
            f"- **最浅駅**: {shallowest.name} ({shallowest.depth_m} m)  \n"
            f"- **平均深度**: {avg:.1f} m  \n"
            f"- **駅数**: {len(depths)}  \n"
        )
        return ReportSection("深度統計", content)

    # ── 複数都市比較レポート ────────────────────────────────────────

    def generate_multi_city_report(
        self, cities: List[CityName]
    ) -> MultiCityReport:
        title = "主要都市 地下鉄プロファイル比較レポート"
        sections = [
            self._comparison_overview(cities),
            self._comparison_depth_table(cities),
            self._comparison_geology_table(cities),
        ]
        return MultiCityReport(
            title=title,
            cities=[c.value for c in cities],
            sections=sections,
        )

    def _comparison_overview(
        self, cities: List[CityName]
    ) -> ReportSection:
        rows = ["| 都市 | 路線数 | 駅数 | 地質層数 |",
                "|------|--------|------|---------|"]
        for city in cities:
            data = self._store.get_city_data(city)
            rows.append(
                f"| {city.value.upper()} | {len(data.lines)} "
                f"| {len(data.stations)} | {len(data.geology_layers)} |"
            )
        return ReportSection("都市別サマリー", "\n".join(rows))

    def _comparison_depth_table(
        self, cities: List[CityName]
    ) -> ReportSection:
        rows = ["| 都市 | 最深 (m) | 最浅 (m) | 平均 (m) |",
                "|------|---------|---------|---------|"]
        for city in cities:
            stations = self._store.stations.list_by_city(city)
            if not stations:
                rows.append(f"| {city.value.upper()} | — | — | — |")
                continue
            depths = [s.depth_m for s in stations]
            rows.append(
                f"| {city.value.upper()} "
                f"| {max(depths):.1f} "
                f"| {min(depths):.1f} "
                f"| {sum(depths)/len(depths):.1f} |"
            )
        return ReportSection("駅深度比較", "\n".join(rows))

    def _comparison_geology_table(
        self, cities: List[CityName]
    ) -> ReportSection:
        rows = ["| 都市 | 基盤岩 深度 (m) | 盛土厚 (m) | 沖積層厚 (m) |",
                "|------|--------------|----------|------------|"]
        for city in cities:
            layers = self._store.geology.list_by_city(city)
            bedrock = next((l for l in layers if l.layer_type.value == "bedrock"), None)
            fill = next((l for l in layers if l.layer_type.value == "fill"), None)
            alluvium = next((l for l in layers if l.layer_type.value == "alluvium"), None)
            br_depth = f"{bedrock.depth_from_m:.0f}+" if bedrock else "—"
            fill_t = f"{fill.thickness_m:.1f}" if fill else "—"
            al_t = f"{alluvium.thickness_m:.1f}" if alluvium else "—"
            rows.append(f"| {city.value.upper()} | {br_depth} | {fill_t} | {al_t} |")
        return ReportSection("地質層比較", "\n".join(rows))
