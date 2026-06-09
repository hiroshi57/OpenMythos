"""
Sprint 58 — LLMO ダッシュボード・CEP管理・競合分析

AIサーチで「選ばれるブランド」になるための LLMO 定点観測・管理フレームワーク。

オブジェクト:
  CepCategory      : CEP（AIに聞くシチュエーション）のカテゴリー分類
  CepEntry         : 単一の CEP 定義（シチュエーション + ターゲット + キーワード群）
  CepStore         : CEP ライブラリ (CRUD + カテゴリーフィルタ)
  MentionSnapshot  : 特定日時・プロンプトでの言及率スナップショット
  CompetitorEntry  : 競合ブランド情報
  CompetitorAnalysis: 競合 vs 自社の比較分析
  LlmoDashboard    : 言及率・引用率・参照率の時系列管理
  LlmoReportEngine : レポート生成（Markdown / JSON）

設計方針:
  - CEP はカテゴリー (問題 / 比較 / 推薦 / 使い方) で分類
  - スナップショットを蓄積して定点観測を可能にする
  - 競合比較では「自社言及率 - 競合言及率」の差分を追跡
  - 全データはメモリ内保持（永続化は外部 DB に委ねる設計）
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CEP 管理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CepCategory(str, Enum):
    PROBLEM     = "problem"      # 問題解決型: 「〇〇が困ったとき」
    COMPARISON  = "comparison"   # 比較型: 「〇〇 vs △△ どっちがいい？」
    RECOMMEND   = "recommend"    # 推薦型: 「〇〇を探している」
    HOW_TO      = "how_to"       # 使い方型: 「〇〇の使い方を教えて」
    PURCHASE    = "purchase"     # 購入前型: 「〇〇を買う前に確認したい」
    OTHER       = "other"        # その他


@dataclass
class CepEntry:
    """単一の CEP（Customer Entry Point）定義"""
    id:          str
    scenario:    str                    # AIに聞くシチュエーション文
    category:    CepCategory            = CepCategory.OTHER
    target:      Optional[str]          = None  # ターゲットペルソナ
    keywords:    List[str]              = field(default_factory=list)
    priority:    int                    = 3     # 1(最高)〜5(最低)
    notes:       str                    = ""
    created_at:  int                    = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":         self.id,
            "scenario":   self.scenario,
            "category":   self.category.value,
            "target":     self.target,
            "keywords":   self.keywords,
            "priority":   self.priority,
            "notes":      self.notes,
            "created_at": self.created_at,
        }


class CepStore:
    """CEP ライブラリ（CRUD + フィルタ）"""

    def __init__(self) -> None:
        self._entries: Dict[str, CepEntry] = {}

    def add(
        self,
        scenario:  str,
        category:  CepCategory = CepCategory.OTHER,
        target:    Optional[str] = None,
        keywords:  Optional[List[str]] = None,
        priority:  int = 3,
        notes:     str = "",
    ) -> CepEntry:
        entry = CepEntry(
            id=uuid.uuid4().hex[:12],
            scenario=scenario,
            category=category,
            target=target,
            keywords=keywords or [],
            priority=priority,
            notes=notes,
        )
        self._entries[entry.id] = entry
        return entry

    def get(self, cep_id: str) -> Optional[CepEntry]:
        return self._entries.get(cep_id)

    def list_all(self) -> List[CepEntry]:
        return sorted(self._entries.values(), key=lambda e: e.priority)

    def by_category(self, category: CepCategory) -> List[CepEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def by_priority(self, max_priority: int = 2) -> List[CepEntry]:
        return [e for e in self._entries.values() if e.priority <= max_priority]

    def delete(self, cep_id: str) -> bool:
        if cep_id in self._entries:
            del self._entries[cep_id]
            return True
        return False

    def count(self) -> int:
        return len(self._entries)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 言及率スナップショット
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class MentionSnapshot:
    """
    特定日時・プロンプトでの言及率スナップショット。

    mention_rate  : 指定プロンプトでブランドが言及された割合 (0〜1)
    citation_rate : 自社サイトが引用元として記載された割合
    reference_rate: AIクローラーが参照した割合
    """
    id:             str
    brand_name:     str
    cep_id:         Optional[str]   = None
    prompt:         str             = ""
    mention_rate:   float           = 0.0
    citation_rate:  float           = 0.0
    reference_rate: float           = 0.0
    measured_at:    int             = field(default_factory=lambda: int(time.time()))
    notes:          str             = ""

    @property
    def overall_score(self) -> float:
        """3指標の加重平均（言及0.5 + 引用0.3 + 参照0.2）"""
        return round(
            self.mention_rate   * 0.50 +
            self.citation_rate  * 0.30 +
            self.reference_rate * 0.20,
            4,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":             self.id,
            "brand_name":     self.brand_name,
            "cep_id":         self.cep_id,
            "prompt":         self.prompt,
            "mention_rate":   self.mention_rate,
            "citation_rate":  self.citation_rate,
            "reference_rate": self.reference_rate,
            "overall_score":  self.overall_score,
            "measured_at":    self.measured_at,
            "notes":          self.notes,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 競合分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CompetitorEntry:
    """競合ブランド情報"""
    id:          str
    name:        str
    category:    str                    = ""
    url:         Optional[str]          = None
    keywords:    List[str]              = field(default_factory=list)
    notes:       str                    = ""
    created_at:  int                    = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "category": self.category, "url": self.url,
            "keywords": self.keywords, "notes": self.notes,
        }


@dataclass
class CompetitorAnalysis:
    """競合 vs 自社 の比較分析結果"""
    brand_name:    str
    competitor_name: str
    prompt:        str
    our_mention:   float    = 0.0
    comp_mention:  float    = 0.0
    gap:           float    = 0.0   # our - comp
    analyzed_at:   int      = field(default_factory=lambda: int(time.time()))

    @property
    def is_winning(self) -> bool:
        return self.gap > 0

    @property
    def gap_label(self) -> str:
        if self.gap > 0.2:   return "大幅リード"
        elif self.gap > 0:   return "僅差リード"
        elif self.gap == 0:  return "同率"
        elif self.gap > -0.2:return "僅差劣後"
        else:                return "大幅劣後"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand_name":      self.brand_name,
            "competitor_name": self.competitor_name,
            "prompt":          self.prompt,
            "our_mention":     self.our_mention,
            "comp_mention":    self.comp_mention,
            "gap":             round(self.gap, 4),
            "gap_label":       self.gap_label,
            "is_winning":      self.is_winning,
            "analyzed_at":     self.analyzed_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LlmoDashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LlmoDashboard:
    """
    LLMO 指標の定点観測・時系列管理ダッシュボード。

    Usage::
        db = LlmoDashboard(brand_name="SunGuard")
        snap = db.add_snapshot(cep_id="c1", prompt="アウトドア向け日焼け止め",
                               mention_rate=0.45, citation_rate=0.30, reference_rate=0.25)
        print(db.trend())        # 時系列トレンド
        print(db.latest_score()) # 最新総合スコア
    """

    def __init__(self, brand_name: str) -> None:
        self.brand_name  = brand_name
        self._snapshots: List[MentionSnapshot] = []
        self._competitors: Dict[str, CompetitorEntry] = {}

    # ── スナップショット管理 ──────────────────────────────────────

    def add_snapshot(
        self,
        prompt:         str,
        mention_rate:   float,
        citation_rate:  float   = 0.0,
        reference_rate: float   = 0.0,
        cep_id:         Optional[str] = None,
        notes:          str     = "",
    ) -> MentionSnapshot:
        snap = MentionSnapshot(
            id=uuid.uuid4().hex[:12],
            brand_name=self.brand_name,
            cep_id=cep_id,
            prompt=prompt,
            mention_rate=mention_rate,
            citation_rate=citation_rate,
            reference_rate=reference_rate,
            notes=notes,
        )
        self._snapshots.append(snap)
        return snap

    def snapshots(self) -> List[MentionSnapshot]:
        return list(self._snapshots)

    def latest_snapshot(self) -> Optional[MentionSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def latest_score(self) -> float:
        snap = self.latest_snapshot()
        return snap.overall_score if snap else 0.0

    def avg_mention_rate(self) -> float:
        if not self._snapshots:
            return 0.0
        return round(sum(s.mention_rate for s in self._snapshots) / len(self._snapshots), 4)

    def trend(self) -> List[Dict[str, Any]]:
        """時系列トレンド（全スナップショットを日時順で返す）"""
        return [s.to_dict() for s in self._snapshots]

    def trend_delta(self) -> float:
        """最新 vs 前回の overall_score 差分。スナップショットが 1 件以下は 0.0"""
        if len(self._snapshots) < 2:
            return 0.0
        latest = self._snapshots[-1].overall_score
        prev   = self._snapshots[-2].overall_score
        return round(latest - prev, 4)

    # ── 競合管理 ─────────────────────────────────────────────────

    def add_competitor(
        self,
        name:     str,
        category: str             = "",
        url:      Optional[str]   = None,
        keywords: Optional[List[str]] = None,
    ) -> CompetitorEntry:
        comp = CompetitorEntry(
            id=uuid.uuid4().hex[:12],
            name=name,
            category=category,
            url=url,
            keywords=keywords or [],
        )
        self._competitors[comp.id] = comp
        return comp

    def competitors(self) -> List[CompetitorEntry]:
        return list(self._competitors.values())

    def analyze_competitor(
        self,
        competitor_id:   str,
        prompt:          str,
        our_mention:     float,
        comp_mention:    float,
    ) -> Optional[CompetitorAnalysis]:
        comp = self._competitors.get(competitor_id)
        if comp is None:
            return None
        return CompetitorAnalysis(
            brand_name=self.brand_name,
            competitor_name=comp.name,
            prompt=prompt,
            our_mention=our_mention,
            comp_mention=comp_mention,
            gap=round(our_mention - comp_mention, 4),
        )

    def snapshot_count(self) -> int:
        return len(self._snapshots)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LlmoReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LlmoReportEngine:
    """LLMO レポート生成（Markdown / JSON）"""

    def __init__(self, dashboard: LlmoDashboard) -> None:
        self._db = dashboard

    def summary(self) -> Dict[str, Any]:
        """ダッシュボードのサマリーを JSON で返す"""
        return {
            "brand_name":      self._db.brand_name,
            "snapshots":       self._db.snapshot_count(),
            "latest_score":    self._db.latest_score(),
            "avg_mention_rate": self._db.avg_mention_rate(),
            "trend_delta":     self._db.trend_delta(),
            "competitors":     len(self._db.competitors()),
        }

    def to_markdown(self) -> str:
        """LLMO 月次レポートの Markdown を生成"""
        snap   = self._db.latest_snapshot()
        delta  = self._db.trend_delta()
        d_sign = "+" if delta >= 0 else ""

        lines = [
            f"# LLMO 定点観測レポート: {self._db.brand_name}",
            "",
            "## 最新スコア",
            "",
        ]

        if snap:
            lines += [
                "| 指標 | スコア |",
                "|------|--------|",
                f"| 言及ポテンシャル | {snap.mention_rate:.2%} |",
                f"| 引用ポテンシャル | {snap.citation_rate:.2%} |",
                f"| 参照ポテンシャル | {snap.reference_rate:.2%} |",
                f"| **総合スコア**  | **{snap.overall_score:.4f}** |",
                "",
                f"前回比: {d_sign}{delta:.4f}",
                "",
            ]
        else:
            lines += ["*スナップショットがありません*", ""]

        if self._db.competitors():
            lines += [
                "## 競合ブランド登録数",
                f"{len(self._db.competitors())} ブランド",
                "",
            ]

        lines += [
            "## スナップショット履歴",
            f"計 {self._db.snapshot_count()} 件",
            "",
        ]
        return "\n".join(lines)
