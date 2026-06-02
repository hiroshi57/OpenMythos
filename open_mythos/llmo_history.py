"""
llmo_history.py — Living LLMO 成長履歴 & ダッシュボード

GrowthSnapshot — 特定時点の LLMO 成長状態のスナップショット
GrowthHistory  — スナップショットの時系列リスト (JSON 永続化)
GrowthDiff     — 2 スナップショット間の before/after 比較
generate_growth_report() — HTML 成長レポートページ生成
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ===========================================================================
# GrowthSnapshot — 特定時点の成長状態
# ===========================================================================


@dataclass
class GrowthSnapshot:
    """
    GrowthCycle が完了した時点の Living LLMO の全状態を記録する。
    """

    snapshot_id: str
    timestamp: float
    trigger: str  # 何がこのスナップショットを起こしたか

    # 変換重み (PatternMiner のランキングスコア)
    transformation_weights: dict[str, float] = field(default_factory=dict)

    # Entity 辞書の状態
    entity_dict_size: int = 0
    top_entities: list[str] = field(default_factory=list)

    # パターンランキング TOP5
    pattern_rankings: list[dict[str, Any]] = field(default_factory=list)

    # 失敗学習の状態
    rejection_count: int = 0
    anti_pattern_count: int = 0

    # チャンピオン変換シーケンス
    champion_sequences: list[list[str]] = field(default_factory=list)

    # スコア統計
    avg_score: float = 0.0

    # 成長メタ情報
    feedback_count: int = 0
    cycles_completed: int = 0

    @property
    def formatted_time(self) -> str:
        """人間が読みやすい日時文字列を返す。"""
        import datetime

        dt = datetime.datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "transformation_weights": self.transformation_weights,
            "entity_dict_size": self.entity_dict_size,
            "top_entities": self.top_entities,
            "pattern_rankings": self.pattern_rankings,
            "rejection_count": self.rejection_count,
            "anti_pattern_count": self.anti_pattern_count,
            "champion_sequences": self.champion_sequences,
            "avg_score": self.avg_score,
            "feedback_count": self.feedback_count,
            "cycles_completed": self.cycles_completed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GrowthSnapshot":
        return cls(
            snapshot_id=d["snapshot_id"],
            timestamp=d["timestamp"],
            trigger=d.get("trigger", "manual"),
            transformation_weights=d.get("transformation_weights", {}),
            entity_dict_size=d.get("entity_dict_size", 0),
            top_entities=d.get("top_entities", []),
            pattern_rankings=d.get("pattern_rankings", []),
            rejection_count=d.get("rejection_count", 0),
            anti_pattern_count=d.get("anti_pattern_count", 0),
            champion_sequences=d.get("champion_sequences", []),
            avg_score=d.get("avg_score", 0.0),
            feedback_count=d.get("feedback_count", 0),
            cycles_completed=d.get("cycles_completed", 0),
        )


# ===========================================================================
# GrowthDiff — 2 スナップショット間の差分
# ===========================================================================


@dataclass
class GrowthDiff:
    """
    2 つの GrowthSnapshot を比較して「成長した点」を可視化する。
    """

    from_snapshot: GrowthSnapshot
    to_snapshot: GrowthSnapshot

    @property
    def score_delta(self) -> float:
        """avg_score の変化量を返す。"""
        return round(self.to_snapshot.avg_score - self.from_snapshot.avg_score, 4)

    @property
    def score_delta_pct(self) -> float:
        """avg_score の変化率(%)を返す。"""
        base = self.from_snapshot.avg_score
        if base == 0:
            return 0.0
        return round((self.score_delta / base) * 100, 1)

    @property
    def entity_growth(self) -> int:
        """entity 辞書のサイズ増加数を返す。"""
        return self.to_snapshot.entity_dict_size - self.from_snapshot.entity_dict_size

    @property
    def ranking_changes(self) -> list[dict[str, Any]]:
        """
        パターンランキングの順位変化を返す。
        [{"name": "add_structure", "from_rank": 3, "to_rank": 1}, ...]
        """
        from_ranks = {
            r["name"]: i + 1 for i, r in enumerate(self.from_snapshot.pattern_rankings)
        }
        to_ranks = {
            r["name"]: i + 1 for i, r in enumerate(self.to_snapshot.pattern_rankings)
        }
        all_names = set(from_ranks) | set(to_ranks)
        changes = []
        for name in all_names:
            fr = from_ranks.get(name, 99)
            to = to_ranks.get(name, 99)
            if fr != to:
                changes.append({"name": name, "from_rank": fr, "to_rank": to})
        return sorted(changes, key=lambda x: x["to_rank"])

    @property
    def new_champions(self) -> list[list[str]]:
        """to_snapshot に追加された新チャンピオンを返す。"""
        from_set = {tuple(c) for c in self.from_snapshot.champion_sequences}
        return [
            c for c in self.to_snapshot.champion_sequences if tuple(c) not in from_set
        ]

    @property
    def days_elapsed(self) -> float:
        """2 スナップショット間の経過日数を返す。"""
        return round(
            (self.to_snapshot.timestamp - self.from_snapshot.timestamp) / 86400, 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_snapshot_id": self.from_snapshot.snapshot_id,
            "to_snapshot_id": self.to_snapshot.snapshot_id,
            "score_delta": self.score_delta,
            "score_delta_pct": self.score_delta_pct,
            "entity_growth": self.entity_growth,
            "ranking_changes": self.ranking_changes,
            "new_champions": self.new_champions,
            "days_elapsed": self.days_elapsed,
        }


# ===========================================================================
# GrowthHistory — スナップショットの時系列ストア
# ===========================================================================


class GrowthHistory:
    """
    GrowthSnapshot を時系列で保存・取得する。

    Usage::

        history = GrowthHistory()
        history.add_snapshot(snapshot)
        diff = history.diff()           # first vs latest
        report_html = history.generate_html_report()
    """

    def __init__(self, path: str | Path = "data/llmo_growth_history.json") -> None:
        self._path = Path(path)
        self._snapshots: list[GrowthSnapshot] = []
        self._load()

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------

    def add_snapshot(self, snapshot: GrowthSnapshot) -> None:
        """スナップショットを追加してファイルに保存する。"""
        self._snapshots.append(snapshot)
        self._save()

    def get_all(self) -> list[GrowthSnapshot]:
        """全スナップショットを時系列順で返す。"""
        return list(self._snapshots)

    def get_latest(self) -> GrowthSnapshot | None:
        """最新スナップショットを返す。なければ None。"""
        return self._snapshots[-1] if self._snapshots else None

    def get_first(self) -> GrowthSnapshot | None:
        """最初のスナップショットを返す。なければ None。"""
        return self._snapshots[0] if self._snapshots else None

    def get_by_id(self, snapshot_id: str) -> GrowthSnapshot | None:
        """指定 ID のスナップショットを返す。なければ None。"""
        for s in self._snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        return None

    def count(self) -> int:
        """スナップショット総数を返す。"""
        return len(self._snapshots)

    # ------------------------------------------------------------------
    # 差分比較
    # ------------------------------------------------------------------

    def diff(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
    ) -> GrowthDiff | None:
        """
        from_id → to_id の差分を返す。
        指定なしなら first → latest の差分を返す。
        スナップショットが2件未満なら None を返す。
        """
        if len(self._snapshots) < 2:
            return None

        from_snap = self.get_by_id(from_id) if from_id else self.get_first()
        to_snap = self.get_by_id(to_id) if to_id else self.get_latest()

        if from_snap is None or to_snap is None:
            return None
        if from_snap.snapshot_id == to_snap.snapshot_id:
            return None

        return GrowthDiff(from_snapshot=from_snap, to_snapshot=to_snap)

    # ------------------------------------------------------------------
    # HTML レポート生成
    # ------------------------------------------------------------------

    def generate_html_report(self) -> str:
        """
        「以前はこうだったが、今はこんなに育った」を示す HTML レポートを生成する。
        スナップショットが 0 件の場合は「まだ成長データがありません」ページを返す。
        """
        if not self._snapshots:
            return _HTML_EMPTY_REPORT

        latest = self.get_latest()
        first = self.get_first()
        diff = self.diff()

        # --- スコア推移グラフ用データ ---
        score_labels = [s.formatted_time for s in self._snapshots]
        score_values = [s.avg_score for s in self._snapshots]

        # --- ランキング変化 ---
        ranking_rows = ""
        for r in (latest.pattern_rankings if latest else []):
            ranking_rows += f"<tr><td>{r.get('rank','—')}</td><td>{r.get('name','')}</td><td>{r.get('success_rate', 0):.0%}</td><td>{r.get('avg_delta', 0):.3f}</td></tr>\n"

        # --- イベント年表 ---
        timeline_rows = ""
        for s in self._snapshots:
            timeline_rows += (
                f"<tr><td>{s.formatted_time}</td><td>{s.trigger}</td>"
                f"<td>{s.avg_score:.3f}</td><td>{s.entity_dict_size}</td></tr>\n"
            )

        # --- diff セクション ---
        diff_html = ""
        if diff:
            sign = "+" if diff.score_delta >= 0 else ""
            diff_html = f"""
            <section class="diff-section">
              <h2>📊 成長サマリー（初回 → 現在）</h2>
              <div class="metrics-grid">
                <div class="metric">
                  <div class="metric-label">スコア変化</div>
                  <div class="metric-value">{sign}{diff.score_delta:.3f} ({sign}{diff.score_delta_pct:.1f}%)</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Entity 辞書 増加</div>
                  <div class="metric-value">+{diff.entity_growth} 語</div>
                </div>
                <div class="metric">
                  <div class="metric-label">経過日数</div>
                  <div class="metric-value">{diff.days_elapsed} 日</div>
                </div>
                <div class="metric">
                  <div class="metric-label">新チャンピオン</div>
                  <div class="metric-value">{len(diff.new_champions)} 件</div>
                </div>
              </div>
            </section>
            """

        # --- チャンピオン ---
        champion_html = ""
        if latest and latest.champion_sequences:
            champs = "<br>".join(
                " → ".join(seq) for seq in latest.champion_sequences[:3]
            )
            champion_html = f"<p class='champion'>{champs}</p>"
        else:
            champion_html = "<p>まだチャンピオンがいません</p>"

        # --- 却下・禁止リスト ---
        rejection_html = (
            f"却下パターン: <b>{latest.rejection_count if latest else 0} 件</b> / "
            f"禁止シーケンス: <b>{latest.anti_pattern_count if latest else 0} 件</b>"
        )

        labels_json = json.dumps(score_labels, ensure_ascii=False)
        values_json = json.dumps(score_values)

        first_time = first.formatted_time if first else "—"
        latest_time = latest.formatted_time if latest else "—"
        entity_size = latest.entity_dict_size if latest else 0
        top_entities_str = (
            ", ".join((latest.top_entities or [])[:10]) if latest else "—"
        )
        cycles = latest.cycles_completed if latest else 0

        return _HTML_TEMPLATE.format(
            first_time=first_time,
            latest_time=latest_time,
            entity_size=entity_size,
            top_entities=top_entities_str,
            ranking_rows=ranking_rows,
            timeline_rows=timeline_rows,
            diff_html=diff_html,
            champion_html=champion_html,
            rejection_html=rejection_html,
            cycles=cycles,
            labels_json=labels_json,
            values_json=values_json,
        )

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in self._snapshots], f, ensure_ascii=False, indent=2
            )

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._snapshots = [GrowthSnapshot.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError):
                self._snapshots = []
        else:
            self._snapshots = []


# ===========================================================================
# GrowthSnapshot ファクトリ
# ===========================================================================


def make_snapshot(
    *,
    trigger: str = "manual",
    growth_store: Any = None,
    adapt_store: Any = None,
    feedback_store: Any = None,
    avg_score: float = 0.0,
    cycles_completed: int = 0,
) -> GrowthSnapshot:
    """
    GrowthStore / AdaptStore / FeedbackStore から GrowthSnapshot を生成する。
    引数がなければ空のスナップショットを返す。
    """
    snapshot = GrowthSnapshot(
        snapshot_id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        trigger=trigger,
        avg_score=avg_score,
        cycles_completed=cycles_completed,
    )

    if growth_store is not None:
        pm = growth_store.pattern_miner
        ranking = pm.get_ranking()
        snapshot.pattern_rankings = [
            {
                "rank": i + 1,
                "name": name,
                "success_rate": round(
                    pm.get_record(name).success_rate if pm.get_record(name) else 0.5, 3
                ),
                "avg_delta": round(
                    pm.get_record(name).avg_delta if pm.get_record(name) else 0.0, 4
                ),
            }
            for i, (name, _) in enumerate(ranking[:5])
        ]
        snapshot.transformation_weights = {
            name: round(score, 4) for name, score in ranking[:10]
        }
        snapshot.entity_dict_size = growth_store.entity_kb.size()
        snapshot.top_entities = growth_store.entity_kb.get_entities(min_confidence=0.5)[
            :10
        ]
        snapshot.rejection_count = growth_store.rejection_memory.total_rejections()
        snapshot.anti_pattern_count = (
            growth_store.anti_pattern_registry.total_forbidden_pairs()
        )
        snapshot.champion_sequences = growth_store.champion_promoter.all_champions()

    if feedback_store is not None:
        snapshot.feedback_count = feedback_store.count()

    return snapshot


# ===========================================================================
# HTML テンプレート
# ===========================================================================

_HTML_EMPTY_REPORT = """<!DOCTYPE html>
<html lang="ja">
<head><meta charset="utf-8"><title>Living LLMO 成長レポート</title></head>
<body>
<h1>🌱 Living LLMO — 成長レポート</h1>
<p>まだ成長データがありません。フィードバックを送信すると成長が始まります。</p>
</body>
</html>"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>Living LLMO — 成長レポート</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           max-width: 960px; margin: 0 auto; padding: 24px; background: #f8fafc; color: #1e293b; }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #22c55e; padding-bottom: 8px; }}
    h2 {{ color: #1e40af; margin-top: 32px; }}
    section {{ background: white; border-radius: 12px; padding: 20px;
               margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
    .metric {{ background: #f0fdf4; border-radius: 8px; padding: 16px; text-align: center; }}
    .metric-label {{ font-size: 12px; color: #64748b; margin-bottom: 4px; }}
    .metric-value {{ font-size: 22px; font-weight: 700; color: #16a34a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #e2e8f0; padding: 8px 12px; text-align: left; font-size: 13px; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; font-size: 13px; }}
    .champion {{ background: #fef9c3; border-left: 4px solid #f59e0b;
                 padding: 10px 14px; border-radius: 4px; font-family: monospace; }}
    .chart-wrap {{ max-height: 260px; }}
    .badge {{ display: inline-block; background: #dcfce7; color: #166534;
              border-radius: 99px; padding: 2px 10px; font-size: 12px; margin: 2px; }}
  </style>
</head>
<body>

<h1>🌱 Living LLMO — 成長レポート</h1>
<p>初回: <b>{first_time}</b> &nbsp;/&nbsp; 現在: <b>{latest_time}</b>
   &nbsp;|&nbsp; 成長サイクル完了: <b>{cycles} 回</b></p>

{diff_html}

<section>
  <h2>📈 スコア推移</h2>
  <div class="chart-wrap">
    <canvas id="scoreChart"></canvas>
  </div>
</section>

<section>
  <h2>🔤 Entity 辞書</h2>
  <p>登録語数: <b>{entity_size} 語</b></p>
  <p>上位ワード:
    {top_entities_badges}
  </p>
</section>

<section>
  <h2>🏆 変換ランキング TOP5 (現在)</h2>
  <table>
    <tr><th>#</th><th>変換名</th><th>成功率</th><th>平均改善量</th></tr>
    {ranking_rows}
  </table>
</section>

<section>
  <h2>🥇 チャンピオン変換シーケンス</h2>
  {champion_html}
</section>

<section>
  <h2>🚫 覚えた失敗</h2>
  <p>{rejection_html}</p>
</section>

<section>
  <h2>📅 成長イベント年表</h2>
  <table>
    <tr><th>日時</th><th>トリガー</th><th>avg_score</th><th>Entity辞書</th></tr>
    {timeline_rows}
  </table>
</section>

<script>
const ctx = document.getElementById('scoreChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {labels_json},
    datasets: [{{
      label: 'avg_score',
      data: {values_json},
      borderColor: '#22c55e',
      backgroundColor: 'rgba(34,197,94,0.1)',
      tension: 0.3,
      fill: true,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ min: 0, max: 1 }} }}
  }}
}});
</script>
</body>
</html>"""

# top_entities を badge 化するためのパッチ（テンプレートに注入）
_HTML_TEMPLATE = _HTML_TEMPLATE.replace(
    "{top_entities_badges}",
    "<!-- badges injected at runtime -->",
)


def _render_html(template: str, top_entities: str, **kwargs: Any) -> str:
    """top_entities を badge 形式に変換してテンプレートに注入する。"""
    badges = "".join(
        f'<span class="badge">{e}</span>'
        for e in (top_entities.split(", ") if top_entities else [])
    )
    return template.format(top_entities_badges=badges, **kwargs)


# generate_html_report 内で _render_html を使うようにパッチ
def _patched_generate_html_report(self: GrowthHistory) -> str:  # noqa: F811
    if not self._snapshots:
        return _HTML_EMPTY_REPORT

    latest = self.get_latest()
    first = self.get_first()
    diff = self.diff()

    score_labels = [s.formatted_time for s in self._snapshots]
    score_values = [s.avg_score for s in self._snapshots]

    ranking_rows = ""
    for r in (latest.pattern_rankings if latest else []):
        ranking_rows += (
            f"<tr><td>{r.get('rank','—')}</td><td>{r.get('name','')}</td>"
            f"<td>{r.get('success_rate', 0):.0%}</td>"
            f"<td>{r.get('avg_delta', 0):.3f}</td></tr>\n"
        )

    timeline_rows = ""
    for s in self._snapshots:
        timeline_rows += (
            f"<tr><td>{s.formatted_time}</td><td>{s.trigger}</td>"
            f"<td>{s.avg_score:.3f}</td><td>{s.entity_dict_size}</td></tr>\n"
        )

    diff_html = ""
    if diff:
        sign = "+" if diff.score_delta >= 0 else ""
        diff_html = (
            f'<section class="diff-section"><h2>📊 成長サマリー（初回 → 現在）</h2>'
            f'<div class="metrics-grid">'
            f'<div class="metric"><div class="metric-label">スコア変化</div>'
            f'<div class="metric-value">{sign}{diff.score_delta:.3f} ({sign}{diff.score_delta_pct:.1f}%)</div></div>'
            f'<div class="metric"><div class="metric-label">Entity 辞書 増加</div>'
            f'<div class="metric-value">+{diff.entity_growth} 語</div></div>'
            f'<div class="metric"><div class="metric-label">経過日数</div>'
            f'<div class="metric-value">{diff.days_elapsed} 日</div></div>'
            f'<div class="metric"><div class="metric-label">新チャンピオン</div>'
            f'<div class="metric-value">{len(diff.new_champions)} 件</div></div>'
            f"</div></section>"
        )

    champion_html = ""
    if latest and latest.champion_sequences:
        champs = "<br>".join(" → ".join(seq) for seq in latest.champion_sequences[:3])
        champion_html = f"<p class='champion'>{champs}</p>"
    else:
        champion_html = "<p>まだチャンピオンがいません</p>"

    rejection_html = (
        f"却下パターン: <b>{latest.rejection_count if latest else 0} 件</b> / "
        f"禁止シーケンス: <b>{latest.anti_pattern_count if latest else 0} 件</b>"
    )

    labels_json = json.dumps(score_labels, ensure_ascii=False)
    values_json = json.dumps(score_values)
    top_entities_str = ", ".join((latest.top_entities or [])[:10]) if latest else ""

    return _render_html(
        _HTML_TEMPLATE,
        top_entities=top_entities_str,
        first_time=first.formatted_time if first else "—",
        latest_time=latest.formatted_time if latest else "—",
        entity_size=latest.entity_dict_size if latest else 0,
        ranking_rows=ranking_rows,
        timeline_rows=timeline_rows,
        diff_html=diff_html,
        champion_html=champion_html,
        rejection_html=rejection_html,
        cycles=latest.cycles_completed if latest else 0,
        labels_json=labels_json,
        values_json=values_json,
    )


# メソッドを差し替え
GrowthHistory.generate_html_report = _patched_generate_html_report  # type: ignore[method-assign]
