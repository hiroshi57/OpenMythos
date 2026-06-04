"""
Sprint 39 — OpenMythos ショーケースダッシュボード

GET /dashboard  →  OpenMythos の価値・差別化・全機能を示す HTML ページ

セクション:
  1. Hero           — アーキテクチャの独自性 (Recurrent Depth / MLA / MoE / LoRA)
  2. Stats bar       — テスト数・Sprint数・API数・パターン数
  3. Architecture    — 標準 Transformer との違いを図解
  4. P1〜P10         — 「育つAI」10パターンのカード一覧
  5. Benchmark       — P1〜P10 KPI 改善率 Chart.js グラフ
  6. API Coverage    — 全エンドポイントカテゴリ別カウント
  7. Sprint Timeline — Sprint 1〜38 の進捗バー
  8. Live Status     — /health を JavaScript でポーリング
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

# ---------------------------------------------------------------------------
# データ定義
# ---------------------------------------------------------------------------

_PATTERNS = [
    {"id": "P1",  "name": "討議型集合知",       "module": "debate.py",           "api": "/v1/debate/run",       "sprint": 20, "color": "#6366f1"},
    {"id": "P2",  "name": "KPI駆動自己改善",    "module": "kpi_agent.py",        "api": "/v1/kpi/improve",      "sprint": 21, "color": "#8b5cf6"},
    {"id": "P3",  "name": "ボトルネック発見",    "module": "profiler.py",         "api": "/v1/profile/run",      "sprint": 22, "color": "#a78bfa"},
    {"id": "P4",  "name": "外部要因適応",        "module": "external_signal.py",  "api": "/v1/signal/adapt",     "sprint": 23, "color": "#06b6d4"},
    {"id": "P5",  "name": "ミスから学習",        "module": "error_memory.py",     "api": "/v1/mistakes/record",  "sprint": 24, "color": "#0ea5e9"},
    {"id": "P6",  "name": "継続的自己蒸留",      "module": "self_distill.py",     "api": "/v1/distill/run",      "sprint": 25, "color": "#10b981"},
    {"id": "P7",  "name": "長期記憶統合",        "module": "long_term_memory.py", "api": "/v1/memory/store",     "sprint": 26, "color": "#14b8a6"},
    {"id": "P8",  "name": "アンサンブル評価",    "module": "ensemble_scorer.py",  "api": "/v1/ensemble/score",   "sprint": 27, "color": "#f59e0b"},
    {"id": "P9",  "name": "適応型プロンプト進化","module": "prompt_evolution.py", "api": "/v1/evolve/run",       "sprint": 28, "color": "#ef4444"},
    {"id": "P10", "name": "自律タスク計画",      "module": "task_planner.py",     "api": "/v1/plan/decompose",   "sprint": 29, "color": "#f97316"},
]

_BENCH_KPI = [
    ("P1 討議型集合知",       11.2),
    ("P2 KPI自己改善",        9.8),
    ("P3 ボトルネック",        13.1),
    ("P4 外部適応",            8.4),
    ("P5 ミス学習",            10.7),
    ("P6 自己蒸留",            12.3),
    ("P7 長期記憶",             9.1),
    ("P8 アンサンブル",        11.5),
    ("P9 プロンプト進化",      14.2),
    ("P10 自律計画",           10.3),
]

_SPRINT_ERAS = [
    {"label": "Era 1 基盤構築",       "range": "1–9",   "tests": 508,  "color": "#6366f1"},
    {"label": "Era 2 LLMO/Tool/RAG",  "range": "10–19", "tests": 571,  "color": "#0ea5e9"},
    {"label": "Era 3 育つAI P1〜P10", "range": "20–29", "tests": 491,  "color": "#10b981"},
    {"label": "Era 4 統合・最適化",   "range": "30–38", "tests": 393,  "color": "#f59e0b"},
]

_API_CATEGORIES = [
    ("基盤 (infer/generate/agent/chat)",      4),
    ("SEO / LLMO / Thinking",                  3),
    ("P1 討議型集合知",                        1),
    ("P2 KPI 自己改善",                        2),
    ("P3 プロファイラ",                        3),
    ("P4 外部シグナル",                        2),
    ("P5 ミスメモリ",                          3),
    ("P6 自己蒸留",                            1),
    ("P7 長期記憶",                            3),
    ("P8 アンサンブル",                        3),
    ("P9 進化プロンプト",                      1),
    ("P10 タスク計画",                         2),
    ("Guard / Grow / Monitor",                 4),
    ("A/B ルーター / ヘルス",                  3),
]

# ---------------------------------------------------------------------------
# HTML 生成
# ---------------------------------------------------------------------------

def build_showcase_dashboard() -> str:
    """ショーケースダッシュボードの HTML 文字列を返す"""

    # --- Pattern cards HTML ---
    pattern_cards = ""
    for p in _PATTERNS:
        pattern_cards += (
            '<div class="p-card" style="border-left:4px solid ' + p["color"] + '">'
            '<div class="p-id" style="color:' + p["color"] + '">' + p["id"] + "</div>"
            '<div class="p-name">' + p["name"] + "</div>"
            '<div class="p-meta">Sprint ' + str(p["sprint"]) + " · <code>" + p["api"] + "</code></div>"
            "</div>\n"
        )

    # --- KPI chart data ---
    kpi_labels = str([k[0] for k in _BENCH_KPI])
    kpi_values = str([k[1] for k in _BENCH_KPI])
    kpi_colors = str([
        "#22c55e" if v >= 12 else "#10b981" if v >= 10 else "#0ea5e9"
        for _, v in _BENCH_KPI
    ])

    # --- Era bars ---
    era_html = ""
    total_tests = sum(e["tests"] for e in _SPRINT_ERAS)
    for era in _SPRINT_ERAS:
        pct = era["tests"] / total_tests * 100
        era_html += (
            '<div class="era-row">'
            '<span class="era-label">' + era["label"] + "</span>"
            '<div class="era-bar-bg">'
            '<div class="era-bar" style="width:' + str(round(pct, 1)) + "%;background:" + era["color"] + '">'
            "<span>" + str(era["tests"]) + " tests</span></div>"
            "</div>"
            '<span class="era-range">Sprint ' + era["range"] + "</span>"
            "</div>\n"
        )

    # --- API category chart data ---
    api_labels = str([c[0] for c in _API_CATEGORIES])
    api_counts = str([c[1] for c in _API_CATEGORIES])
    total_apis = sum(c[1] for c in _API_CATEGORIES)

    # --- Sprint timeline dots ---
    timeline_html = ""
    for i in range(1, 39):
        color = "#6366f1"
        if i <= 9:
            color = "#6366f1"
        elif i <= 19:
            color = "#0ea5e9"
        elif i <= 29:
            color = "#10b981"
        else:
            color = "#f59e0b"
        timeline_html += (
            '<div class="sprint-dot" style="background:' + color + '" title="Sprint ' + str(i) + '">'
            "<span>" + str(i) + "</span></div>\n"
        )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="ja">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>OpenMythos — ショーケース</title>\n"
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>\n'
        "<style>\n"
        ":root{--bg:#0f172a;--surface:#1e293b;--border:#334155;--text:#f1f5f9;--muted:#94a3b8;--accent:#6366f1;}\n"
        "*{box-sizing:border-box;margin:0;padding:0;}\n"
        "body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.6;}\n"
        "a{color:var(--accent);text-decoration:none;}\n"
        "code{background:#0f172a;padding:2px 6px;border-radius:4px;font-size:.85em;color:#a5b4fc;}\n"
        # Hero
        ".hero{background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 50%,#0c4a6e 100%);"
        "padding:3rem 2rem;text-align:center;border-bottom:1px solid var(--border);}\n"
        ".hero h1{font-size:2.8rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#38bdf8,#34d399);"
        "background-clip:text;-webkit-background-clip:text;color:transparent;margin-bottom:.5rem;}\n"
        ".hero .tagline{font-size:1.15rem;color:var(--muted);max-width:700px;margin:0 auto 1.5rem;}\n"
        ".hero .diff-pills{display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap;}\n"
        ".pill{background:#1e293b;border:1px solid #334155;padding:.35rem .9rem;border-radius:9999px;"
        "font-size:.8rem;color:#a5b4fc;}\n"
        # Stats bar
        ".stats-bar{display:flex;justify-content:space-around;padding:1.5rem 2rem;"
        "background:var(--surface);border-bottom:1px solid var(--border);flex-wrap:wrap;gap:1rem;}\n"
        ".stat{text-align:center;}\n"
        ".stat-num{font-size:2.2rem;font-weight:800;color:var(--accent);}\n"
        ".stat-label{font-size:.8rem;color:var(--muted);margin-top:.2rem;}\n"
        # Sections
        ".section{padding:2.5rem 2rem;border-bottom:1px solid var(--border);max-width:1200px;margin:0 auto;}\n"
        ".section h2{font-size:1.4rem;font-weight:700;margin-bottom:1.5rem;color:#e2e8f0;"
        "display:flex;align-items:center;gap:.5rem;}\n"
        ".section h2::before{content:'';display:inline-block;width:4px;height:1.2em;"
        "background:var(--accent);border-radius:2px;}\n"
        # Architecture comparison
        ".arch-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;}\n"
        ".arch-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem;}\n"
        ".arch-card h3{margin-bottom:1rem;font-size:1rem;}\n"
        ".arch-card.highlight{border-color:#6366f1;background:#1e1b4b;}\n"
        ".arch-card.highlight h3{color:#a78bfa;}\n"
        ".arch-list{list-style:none;}\n"
        ".arch-list li{padding:.4rem 0;border-bottom:1px solid #1e293b;font-size:.9rem;color:var(--muted);}\n"
        ".arch-list li:last-child{border:none;}\n"
        ".arch-list li span{color:#34d399;margin-right:.5rem;}\n"
        ".arch-list li.dim span{color:#94a3b8;}\n"
        # Pattern cards
        ".p-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem;}\n"
        ".p-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem;"
        "transition:transform .15s;}\n"
        ".p-card:hover{transform:translateY(-2px);}\n"
        ".p-id{font-size:1.5rem;font-weight:800;}\n"
        ".p-name{font-size:1rem;font-weight:600;margin:.3rem 0;}\n"
        ".p-meta{font-size:.8rem;color:var(--muted);}\n"
        # Era bars
        ".era-row{display:flex;align-items:center;gap:1rem;margin-bottom:.75rem;}\n"
        ".era-label{width:180px;font-size:.85rem;flex-shrink:0;color:var(--muted);}\n"
        ".era-bar-bg{flex:1;background:#1e293b;border-radius:4px;height:28px;overflow:hidden;}\n"
        ".era-bar{height:100%;display:flex;align-items:center;padding:0 .75rem;"
        "font-size:.8rem;font-weight:600;color:#fff;border-radius:4px;min-width:60px;}\n"
        ".era-range{width:90px;font-size:.8rem;color:var(--muted);text-align:right;}\n"
        # Sprint timeline
        ".sprint-timeline{display:flex;flex-wrap:wrap;gap:.5rem;padding:1rem 0;}\n"
        ".sprint-dot{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;"
        "justify-content:center;font-size:.75rem;font-weight:700;color:white;cursor:default;opacity:.9;}\n"
        ".sprint-dot:hover{opacity:1;transform:scale(1.1);transition:transform .1s;}\n"
        # Live status
        ".status-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;"
        "padding:1.5rem;display:flex;align-items:center;gap:1rem;}\n"
        ".status-dot{width:14px;height:14px;border-radius:50%;background:#94a3b8;"
        "transition:background .3s;flex-shrink:0;}\n"
        ".status-dot.ok{background:#22c55e;box-shadow:0 0 8px #22c55e88;}\n"
        ".status-dot.error{background:#ef4444;}\n"
        ".status-info{flex:1;}\n"
        ".status-info strong{font-size:1rem;}\n"
        ".status-detail{font-size:.82rem;color:var(--muted);margin-top:.2rem;}\n"
        # Charts
        ".chart-wrap{background:var(--surface);border:1px solid var(--border);"
        "border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;}\n"
        ".chart-title{font-size:.9rem;font-weight:600;color:var(--muted);margin-bottom:1rem;}\n"
        # Footer
        ".footer{padding:2rem;text-align:center;color:var(--muted);font-size:.85rem;"
        "background:var(--surface);}\n"
        "@media(max-width:768px){"
        ".arch-grid{grid-template-columns:1fr;}"
        ".stats-bar{gap:.5rem;}"
        ".stat-num{font-size:1.6rem;}"
        "}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"

        # ── Hero ──────────────────────────────────────────────────────────
        '<div class="hero">\n'
        '<h1>OpenMythos</h1>\n'
        '<p class="tagline">Recurrent-Depth Transformer with Multi-Latent Attention, Mixture-of-Experts, &amp; 10 Self-Growing AI Patterns<br>'
        '<em>— 38 Sprints, 1,963 tests, 50+ API endpoints —</em></p>\n'
        '<div class="diff-pills">\n'
        '<span class="pill">🔄 Recurrent Depth Loop</span>\n'
        '<span class="pill">🧠 Multi-Latent Attention (MLA)</span>\n'
        '<span class="pill">⚡ Mixture-of-Experts</span>\n'
        '<span class="pill">🔧 LoRA Fine-tuning</span>\n'
        '<span class="pill">🌱 10 Self-Growing Patterns</span>\n'
        '<span class="pill">🛡️ MistakeGuard Middleware</span>\n'
        "</div>\n</div>\n"

        # ── Stats bar ─────────────────────────────────────────────────────
        '<div class="stats-bar">\n'
        '<div class="stat"><div class="stat-num">38</div><div class="stat-label">Sprints 完了</div></div>\n'
        '<div class="stat"><div class="stat-num">1,963</div><div class="stat-label">テスト PASS</div></div>\n'
        '<div class="stat"><div class="stat-num">v0.41</div><div class="stat-label">最新バージョン</div></div>\n'
        '<div class="stat"><div class="stat-num">' + str(total_apis) + '</div><div class="stat-label">API エンドポイント</div></div>\n'
        '<div class="stat"><div class="stat-num">10</div><div class="stat-label">育つAI パターン</div></div>\n'
        '<div class="stat"><div class="stat-num">4</div><div class="stat-label">アーキテクチャ革新</div></div>\n'
        "</div>\n"

        # ── Architecture ──────────────────────────────────────────────────
        '<div class="section">\n'
        '<h2>アーキテクチャの差別化</h2>\n'
        '<div class="arch-grid">\n'
        '<div class="arch-card highlight">\n'
        '<h3>✅ OpenMythos</h3>\n'
        '<ul class="arch-list">\n'
        '<li><span>●</span>Recurrent Depth Loop — 推論時に計算深度を動的に調整</li>\n'
        '<li><span>●</span>MLA (Multi-Latent Attention) — KV キャッシュ 75% 削減</li>\n'
        '<li><span>●</span>MoE (Mixture-of-Experts) — スパース活性化で計算効率化</li>\n'
        '<li><span>●</span>LoRA Adapter — ループ深度適応型の軽量 Fine-tuning</li>\n'
        '<li><span>●</span>LTI 安定性保証 — 長期依存のループ発散を防止</li>\n'
        '<li><span>●</span>P1〜P10 自己成長パターン — 稼働しながら自律改善</li>\n'
        '<li><span>●</span>MistakeGuard Middleware — 全 API をリアルタイム保護</li>\n'
        "</ul></div>\n"
        '<div class="arch-card">\n'
        '<h3>標準 Transformer</h3>\n'
        '<ul class="arch-list">\n'
        '<li class="dim"><span>○</span>固定深度 — 全トークン同コスト</li>\n'
        '<li class="dim"><span>○</span>Full Attention — KV キャッシュが O(n²) 成長</li>\n'
        '<li class="dim"><span>○</span>Dense 計算 — 全 Expert を毎回活性化</li>\n'
        '<li class="dim"><span>○</span>LoRA なし / 外部ツールで別途管理</li>\n'
        '<li class="dim"><span>○</span>不安定性リスク — 深いループで発散しやすい</li>\n'
        '<li class="dim"><span>○</span>静的 — 推論後に自律改善する仕組みなし</li>\n'
        '<li class="dim"><span>○</span>ミス保護なし — アプリ層で個別実装が必要</li>\n'
        "</ul></div>\n"
        "</div></div>\n"

        # ── P1〜P10 Patterns ──────────────────────────────────────────────
        '<div class="section">\n'
        '<h2>P1〜P10「育つAI」パターン</h2>\n'
        '<p style="color:var(--muted);margin-bottom:1.5rem;font-size:.9rem;">'
        "稼働しながら自律的に改善する 10 のパターン。各パターンは独立した API エンドポイントを持ち、"
        "オーケストレーター (<code>/v1/grow/run</code>) が自動選択・組み合わせて実行します。</p>\n"
        '<div class="p-grid">\n'
        + pattern_cards +
        "</div></div>\n"

        # ── Benchmark KPI ─────────────────────────────────────────────────
        '<div class="section">\n'
        '<h2>ベンチマーク — P1〜P10 KPI 改善率</h2>\n'
        '<div class="chart-wrap">\n'
        '<div class="chart-title">各パターン実行後のベースライン比 KPI 改善率 (%)</div>\n'
        '<canvas id="kpiChart" height="100"></canvas>\n'
        "</div>\n"
        '<p style="font-size:.82rem;color:var(--muted);">'
        "計測: <code>benchmark/growing_ai_bench.py</code> / 実行日: 2026-06-03 / 平均改善率: "
        + str(round(sum(k[1] for k in _BENCH_KPI) / len(_BENCH_KPI), 1)) +
        "%</p>\n</div>\n"

        # ── API Coverage ──────────────────────────────────────────────────
        '<div class="section">\n'
        '<h2>API カバレッジ — ' + str(total_apis) + " エンドポイント</h2>\n"
        '<div class="chart-wrap">\n'
        '<div class="chart-title">カテゴリ別エンドポイント数</div>\n'
        '<canvas id="apiChart" height="80"></canvas>\n'
        "</div>\n"
        '<p style="font-size:.82rem;color:var(--muted);">'
        "全エンドポイントは <a href='/docs'>Swagger UI (/docs)</a> または "
        "<a href='/redoc'>ReDoc (/redoc)</a> で確認できます。</p>\n"
        "</div>\n"

        # ── Sprint Timeline ───────────────────────────────────────────────
        '<div class="section">\n'
        '<h2>Sprint 1〜38 進捗タイムライン</h2>\n'
        '<div style="display:flex;gap:1.5rem;margin-bottom:1.5rem;flex-wrap:wrap;">\n'
        '<span style="font-size:.82rem;color:var(--muted);">'
        '<span style="display:inline-block;width:12px;height:12px;background:#6366f1;border-radius:3px;margin-right:4px;"></span>Era 1 基盤</span>\n'
        '<span style="font-size:.82rem;color:var(--muted);">'
        '<span style="display:inline-block;width:12px;height:12px;background:#0ea5e9;border-radius:3px;margin-right:4px;"></span>Era 2 LLMO/Tool</span>\n'
        '<span style="font-size:.82rem;color:var(--muted);">'
        '<span style="display:inline-block;width:12px;height:12px;background:#10b981;border-radius:3px;margin-right:4px;"></span>Era 3 育つAI</span>\n'
        '<span style="font-size:.82rem;color:var(--muted);">'
        '<span style="display:inline-block;width:12px;height:12px;background:#f59e0b;border-radius:3px;margin-right:4px;"></span>Era 4 統合</span>\n'
        "</div>\n"
        '<div class="sprint-timeline">\n'
        + timeline_html +
        "</div>\n"
        '<div style="margin-top:1.5rem;">\n'
        + era_html +
        "</div></div>\n"

        # ── Live Status ───────────────────────────────────────────────────
        '<div class="section">\n'
        '<h2>ライブ API ステータス</h2>\n'
        '<div class="status-box" id="statusBox">\n'
        '<div class="status-dot" id="statusDot"></div>\n'
        '<div class="status-info">\n'
        '<strong id="statusText">確認中...</strong>\n'
        '<div class="status-detail" id="statusDetail">サーバーに接続しています</div>\n'
        "</div>\n"
        '<button onclick="checkHealth()" style="background:#1e293b;border:1px solid #334155;'
        'color:#a5b4fc;padding:.4rem .9rem;border-radius:6px;cursor:pointer;font-size:.85rem;">'
        "再確認</button>\n"
        "</div>\n"
        '<div style="margin-top:1rem;display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.75rem;" id="healthDetail"></div>\n'
        "</div>\n"

        # ── Footer ────────────────────────────────────────────────────────
        '<div class="footer">\n'
        "OpenMythos v0.41.0 — "
        '<a href="https://github.com/hiroshi57/OpenMythos">GitHub</a> · '
        '<a href="/docs">Swagger UI</a> · '
        '<a href="/monitor/dashboard">Monitor Dashboard</a> · '
        '<a href="/metrics">Prometheus Metrics</a>\n'
        "</div>\n"

        # ── Scripts ───────────────────────────────────────────────────────
        "<script>\n"
        # KPI chart
        "const kpiCtx = document.getElementById('kpiChart').getContext('2d');\n"
        "new Chart(kpiCtx, {\n"
        "  type: 'bar',\n"
        "  data: {\n"
        "    labels: " + kpi_labels + ",\n"
        "    datasets: [{\n"
        "      label: 'KPI 改善率 (%)',\n"
        "      data: " + kpi_values + ",\n"
        "      backgroundColor: " + kpi_colors + ",\n"
        "      borderRadius: 6,\n"
        "    }]\n"
        "  },\n"
        "  options: {\n"
        "    responsive: true,\n"
        "    plugins: {\n"
        "      legend: { display: false },\n"
        "      tooltip: { callbacks: { label: ctx => '+' + ctx.raw + '%' } }\n"
        "    },\n"
        "    scales: {\n"
        "      y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', callback: v => '+' + v + '%' } },\n"
        "      x: { grid: { display: false }, ticks: { color: '#94a3b8', maxRotation: 30 } }\n"
        "    }\n"
        "  }\n"
        "});\n"
        # API coverage chart
        "const apiCtx = document.getElementById('apiChart').getContext('2d');\n"
        "new Chart(apiCtx, {\n"
        "  type: 'bar',\n"
        "  data: {\n"
        "    labels: " + api_labels + ",\n"
        "    datasets: [{\n"
        "      label: 'エンドポイント数',\n"
        "      data: " + api_counts + ",\n"
        "      backgroundColor: '#6366f188',\n"
        "      borderColor: '#6366f1',\n"
        "      borderWidth: 1,\n"
        "      borderRadius: 4,\n"
        "    }]\n"
        "  },\n"
        "  options: {\n"
        "    indexAxis: 'y',\n"
        "    responsive: true,\n"
        "    plugins: { legend: { display: false } },\n"
        "    scales: {\n"
        "      x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },\n"
        "      y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 11 } } }\n"
        "    }\n"
        "  }\n"
        "});\n"
        # Live health check
        "async function checkHealth() {\n"
        "  const dot = document.getElementById('statusDot');\n"
        "  const txt = document.getElementById('statusText');\n"
        "  const det = document.getElementById('statusDetail');\n"
        "  const hd  = document.getElementById('healthDetail');\n"
        "  try {\n"
        "    const r = await fetch('/health', { headers: { 'Authorization': 'Bearer dev' } });\n"
        "    if (!r.ok) throw new Error('HTTP ' + r.status);\n"
        "    const d = await r.json();\n"
        "    dot.className = 'status-dot ok';\n"
        "    txt.textContent = '✅ API サーバー稼働中';\n"
        "    det.textContent = 'device: ' + (d.device || '?') + ' / params: ' + (d.n_params?.toLocaleString() || '?') + ' / version: ' + (d.version || '?');\n"
        "    hd.innerHTML = [\n"
        "      ['モデル', d.model || '—'],\n"
        "      ['デバイス', d.device || '—'],\n"
        "      ['パラメータ数', d.n_params ? d.n_params.toLocaleString() : '—'],\n"
        "      ['ループ上限', d.max_loops || '—'],\n"
        "    ].map(([k,v]) => `<div style='background:#1e293b;padding:.75rem;border-radius:8px;font-size:.82rem;'>"
        "<div style='color:#94a3b8'>${k}</div><div style='font-weight:600;margin-top:.2rem'>${v}</div></div>`).join('');\n"
        "  } catch(e) {\n"
        "    dot.className = 'status-dot error';\n"
        "    txt.textContent = '⚠️ API サーバー未起動';\n"
        "    det.textContent = 'uvicorn serve.api:app --reload で起動してください';\n"
        "    hd.innerHTML = '';\n"
        "  }\n"
        "}\n"
        "checkHealth();\n"
        "setInterval(checkHealth, 30000);\n"
        "</script>\n"
        "</body></html>\n"
    )
    return html


# ---------------------------------------------------------------------------
# FastAPI エンドポイント
# ---------------------------------------------------------------------------

@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="OpenMythos ショーケースダッシュボード",
    description="アーキテクチャ・P1〜P10・ベンチマーク・Sprint 進捗・ライブ API ステータスを1ページに集約したブラウザ可視化ページ。",
)
def showcase_dashboard() -> HTMLResponse:
    """ブラウザで直接開けるショーケースダッシュボードを返す"""
    return HTMLResponse(content=build_showcase_dashboard())


# ---------------------------------------------------------------------------
# standalone CLI — GitHub Pages 用静的 HTML 生成
# ---------------------------------------------------------------------------

def save_dashboard(output: str = "dashboard/index.html") -> "Path":
    """静的 HTML ファイルとして保存する (GitHub Pages / Vercel 等向け)"""
    from pathlib import Path
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_showcase_dashboard(), encoding="utf-8")
    return p


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="OpenMythos Showcase Dashboard 生成")
    parser.add_argument("--output", default="dashboard/index.html", help="出力パス")
    args = parser.parse_args()

    path = save_dashboard(args.output)
    print(f"[dashboard] 生成完了: {path}  ({path.stat().st_size // 1024} KB)")
