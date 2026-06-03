#!/usr/bin/env python3
"""
benchmark/project_dashboard.py — OpenMythos プロジェクト全体ダッシュボード

Sprint 1〜37 の全進捗 (テスト数・バージョン・テーマ) を Chart.js で可視化する
self-contained HTML ダッシュボードを生成する。

利用可能な場合は benchmark/results/*.json から Growing AI ベンチマーク結果も取り込む。

使い方:
    # ダッシュボードを生成して保存
    python benchmark/project_dashboard.py --output benchmark/results/project_dashboard.html

    # ベンチマーク JSON も取り込む
    python benchmark/project_dashboard.py \\
        --output dashboard.html \\
        --bench-dir benchmark/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# リポジトリルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Sprint 全データ (Plans.md から抽出 — 静的マスター)
# ---------------------------------------------------------------------------

SPRINT_DATA: List[Dict] = [
    {"sprint": 1,  "theme": "HyperloopMythos / Inference Engine",          "module": "main.py hyperloop.py",              "tests": 227,  "version": ""},
    {"sprint": 2,  "theme": "Inference 高度化 (beam/quant/KV cache)",        "module": "main.py 拡張",                       "tests": 250,  "version": ""},
    {"sprint": 3,  "theme": "ドキュメント & エコシステム",                    "module": "docs/scripts",                      "tests": 257,  "version": "v0.6"},
    {"sprint": 4,  "theme": "Training 基盤 / MoDa / variants",               "module": "moda.py variants.py",               "tests": 257,  "version": ""},
    {"sprint": 5,  "theme": "Training 品質 / LoRA / HF Hub / CLI",           "module": "cli.py logger_utils.py",            "tests": 284,  "version": "v0.7"},
    {"sprint": 6,  "theme": "推論最適化 / TrainLogger / Agents",              "module": "agents.py",                         "tests": 380,  "version": ""},
    {"sprint": 7,  "theme": "Serving / データパイプライン",                   "module": "serve/api.py",                      "tests": 420,  "version": "v0.12"},
    {"sprint": 8,  "theme": "Fine-tuning / /v1/chat / SLA",                  "module": "scripts/finetune.py",               "tests": 468,  "version": ""},
    {"sprint": 9,  "theme": "Marketing eval / A/B / batch API",              "module": "serve/ab_router.py",                "tests": 508,  "version": "v0.13"},
    {"sprint": 10, "theme": "LLMO / Extended Thinking / Structured Output",  "module": "llmo.py thinking.py structured.py", "tests": 560,  "version": "v0.14"},
    {"sprint": 11, "theme": "Tool Use / Long Context / RAG",                 "module": "tools.py rag.py rope_extension.py", "tests": 664,  "version": "v0.15"},
    {"sprint": 12, "theme": "ReAct / Prefix Cache / Conversation Memory",    "module": "react.py prefix_cache.py",          "tests": 729,  "version": "v0.16"},
    {"sprint": 13, "theme": "Mixture-of-Depths / SwarmOrchestrator",         "module": "mod.py swarm.py",                   "tests": 836,  "version": ""},
    {"sprint": 14, "theme": "GPU pretrain / Benchmark / GCP deploy",         "module": "scripts/pretrain.py benchmark/",    "tests": 856,  "version": "v0.17"},
    {"sprint": 15, "theme": "日本語形態素解析 / A/B テスト / ドリフト検出",   "module": "llmo.py 拡張",                       "tests": 888,  "version": "v0.18"},
    {"sprint": 16, "theme": "SEO パイプライン / セキュリティ",                "module": "seo_pipeline.py security.py",       "tests": 958,  "version": "v0.19"},
    {"sprint": 17, "theme": "API 認証 / レート制限 / Docker 本番化",          "module": "serve/auth.py serve/Dockerfile",    "tests": 998,  "version": "v0.20"},
    {"sprint": 18, "theme": "ファインチューニング実証 / ROAS / ペルソナ",     "module": "tools_marketing.py 拡張",           "tests": 1037, "version": "v0.21"},
    {"sprint": 19, "theme": "LLMO 強化 — score_with_query / LLMOOptimizer",  "module": "llmo.py 拡張",                       "tests": 1079, "version": "v0.22"},
    {"sprint": 20, "theme": "P1: 討議型集合知 — DebateOrchestrator",         "module": "debate.py",                         "tests": 1138, "version": "v0.23", "pattern": "P1"},
    {"sprint": 21, "theme": "P2: KPI 駆動自己改善 — KPIAgent",               "module": "kpi_agent.py",                      "tests": 1204, "version": "v0.24", "pattern": "P2"},
    {"sprint": 22, "theme": "P3: ボトルネック発見・解消 — ProfilerAgent",     "module": "profiler.py",                       "tests": 1265, "version": "v0.25", "pattern": "P3"},
    {"sprint": 23, "theme": "P4: 外部要因適応 — ExternalSignalAgent",        "module": "external_signal.py",                "tests": 1325, "version": "v0.26", "pattern": "P4"},
    {"sprint": 24, "theme": "P5: ミスから学習 — MistakeGuard",               "module": "error_memory.py",                   "tests": 1365, "version": "v0.27", "pattern": "P5"},
    {"sprint": 25, "theme": "P6: 継続的自己蒸留 — SelfDistillLoop",          "module": "self_distill.py",                   "tests": 1408, "version": "v0.28", "pattern": "P6"},
    {"sprint": 26, "theme": "P7: 長期記憶統合 — LongTermMemoryAgent",        "module": "long_term_memory.py",               "tests": 1450, "version": "v0.29", "pattern": "P7"},
    {"sprint": 27, "theme": "P8: アンサンブル品質評価 — EnsembleScorer",     "module": "ensemble_scorer.py",                "tests": 1490, "version": "v0.30", "pattern": "P8"},
    {"sprint": 28, "theme": "P9: 適応型プロンプト進化 — PromptEvolution",    "module": "prompt_evolution.py",               "tests": 1530, "version": "v0.31", "pattern": "P9"},
    {"sprint": 29, "theme": "P10: 自律タスク計画 — TaskPlanner",             "module": "task_planner.py",                   "tests": 1570, "version": "v0.32", "pattern": "P10"},
    {"sprint": 30, "theme": "統合: P1〜P10 GrowingAIOrchestrator",           "module": "growing_ai_orchestrator.py",        "tests": 1617, "version": "v0.33"},
    {"sprint": 31, "theme": "GPU LoRA SFT 統合 — LoraTrainer",               "module": "lora_trainer.py",                   "tests": 1657, "version": "v0.34"},
    {"sprint": 32, "theme": "エラーメモリ永続化 — SQLite backend",           "module": "error_memory.py",                   "tests": 1697, "version": "v0.35"},
    {"sprint": 33, "theme": "LongTermMemory FAISS ANN インデックス",          "module": "long_term_memory.py",               "tests": 1737, "version": "v0.36"},
    {"sprint": 34, "theme": "MistakeGuardMiddleware — 全 API 透過チェック",   "module": "error_memory.py",                   "tests": 1777, "version": "v0.37"},
    {"sprint": 35, "theme": "Growing AI ベンチマーク強化 (P1〜P10 KPI)",      "module": "benchmark/growing_ai_bench.py",     "tests": 1822, "version": "v0.38"},
    {"sprint": 36, "theme": "API ドキュメント整備 + CI ベンチ自動化",         "module": "serve/api.py bench.yml",            "tests": 1861, "version": "v0.39"},
    {"sprint": 37, "theme": "ベンチマーク可視化ダッシュボード + E2E テスト",  "module": "benchmark/report.py",               "tests": 1922, "version": "v0.40"},
]

ERAS = [
    {"label": "Era 1: 基盤構築",          "start": 1,  "end": 9,  "color": "rgba(52,152,219,0.75)"},
    {"label": "Era 2: LLMO / Tool / RAG", "start": 10, "end": 19, "color": "rgba(46,204,113,0.75)"},
    {"label": "Era 3: 育つ AI P1〜P10",   "start": 20, "end": 29, "color": "rgba(155,89,182,0.75)"},
    {"label": "Era 4: 統合・最適化",       "start": 30, "end": 37, "color": "rgba(231,76,60,0.75)"},
]

PATTERNS = [
    {"id": "P1",  "name": "討議型集合知",        "sprint": 20, "api": "/v1/debate/run"},
    {"id": "P2",  "name": "KPI 駆動自己改善",     "sprint": 21, "api": "/v1/kpi/*"},
    {"id": "P3",  "name": "ボトルネック発見・解消","sprint": 22, "api": "/v1/profile/*"},
    {"id": "P4",  "name": "外部要因適応",          "sprint": 23, "api": "/v1/signal/*"},
    {"id": "P5",  "name": "ミスから学習",          "sprint": 24, "api": "/v1/mistakes/*"},
    {"id": "P6",  "name": "継続的自己蒸留",        "sprint": 25, "api": "/v1/distill/*"},
    {"id": "P7",  "name": "長期記憶統合",          "sprint": 26, "api": "/v1/memory/*"},
    {"id": "P8",  "name": "アンサンブル品質評価",  "sprint": 27, "api": "/v1/ensemble/*"},
    {"id": "P9",  "name": "適応型プロンプト進化",  "sprint": 28, "api": "/v1/evolve/*"},
    {"id": "P10", "name": "自律タスク計画",        "sprint": 29, "api": "/v1/plan/*"},
]


# ---------------------------------------------------------------------------
# BenchmarkReport 読み込み
# ---------------------------------------------------------------------------

def _load_bench_results(bench_dir: Optional[str]) -> List[dict]:
    if not bench_dir:
        return []
    results = []
    for p in sorted(Path(bench_dir).glob("bench-*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            results.append({
                "timestamp": d.get("timestamp", "")[:10],
                "n_patterns": d.get("n_patterns", 0),
                "n_success":  d.get("n_success",  0),
                "avg_improvement_pct": d.get("avg_improvement_pct", 0),
                "avg_latency_ms":      d.get("avg_latency_ms",      0),
                "total_latency_ms":    d.get("total_latency_ms",    0),
                "results": d.get("results", []),
            })
        except Exception:
            continue
    return sorted(results, key=lambda x: x["timestamp"])


# ---------------------------------------------------------------------------
# ヘルパー: era色を返す
# ---------------------------------------------------------------------------

_ERA_COLORS = [
    "rgba(52,152,219,0.75)",
    "rgba(46,204,113,0.75)",
    "rgba(155,89,182,0.75)",
    "rgba(231,76,60,0.75)",
]

def _era_color(sprint: int) -> str:
    for i, era in enumerate(ERAS):
        if era["start"] <= sprint <= era["end"]:
            return _ERA_COLORS[i]
    return "rgba(127,140,141,0.7)"


# ---------------------------------------------------------------------------
# KPI セクション HTML / JS を文字列で生成 (f-string ネストを回避)
# ---------------------------------------------------------------------------

def _build_bench_section(bench_results: List[dict]) -> str:
    """BenchmarkReport が存在する場合の KPI セクション HTML+JS を返す。"""
    if not bench_results:
        return ""

    latest = bench_results[-1]
    b_dates   = [b["timestamp"] for b in bench_results]
    b_avgs    = [b["avg_improvement_pct"] for b in bench_results]
    b_success = [round(b["n_success"] / max(b["n_patterns"], 1) * 100, 1) for b in bench_results]
    b_latency = [b["avg_latency_ms"] for b in bench_results]

    pat_ids       = [r["pattern_id"].upper() for r in latest["results"]]
    pat_deltas    = [r["improvement_pct"]    for r in latest["results"]]
    pat_baselines = [r["baseline_score"]     for r in latest["results"]]
    pat_finals    = [r["final_score"]        for r in latest["results"]]
    pat_colors    = [
        "rgba(40,167,69,0.75)" if r["success"] else "rgba(215,58,73,0.75)"
        for r in latest["results"]
    ]

    avg_color = "#28a745" if b_avgs[-1] >= 0 else "#d73a49"

    # トレンドチャート (2件以上の場合のみ)
    trend_canvas = ""
    trend_script = ""
    if len(bench_results) >= 2:
        trend_canvas = (
            "\n    <div class='chart-card wide'>"
            "\n      <canvas id='kpiTrendChart'></canvas>"
            "\n    </div>"
        )
        trend_script = (
            "\n  // KPI トレンドライン\n"
            "  (function() {\n"
            "    var ctx = document.getElementById('kpiTrendChart').getContext('2d');\n"
            "    new Chart(ctx, {\n"
            "      type: 'line',\n"
            "      data: {\n"
            "        labels: " + json.dumps(b_dates) + ",\n"
            "        datasets: [\n"
            "          {\n"
            "            label: '平均 KPI 改善率 (Δ%)',\n"
            "            data: " + json.dumps(b_avgs) + ",\n"
            "            borderColor: 'rgba(52,152,219,1)',\n"
            "            backgroundColor: 'rgba(52,152,219,0.1)',\n"
            "            tension: 0.35, fill: true, yAxisID: 'y',\n"
            "          },\n"
            "          {\n"
            "            label: '成功率 (%)',\n"
            "            data: " + json.dumps(b_success) + ",\n"
            "            borderColor: 'rgba(46,204,113,1)',\n"
            "            borderDash: [5,5], tension: 0.35,\n"
            "            fill: false, yAxisID: 'y2',\n"
            "          },\n"
            "        ],\n"
            "      },\n"
            "      options: {\n"
            "        responsive: true,\n"
            "        interaction: { mode: 'index', intersect: false },\n"
            "        plugins: { title: { display: true,"
            " text: 'KPI 改善率・成功率 週次トレンド' } },\n"
            "        scales: {\n"
            "          y:  { title: { display: true, text: '改善率 (%)' } },\n"
            "          y2: { position: 'right', min: 0, max: 100,\n"
            "                title: { display: true, text: '成功率 (%)' },\n"
            "                grid: { drawOnChartArea: false } },\n"
            "        },\n"
            "      },\n"
            "    });\n"
            "  })();\n"
        )

    parts = [
        "  <!-- ========== KPI ベンチマーク (Sprint 35+) ========== -->",
        "  <h2 class='section-title'>⚡ Growing AI KPI ベンチマーク (Sprint 35+)</h2>",
        "  <div class='kpi-summary-bar'>",
        "    <div class='kpi-badge'>",
        "      <span class='kpi-val'>" + str(latest["n_success"]) + "/" + str(latest["n_patterns"]) + "</span>",
        "      <span class='kpi-lbl'>最新成功率</span>",
        "    </div>",
        "    <div class='kpi-badge'>",
        "      <span class='kpi-val' style='color:" + avg_color + "'>"
            + ("{:+.1f}%".format(b_avgs[-1])) + "</span>",
        "      <span class='kpi-lbl'>平均 KPI 改善率</span>",
        "    </div>",
        "    <div class='kpi-badge'>",
        "      <span class='kpi-val'>" + "{:.1f} ms".format(b_latency[-1]) + "</span>",
        "      <span class='kpi-lbl'>平均レイテンシ</span>",
        "    </div>",
        "    <div class='kpi-badge'>",
        "      <span class='kpi-val'>" + latest["timestamp"] + "</span>",
        "      <span class='kpi-lbl'>最終実行日</span>",
        "    </div>",
        "  </div>",
        "  <div class='charts'>",
        "    <div class='chart-card'><canvas id='kpiDeltaChart'></canvas></div>",
        "    <div class='chart-card'><canvas id='kpiCompareChart'></canvas></div>",
        trend_canvas,
        "  </div>",
        "  <script>",
        "  // KPI 改善率 棒グラフ",
        "  (function() {",
        "    var ctx = document.getElementById('kpiDeltaChart').getContext('2d');",
        "    new Chart(ctx, {",
        "      type: 'bar',",
        "      data: {",
        "        labels: " + json.dumps(pat_ids) + ",",
        "        datasets: [{",
        "          label: 'KPI 改善率 (Δ%)',",
        "          data: " + json.dumps(pat_deltas) + ",",
        "          backgroundColor: " + json.dumps(pat_colors) + ",",
        "          borderRadius: 5,",
        "        }],",
        "      },",
        "      options: {",
        "        responsive: true,",
        "        plugins: {",
        "          legend: { display: false },",
        "          title: { display: true, text: 'パターン別 KPI 改善率 Δ% (最新実行)' },",
        "        },",
        "        scales: { y: { title: { display: true, text: 'Δ%' } } },",
        "      },",
        "    });",
        "  })();",
        "",
        "  // Baseline vs Final",
        "  (function() {",
        "    var ctx = document.getElementById('kpiCompareChart').getContext('2d');",
        "    new Chart(ctx, {",
        "      type: 'bar',",
        "      data: {",
        "        labels: " + json.dumps(pat_ids) + ",",
        "        datasets: [",
        "          { label: 'Baseline', data: " + json.dumps(pat_baselines) + ",",
        "            backgroundColor: 'rgba(88,96,105,0.55)', borderRadius: 4 },",
        "          { label: 'Final', data: " + json.dumps(pat_finals) + ",",
        "            backgroundColor: 'rgba(40,167,69,0.7)', borderRadius: 4 },",
        "        ],",
        "      },",
        "      options: {",
        "        responsive: true,",
        "        plugins: { title: { display: true, text: 'Baseline vs Final スコア' } },",
        "        scales: { y: { min: 0, max: 1 } },",
        "      },",
        "    });",
        "  })();",
        trend_script,
        "  </script>",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# メインダッシュボード生成
# ---------------------------------------------------------------------------

def build_dashboard(bench_dir: Optional[str] = None) -> str:
    """
    Sprint 1〜37 プロジェクト全体ダッシュボード HTML を生成する。

    Parameters
    ----------
    bench_dir : str, optional
        BenchmarkReport JSON ディレクトリ (例: benchmark/results)。
        指定時は Sprint 35+ の KPI データも取り込む。

    Returns
    -------
    str
        self-contained HTML 文字列 (Chart.js CDN)
    """
    bench_results = _load_bench_results(bench_dir)

    sprint_nums  = [d["sprint"] for d in SPRINT_DATA]
    test_counts  = [d["tests"]  for d in SPRINT_DATA]
    test_deltas  = [
        test_counts[i] - (test_counts[i - 1] if i > 0 else 0)
        for i in range(len(test_counts))
    ]
    versions     = [d.get("version", "") for d in SPRINT_DATA]
    bar_colors   = [_era_color(d["sprint"]) for d in SPRINT_DATA]
    themes       = [d["theme"] for d in SPRINT_DATA]

    total_tests  = SPRINT_DATA[-1]["tests"]
    latest_ver   = SPRINT_DATA[-1]["version"]
    n_releases   = len([d for d in SPRINT_DATA if d.get("version")])
    n_sprints    = len(SPRINT_DATA)

    # release データ
    rel_data = [
        {"sprint": d["sprint"], "version": d["version"], "tests": d["tests"]}
        for d in SPRINT_DATA if d.get("version")
    ]

    # P1〜P10 カード
    pattern_cards = ""
    for p in PATTERNS:
        pattern_cards += (
            "\n      <div class='pat-card'>"
            "\n        <div class='pat-id'>" + p["id"] + "</div>"
            "\n        <div class='pat-name'>" + p["name"] + "</div>"
            "\n        <div class='pat-meta'>Sprint " + str(p["sprint"]) + " · <code>" + p["api"] + "</code></div>"
            "\n      </div>"
        )

    # Sprint テーブル行
    sprint_rows = ""
    for d in SPRINT_DATA:
        ver = d.get("version", "")
        pat = d.get("pattern", "")
        ver_cell = ('<span class="ver-badge">' + ver + '</span>') if ver else ""
        pat_span = (' <span class="pat-badge">' + pat + '</span>') if pat else ""
        sprint_rows += (
            "<tr>"
            "<td style='text-align:center'><strong>" + str(d["sprint"]) + "</strong></td>"
            "<td>" + d["theme"] + pat_span + "</td>"
            "<td><code>" + d["module"][:45] + "</code></td>"
            "<td style='text-align:right'>" + "{:,}".format(d["tests"]) + "</td>"
            "<td style='text-align:center'>" + ver_cell + "</td>"
            "</tr>"
        )

    # KPI ベンチマークセクション
    bench_section = _build_bench_section(bench_results)

    # ── 組み立て ────────────────────────────────────────────────────

    js_sprint_themes  = json.dumps(themes,      ensure_ascii=False)
    js_sprint_nums    = json.dumps(sprint_nums)
    js_test_counts    = json.dumps(test_counts)
    js_test_deltas    = json.dumps(test_deltas)
    js_versions       = json.dumps(versions,    ensure_ascii=False)
    js_bar_colors     = json.dumps(bar_colors)
    js_rel_data       = json.dumps(rel_data,    ensure_ascii=False)

    html = (
        "<!DOCTYPE html>\n"
        "<html lang='ja'>\n"
        "<head>\n"
        "  <meta charset='UTF-8'>\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
        "  <title>OpenMythos — Project Dashboard (Sprint 1〒37)</title>\n"
        "  <script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js'></script>\n"
        "  <style>\n"
        "    *, *::before, *::after { box-sizing: border-box; }\n"
        "    body {\n"
        "      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;\n"
        "      background: #0d1117; color: #c9d1d9; margin: 0; padding: 0;\n"
        "    }\n"
        "    header {\n"
        "      background: linear-gradient(135deg, #161b22 0%, #1f2937 60%, #0f3460 100%);\n"
        "      padding: 1.5em 2em; border-bottom: 1px solid #30363d;\n"
        "    }\n"
        "    header h1 { margin: 0; font-size: 1.7em; color: #f0f6fc; }\n"
        "    header p  { margin: 0.4em 0 0; color: #8b949e; font-size: 0.9em; }\n"
        "    .container { max-width: 1280px; margin: 0 auto; padding: 1.5em 1.5em 3em; }\n"
        "    .summary-cards {\n"
        "      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));\n"
        "      gap: 1em; margin: 1.2em 0;\n"
        "    }\n"
        "    .card {\n"
        "      background: #161b22; border: 1px solid #30363d; border-radius: 10px;\n"
        "      padding: 1.1em; text-align: center;\n"
        "    }\n"
        "    .card .val { font-size: 2.2em; font-weight: 700; color: #58a6ff; }\n"
        "    .card .lbl { font-size: 0.78em; color: #8b949e; text-transform: uppercase;\n"
        "                 letter-spacing: 0.07em; margin-top: 0.3em; }\n"
        "    .section-title {\n"
        "      font-size: 1.05em; color: #8b949e; margin: 1.8em 0 0.8em;\n"
        "      border-left: 4px solid #58a6ff; padding-left: 0.6em; font-weight: 600;\n"
        "    }\n"
        "    .charts {\n"
        "      display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));\n"
        "      gap: 1.2em;\n"
        "    }\n"
        "    .chart-card {\n"
        "      background: #161b22; border: 1px solid #30363d; border-radius: 10px;\n"
        "      padding: 1.2em;\n"
        "    }\n"
        "    .chart-card.wide { grid-column: 1 / -1; }\n"
        "    .era-legend { display: flex; flex-wrap: wrap; gap: 0.6em; margin-bottom: 0.8em; }\n"
        "    .era-chip {\n"
        "      display: flex; align-items: center; gap: 0.4em;\n"
        "      background: #161b22; border: 1px solid #30363d;\n"
        "      border-radius: 20px; padding: 0.25em 0.75em; font-size: 0.82em;\n"
        "    }\n"
        "    .era-dot { width: 10px; height: 10px; border-radius: 50%; }\n"
        "    .kpi-summary-bar { display: flex; flex-wrap: wrap; gap: 1em; margin: 0.8em 0; }\n"
        "    .kpi-badge {\n"
        "      background: #161b22; border: 1px solid #30363d; border-radius: 8px;\n"
        "      padding: 0.7em 1.2em; text-align: center; min-width: 130px;\n"
        "    }\n"
        "    .kpi-val { display: block; font-size: 1.5em; font-weight: 700; color: #58a6ff; }\n"
        "    .kpi-lbl { display: block; font-size: 0.75em; color: #8b949e; margin-top: 0.2em; }\n"
        "    .pattern-grid {\n"
        "      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));\n"
        "      gap: 0.8em; margin: 0.8em 0;\n"
        "    }\n"
        "    .pat-card {\n"
        "      background: linear-gradient(135deg, #161b22, #1f2937);\n"
        "      border: 1px solid #30363d; border-radius: 8px; padding: 0.9em;\n"
        "    }\n"
        "    .pat-id   { font-size: 1.2em; font-weight: 700; color: #a371f7; }\n"
        "    .pat-name { font-size: 0.9em; color: #e6edf3; margin: 0.2em 0; }\n"
        "    .pat-meta { font-size: 0.75em; color: #8b949e; }\n"
        "    table { border-collapse: collapse; width: 100%; font-size: 0.85em; }\n"
        "    th {\n"
        "      background: #21262d; color: #8b949e; padding: 8px 10px;\n"
        "      text-align: left; border-bottom: 1px solid #30363d;\n"
        "    }\n"
        "    td { padding: 6px 10px; border-bottom: 1px solid #21262d; }\n"
        "    tr:hover td { background: #161b22; }\n"
        "    code { background: #21262d; padding: 1px 4px; border-radius: 3px;\n"
        "           font-size: 0.9em; color: #79c0ff; }\n"
        "    .ver-badge {\n"
        "      background: #0f3460; color: #58a6ff; border-radius: 4px;\n"
        "      padding: 1px 6px; font-size: 0.8em; font-weight: 600;\n"
        "    }\n"
        "    .pat-badge {\n"
        "      background: #2d1b69; color: #a371f7; border-radius: 4px;\n"
        "      padding: 1px 5px; font-size: 0.75em; margin-left: 4px;\n"
        "    }\n"
        "    footer {\n"
        "      text-align: center; color: #484f58; font-size: 0.8em;\n"
        "      margin: 2em 0 0.5em; padding-top: 1em;\n"
        "      border-top: 1px solid #21262d;\n"
        "    }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "  <h1>\U0001f680 OpenMythos — Project Dashboard</h1>\n"
        "  <p>Sprint 1〒37 全進捗 · 累計 " + "{:,}".format(total_tests) + " tests · " + latest_ver + " · 2026-06-03</p>\n"
        "</header>\n"
        "\n"
        "<div class='container'>\n"
        "\n"
        "  <!-- サマリー카ード -->\n"
        "  <div class='summary-cards'>\n"
        "    <div class='card'><div class='val'>" + str(n_sprints) + "</div><div class='lbl'>完了 Sprint</div></div>\n"
        "    <div class='card'><div class='val'>" + "{:,}".format(total_tests) + "</div><div class='lbl'>累計テスト数</div></div>\n"
        "    <div class='card'><div class='val'>" + latest_ver + "</div><div class='lbl'>最新バージョン</div></div>\n"
        "    <div class='card'><div class='val'>10</div><div class='lbl'>育つAI パターン</div></div>\n"
        "    <div class='card'><div class='val'>" + str(n_releases) + "</div><div class='lbl'>リリース数</div></div>\n"
        "  </div>\n"
        "\n"
        "  <h2 class='section-title'>\U0001f4c8 テスト数成長曲線 (Sprint 1〒37)</h2>\n"
        "\n"
        "  <div class='era-legend'>\n"
        "    <div class='era-chip'><div class='era-dot' style='background:rgba(52,152,219,0.9)'></div>Era 1: 基盤構築 (1-9)</div>\n"
        "    <div class='era-chip'><div class='era-dot' style='background:rgba(46,204,113,0.9)'></div>Era 2: LLMO / Tool / RAG (10-19)</div>\n"
        "    <div class='era-chip'><div class='era-dot' style='background:rgba(155,89,182,0.9)'></div>Era 3: 育つ AI P1〒10 (20-29)</div>\n"
        "    <div class='era-chip'><div class='era-dot' style='background:rgba(231,76,60,0.9)'></div>Era 4: 統合・最適化 (30-37)</div>\n"
        "  </div>\n"
        "\n"
        "  <div class='charts'>\n"
        "    <div class='chart-card wide'><canvas id='growthChart'></canvas></div>\n"
        "    <div class='chart-card'><canvas id='deltaChart'></canvas></div>\n"
        "    <div class='chart-card'><canvas id='releaseChart'></canvas></div>\n"
        "  </div>\n"
        "\n"
        "  <h2 class='section-title'>\U0001f9e0 育つ AI 10 パターン (P1〒10)</h2>\n"
        "  <div class='pattern-grid'>" + pattern_cards + "\n  </div>\n"
        "\n"
        + bench_section +
        "\n"
        "  <h2 class='section-title'>\U0001f4cb Sprint 全サマリー (1〒37)</h2>\n"
        "  <div class='chart-card wide' style='overflow-x:auto'>\n"
        "    <table>\n"
        "      <thead><tr>\n"
        "        <th style='text-align:center;width:60px'>Sprint</th>\n"
        "        <th>テーマ</th>\n"
        "        <th>コアモジュール</th>\n"
        "        <th style='text-align:right'>累計テスト</th>\n"
        "        <th style='text-align:center'>Ver</th>\n"
        "      </tr></thead>\n"
        "      <tbody>" + sprint_rows + "</tbody>\n"
        "    </table>\n"
        "  </div>\n"
        "\n"
        "</div>\n"
        "\n"
        "<footer>Generated by OpenMythos <code>benchmark/project_dashboard.py</code> — 2026-06-03</footer>\n"
        "\n"
        "<script>\n"
        "Chart.defaults.color = '#8b949e';\n"
        "Chart.defaults.borderColor = '#30363d';\n"
        "\n"
        "// テスト数成長曲線\n"
        "(function() {\n"
        "  var ctx     = document.getElementById('growthChart').getContext('2d');\n"
        "  var labels  = " + js_sprint_nums + ";\n"
        "  var data    = " + js_test_counts + ";\n"
        "  var colors  = " + js_bar_colors  + ";\n"
        "  var themes  = " + js_sprint_themes + ";\n"
        "  var vers    = " + js_versions + ";\n"
        "\n"
        "  var eraPlugin = {\n"
        "    id: 'eraBackground',\n"
        "    beforeDraw: function(chart) {\n"
        "      var eras = [\n"
        "        { start: 1, end: 9,  color: 'rgba(52,152,219,0.06)' },\n"
        "        { start: 10, end: 19, color: 'rgba(46,204,113,0.06)' },\n"
        "        { start: 20, end: 29, color: 'rgba(155,89,182,0.06)' },\n"
        "        { start: 30, end: 37, color: 'rgba(231,76,60,0.06)'  },\n"
        "      ];\n"
        "      var xA = chart.scales.x, yA = chart.scales.y, c = chart.ctx;\n"
        "      eras.forEach(function(e) {\n"
        "        var x1 = xA.getPixelForValue(e.start - 1 - 0.5);\n"
        "        var x2 = xA.getPixelForValue(e.end   - 1 + 0.5);\n"
        "        c.fillStyle = e.color;\n"
        "        c.fillRect(x1, yA.top, x2 - x1, yA.bottom - yA.top);\n"
        "      });\n"
        "    },\n"
        "  };\n"
        "\n"
        "  new Chart(ctx, {\n"
        "    type: 'line',\n"
        "    plugins: [eraPlugin],\n"
        "    data: {\n"
        "      labels: labels.map(function(n) { return 'S' + n; }),\n"
        "      datasets: [{\n"
        "        label: '累計テスト数',\n"
        "        data: data,\n"
        "        borderColor: '#58a6ff',\n"
        "        backgroundColor: 'rgba(88,166,255,0.12)',\n"
        "        tension: 0.35, fill: true,\n"
        "        pointBackgroundColor: colors,\n"
        "        pointRadius: 4, pointHoverRadius: 7,\n"
        "      }],\n"
        "    },\n"
        "    options: {\n"
        "      responsive: true,\n"
        "      plugins: {\n"
        "        legend: { display: false },\n"
        "        title: { display: true, text: '累計テスト数の成長 (Sprint 1〒37)', color: '#c9d1d9' },\n"
        "        tooltip: {\n"
        "          callbacks: {\n"
        "            title: function(items) {\n"
        "              var i = items[0].dataIndex;\n"
        "              return 'Sprint ' + labels[i] + ': ' + themes[i];\n"
        "            },\n"
        "            afterLabel: function(item) {\n"
        "              var v = vers[item.dataIndex];\n"
        "              return v ? 'Version: ' + v : '';\n"
        "            },\n"
        "          }\n"
        "        },\n"
        "      },\n"
        "      scales: {\n"
        "        x: { grid: { color: 'rgba(48,54,61,0.6)' } },\n"
        "        y: { title: { display: true, text: '累計テスト数', color: '#8b949e' },\n"
        "             grid: { color: 'rgba(48,54,61,0.6)' } },\n"
        "      },\n"
        "    },\n"
        "  });\n"
        "})();\n"
        "\n"
        "// Sprint 別テスト追加数\n"
        "(function() {\n"
        "  var ctx    = document.getElementById('deltaChart').getContext('2d');\n"
        "  var deltas = " + js_test_deltas + ";\n"
        "  var colors = " + js_bar_colors  + ";\n"
        "  var labels = " + js_sprint_nums + ";\n"
        "  new Chart(ctx, {\n"
        "    type: 'bar',\n"
        "    data: {\n"
        "      labels: labels.map(function(n) { return 'S' + n; }),\n"
        "      datasets: [{ label: 'テスト追加数', data: deltas,\n"
        "                   backgroundColor: colors, borderRadius: 3 }],\n"
        "    },\n"
        "    options: {\n"
        "      responsive: true,\n"
        "      plugins: { legend: { display: false },\n"
        "        title: { display: true, text: 'Sprint 別 テスト追加数', color: '#c9d1d9' } },\n"
        "      scales: {\n"
        "        x: { grid: { color: 'rgba(48,54,61,0.6)' } },\n"
        "        y: { title: { display: true, text: '追加テスト数', color: '#8b949e' },\n"
        "             grid: { color: 'rgba(48,54,61,0.6)' } },\n"
        "      },\n"
        "    },\n"
        "  });\n"
        "})();\n"
        "\n"
        "// バージョンリリース\n"
        "(function() {\n"
        "  var ctx     = document.getElementById('releaseChart').getContext('2d');\n"
        "  var relData = " + js_rel_data + ";\n"
        "  new Chart(ctx, {\n"
        "    type: 'bar',\n"
        "    data: {\n"
        "      labels: relData.map(function(d) { return d.version; }),\n"
        "      datasets: [{\n"
        "        label: '累計テスト数',\n"
        "        data: relData.map(function(d) { return d.tests; }),\n"
        "        backgroundColor: 'rgba(88,166,255,0.7)',\n"
        "        borderRadius: 4,\n"
        "      }],\n"
        "    },\n"
        "    options: {\n"
        "      responsive: true,\n"
        "      plugins: {\n"
        "        legend: { display: false },\n"
        "        title: { display: true, text: 'バージョン別 累計テスト数', color: '#c9d1d9' },\n"
        "        tooltip: { callbacks: {\n"
        "          title: function(items) {\n"
        "            return relData[items[0].dataIndex].version +\n"
        "                   ' (Sprint ' + relData[items[0].dataIndex].sprint + ')';\n"
        "          },\n"
        "        }},\n"
        "      },\n"
        "      scales: {\n"
        "        x: { grid: { color: 'rgba(48,54,61,0.6)' } },\n"
        "        y: { title: { display: true, text: 'テスト数', color: '#8b949e' },\n"
        "             grid: { color: 'rgba(48,54,61,0.6)' } },\n"
        "      },\n"
        "    },\n"
        "  });\n"
        "})();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )
    return html


def save_dashboard(output: str, bench_dir: Optional[str] = None) -> Path:
    """ダッシュボードを HTML ファイルに保存する。"""
    html = build_dashboard(bench_dir=bench_dir)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMythos プロジェクト全体ダッシュボード生成"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="benchmark/results/project_dashboard.html",
        help="出力 HTML ファイルパス",
    )
    parser.add_argument(
        "--bench-dir", type=str, default=None,
        help="BenchmarkReport JSON ディレクトリ (例: benchmark/results)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = save_dashboard(args.output, bench_dir=args.bench_dir)
    html_len = out.stat().st_size
    print(f"Dashboard saved: {out}  ({html_len // 1024} KB)")
    print(f"  Sprints: {len(SPRINT_DATA)}  |  Tests: {SPRINT_DATA[-1]['tests']:,}  |  {SPRINT_DATA[-1]['version']}")


if __name__ == "__main__":
    main()
