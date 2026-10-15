#!/usr/bin/env python3
"""Sprint 37.1 — PatternSelector 精度検証"""
from open_mythos.growing_ai_orchestrator import PatternSelector, PatternType

sel = PatternSelector()
cases = [
    ("このコンテンツの品質を評価してください", [], PatternType.ENSEMBLE),
    ("AとBどちらが良いか議論してください", [], PatternType.DEBATE),
    ("KPIを改善したい。目標を達成するには", [], PatternType.KPI),
    ("処理が遅い。ボトルネックを見つけてほしい", [], PatternType.PROFILER),
    ("市場のトレンドと競合を調査して外部環境を分析", [], PatternType.SIGNAL),
    ("エラーが頻発している。再発防止策を", [], PatternType.MISTAKE),
    ("要約して品質を高める蒸留を実施したい", [], PatternType.DISTILL),
    ("過去の履歴と記憶を参照して回答", [], PatternType.MEMORY),
    ("プロンプトを進化させて最適化", [], PatternType.EVOLVE),
    ("タスクを分解してステップを計画する", [], PatternType.PLAN),
    ("どうすれば改善できるか?", [], PatternType.ENSEMBLE),
    ("失敗が続いている。原因を", [], PatternType.MISTAKE),
    ("SEO改善", ["ensemble", "score"], PatternType.ENSEMBLE),
    ("debate this topic", [], PatternType.DEBATE),
    ("plan the workflow step by step", [], PatternType.PLAN),
    ("slow response latency issue", [], PatternType.PROFILER),
    ("summarize and distill key points", [], PatternType.DISTILL),
    ("recall history and past context", [], PatternType.MEMORY),
    ("genetic prompt evolution", [], PatternType.EVOLVE),
    ("error prevention risk guard", [], PatternType.MISTAKE),
]
result = sel.selection_rate(cases)
print(f"accuracy  : {result['accuracy']:.1%} ({result['n_correct']}/{result['n_cases']})")
print(f"error_rate: {result['error_rate']:.1%}  (target: < 5%)")
