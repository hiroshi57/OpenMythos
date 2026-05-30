# OpenMythos — Next Tasks
> 作成: 2026-05-30 | 現在バージョン: v0.16.0 (729 PASS / Sprint 1〜13完了)
> 目標: **ClaudeMythosに近づく、デジタルアイデンティティ特化 SEO/LLMO/広告 強化モデル**

---

## ミッション定義

```
OpenMythos ≠ 汎用LLM
OpenMythos = デジタルマーケティング × アイデンティティ領域に深く刺さる専門モデル

強みを尖らせる3軸:
  [1] SEO / LLMO 生成品質   — AIサーチ時代のコンテンツ最適化
  [2] 広告 ROI 予測・最適化  — CTR・CVR・ROAS のインテリジェント推論
  [3] デジタルアイデンティティ — ブランド・ペルソナ・ターゲティング精度
```

---

## Sprint 14: 実行品質の完成 & 外販準備 v0.17.0

> **Why**: Sprint 13までは「機能を作る」フェーズ。Sprint 14は「売れる状態にする」フェーズ。
> テストが通ること ≠ 外販できること。デモ・ドキュメント・再現性が必要。

| task-id | 内容 | 優先度 | DoD |
|---------|------|--------|-----|
| 14.1 | **torch 環境整備** — requirements.txt に `torch>=2.1` を明記、`pip install -e .[dev]` で全テスト PASS | 🔴 必須 | `pytest tests/ -x` が 729+ PASS |
| 14.2 | **Sprint 13 master merge** — `feature/sprint12-react-cache-memory` を master にマージ、v0.16.0 タグ打ち | 🔴 必須 | master = 729 PASS / v0.16.0 |
| 14.3 | **今回のバグ修正コミット** — scatter()/import re/cpa→cpc/llmo/structured 8件を1コミット | 🔴 必須 | git commit + push |
| 14.4 | **デモノートブック作成** — `examples/demo_seo_llmo.ipynb` — SEO生成→LLMOスコア→広告ROI の3ステップデモ | 🟡 重要 | Colabで動作確認済み |
| 14.5 | **README 外販版リライト** — 「何ができるか」「誰向けか」「5分で動かす方法」を英日で記載 | 🟡 重要 | GitHub READMEとして完結 |
| 14.6 | **PyPI v0.17.0 リリース** — CHANGELOG 更新 + `python -m build` + `twine upload` | 🟢 任意 | PyPI で `pip install open-mythos` が通る |

---

## Sprint 15: SEO / LLMO エンジン深化 v0.18.0

> **Why**: 「SEO/LLMO強い」と言うからには、業界標準の指標を全部計算できないと外販で負ける。

| task-id | 内容 | 優先度 | 詳細 |
|---------|------|--------|------|
| 15.1 | **日本語形態素解析統合** — `janome` or `fugashi` で `llmo.py` の単語分割を強化 | 🔴 必須 | 「デジタルマーケティング」が1語として抽出される |
| 15.2 | **キーワード密度の精密計算** — 見出し・本文・メタの重み付け別計算 (title: ×3, h1: ×2, body: ×1) | 🟡 重要 | `LLMOScore` に `weighted_keyword_density` 追加 |
| 15.3 | **LLMO ベンチマーク** — 既存LLM (GPT-4o / Claude Sonnet) vs OpenMythos の LLMO スコア比較表 | 🟡 重要 | `benchmark/llmo_bench.py` + 結果を README に掲載 |
| 15.4 | **SEO A/B テスト機能** — `llmo.compare()` を拡張して複数バリアント一括評価 + 統計的有意差判定 | 🟡 重要 | `LLMOScorer.ab_test(variants: list[str]) -> ABTestResult` |
| 15.5 | **Core Web Vitals シミュレーター** — `structured.py` の LCP/CLS スキーマに対応した疑似スコア算出 | 🟢 任意 | コンテンツ長・画像数から LCP を推定 |

---

## Sprint 16: 広告 ROI / ターゲティング強化 v0.19.0

> **Why**: 広告領域こそデジタルアイデンティティの本丸。ROI予測が「使える精度」になれば差別化になる。

| task-id | 内容 | 優先度 | 詳細 |
|---------|------|--------|------|
| 16.1 | **Quality Score 予測モデル** — `tools_marketing.py` に Google Ads QS (1-10) 推定ロジック追加 | 🔴 必須 | 入力: 広告文 + LP + キーワード → QS 予測 |
| 16.2 | **広告コピー A/B 生成** — `SwarmOrchestrator` の `broadcast` 戦略で複数バリアントを並列生成 | 🟡 重要 | 1クエリで5バリアント + LLMOスコア付きで返す |
| 16.3 | **ROAS シミュレーター** — 過去 CPCデータ・CVR・客単価を入力として ROAS 範囲を予測 | 🟡 重要 | `calculate_roi()` の拡張。信頼区間付き |
| 16.4 | **ペルソナ×広告マッチング** — RAGPipeline でペルソナドキュメントを検索し広告文の関連度スコア | 🟢 任意 | `rag.py` + `tools_marketing.py` の連携 |
| 16.5 | **Impression Share 予測** — `AD_PERFORMANCE_SCHEMA` の `impression_share` フィールドに実装を追加 | 🟢 任意 | 競合スコア・入札・QS から推定 |

---

## Sprint 17: エージェント実用化 & 外販 API 完成 v0.20.0

> **Why**: SwarmOrchestrator + ReActAgent が「実際のマーケ業務」を自律実行できれば、競合にない武器になる。

| task-id | 内容 | 優先度 | 詳細 |
|---------|------|--------|------|
| 17.1 | **SEOエージェントワークフロー** — 「キーワード調査→構成作成→コンテンツ生成→LLMOスコア→改善」を ReActAgent で自動化 | 🔴 必須 | `examples/seo_agent_workflow.py` |
| 17.2 | **マルチエージェント広告プランナー** — SwarmOrchestrator `pipeline` で「ペルソナ分析→コピー生成→ROI試算」を直列実行 | 🔴 必須 | 3エージェント pipeline のデモ |
| 17.3 | **APIキー認証** — `serve/api.py` に Bearer Token 認証を追加 (外販時の必須要件) | 🔴 必須 | `Authorization: Bearer <key>` ヘッダー検証 |
| 17.4 | **レート制限** — エンドポイント別 RPM/TPM 制限 + 429 レスポンス | 🟡 重要 | `slowapi` or カスタムミドルウェア |
| 17.5 | **Docker 本番イメージ** — `docker-compose.yml` を production 対応に。Gunicorn + uvicorn workers | 🟡 重要 | `docker build && docker run` で `/health` が返る |
| 17.6 | **OpenAPI ドキュメント整備** — 全エンドポイントに `summary` / `description` / `example` を追加 | 🟢 任意 | `/docs` (Swagger UI) が外販説明資料になる水準 |

---

## 外販チェックリスト（出荷判定基準）

```
[ ] テスト: pytest 全 PASS（torch 環境で確認）
[ ] デモ: examples/demo_seo_llmo.ipynb が Colab でゼロから動く
[ ] ドキュメント: README に「5分で動かす」手順がある
[ ] API: /health, /v1/seo/score, /v1/seo/generate が実際に応答する
[ ] 認証: Bearer Token なしのリクエストを 401 で弾く
[ ] パッケージ: pip install open-mythos でインストールできる
[ ] 差別化: 「なぜ GPT-4o / Claude より OpenMythos がいいか」を1ページで説明できる
```

---

## 技術的負債（後回しにしてよいが忘れないリスト）

| 項目 | 場所 | 内容 |
|------|------|------|
| 文字単位トークナイザ | `generate_seo.py`, `thinking.py`, `react.py` | 本物の tokenizer (SentencePiece / tiktoken) に替えるべき |
| HuggingFace tokenizer 統合 | `serve/api.py` state.tokenizer | 現在スタブ。実 tokenizer 接続で精度向上 |
| API リトライ機構 | `serve/api.py` | exponential backoff (max 3回) 未実装 |
| Agents 並列 fan-out | `serve/api.py` | Promise.all 相当の並列推論 未実装 |
| `speculative_decode()` の eps | `main.py:1540` | `1e-10` → `1e-8` に変更推奨（数値安定性） |

---

## バージョンロードマップ

| バージョン | Sprint | キーワード |
|-----------|--------|-----------|
| v0.16.0 | Sprint 12-13 完了 | ReAct / Swarm / MoD — **現在地** |
| v0.17.0 | Sprint 14 | バグ修正完了 + デモ + README |
| v0.18.0 | Sprint 15 | SEO/LLMO エンジン深化 |
| v0.19.0 | Sprint 16 | 広告 ROI 強化 |
| v0.20.0 | Sprint 17 | エージェント実用化 + 外販 API 完成 🚀 |
