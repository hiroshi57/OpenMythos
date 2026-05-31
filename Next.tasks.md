# OpenMythos — Next Tasks
> 作成: 2026-05-30 | 最終更新: 2026-05-31 | 現在バージョン: v0.17.0 (860+ PASS / Sprint 1〜14完了)
> 目標: **Claude Opus 4.8 / Mythos の弱点を突き、SEO・LLMO・広告特化で上回る**

---

## 競合分析サマリー（2026-05-31 調査）

### Claude Opus 4.8 の技術的弱点（攻略ポイント）

| 弱点 | 内容 | OpenMythos の対抗策 |
| ---- | ---- | ------------------- |
| コンテキストドリフト | 長文・複合指示で汎用出力に劣化。SEO戦略と執筆を同一会話に混在させると品質落下 | ReActAgent の 2-phase ワークフロー（戦略→執筆を分離）で解決 |
| プロンプトインジェクション耐性後退 | 4.7の2.3% → 4.8で7%に悪化。エージェントパイプラインで脆弱 | 社内ツールに閉じた構成でリスクゼロ、外部注入経路なし |
| 訓練時ゲーミング傾向 | 「採点者に良く見せる」方向に推論最適化。実務精度と乖離 | 実績SEOデータ（順位・CV・エンゲージメント）でFTすることで回避 |
| 日本語コストが実質割高 | トークン消費1〜1.5倍増。API単価×増加量でコスト膨張 | オンプレ運用で API コストゼロ。日本語特化FTで精度も上 |
| 汎用SEOしかできない | 自社コンテンツ評価基準・実績データを知らない | 社内データでFT → 自社タスクでは Opus 4.8 を上回れる |
| マルチモーダル弱い | 画像・動画・OCRは Gemini 3.1 Pro に劣後 | スコープ外。テキスト特化に集中 |
| Simon Willison 評: 「棄権が増えただけ」 | 不確かな質問を「わからない」と回避。正答率は未向上 | ループ深度で多段推論 → 難問を解く設計 |

### Claude Mythos（数週間以内に一般公開予定）に備えた設計

- **現在**: 招待制（Project Glasswing）。サイバーセキュリティで突出した能力を持つため安全対策整備中
- **公開後の対応**: アーキテクチャ解析 → `docs/mythos_analysis.md` 更新 → RDT 実装との差分をベンチマーク
- **差別化継続**: Mythos がどれほど強くても「SEO/LLMO特化+オープン重み+オンプレ」は代替不可

---

## ミッション再定義（v2 — 2026-05-31）

```
OpenMythos の勝ち筋:

  [1] Claude Opus 4.8 の弱点（コンテキストドリフト・ゲーミング・日本語コスト）を
      構造的に回避した設計

  [2] SEO / LLMO / 広告の「実務精度」で Opus 4.8 API を上回る
      汎用ベンチ（MMLU/HumanEval）で戦う必要はない

  [3] オープン重み + オンプレ + 日本語特化 = 機密データを外部送信しない
      これは Anthropic が絶対に提供できない価値

  [4] Mythos 公開後に即座に比較・差分分析できる態勢
      「技術的に最も詳しいオープン実装」という立ち位置を確立
```

---

## Sprint 14: 外販準備 v0.17.0 ✅ 完了（2026-05-31）

| task-id | 内容 | 状態 |
| ------- | ---- | ---- |
| 14.1〜14.5 | torch環境整備 / master merge / バグ修正 / デモ / README | ✅ 全完了 |
| 14-extra | react.py UnboundLocalError 修正 / structured.py bool型バグ修正 / テスト品質強化9件 | ✅ 全完了 |

テスト結果: 860+ PASS / Warning 2件（PyTorch CPU autocast 固有・無害）

---

## Sprint 15: Opus 4.8 対抗 — SEO/LLMO 実務精度強化 v0.18.0

> **Why**: Opus 4.8 の「コンテキストドリフト」「汎用SEO」弱点を突く。
> 2-phase ワークフロー + 日本語形態素解析 + ベンチマーク比較表で「実務で勝つ」を証明する。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 15.1 | **2-phase SEO ワークフロー** — ReActAgent で「戦略フェーズ（キーワード/構成）」と「執筆フェーズ」を分離。コンテキストドリフト回避 | 🔴 必須 | `examples/seo_two_phase_workflow.py` が動作 |
| 15.2 | **日本語形態素解析統合** — `janome` or `fugashi` で `llmo.py` の単語分割強化。「デジタルマーケティング」が1語として抽出 | 🔴 必須 | 日本語テキストの entity_density が向上 |
| 15.3 | **Opus 4.8 対抗ベンチマーク** — `benchmark/llmo_bench.py` で Opus 4.8 API vs OpenMythos のLLMOスコアを比較。「実務精度で上回る」を数値で証明 | 🔴 必須 | 比較表を README に掲載 |
| 15.4 | **キーワード密度精密計算** — title:×3, h1:×2, body:×1 の重み付け計算。`LLMOScore.weighted_keyword_density` 追加 | 🟡 重要 | 既存テスト PASS + 新テスト追加 |
| 15.5 | **SEO A/B テスト** — `LLMOScorer.ab_test(variants: list[str]) -> ABTestResult`。統計的有意差判定付き | 🟡 重要 | `test_sprint15.py` で検証 |
| 15.6 | **コンテキストドリフト検出** — 会話ターン数 × プロンプト多様性から drift リスクスコアを算出 | 🟡 重要 | `ConversationMemory` に `drift_score` プロパティ追加 |
| 15.7 | **Core Web Vitals シミュレーター** — `structured.py` の LCP/CLS スキーマに疑似スコア算出実装 | 🟢 任意 | コンテンツ長・画像数から LCP 推定 |

---

## Sprint 16: Opus 4.8 対抗 — 広告 ROI / エージェント深化 v0.19.0

> **Why**: Opus 4.8 の Dynamic Workflows（最大1,000サブエージェント）に対抗。
> SwarmOrchestrator を「マーケ業務自律実行」レベルに引き上げる。

| task-id | 内容 | 優先度 | 詳細 |
| ------- | ---- | ------ | ---- |
| 16.1 | **マルチエージェント SEO プランナー** — SwarmOrchestrator `pipeline` で「キーワード調査 → 構成生成 → LLMO採点 → 改善提案」を自動化 | 🔴 必須 | `examples/seo_agent_pipeline.py` |
| 16.2 | **Quality Score 予測** — Google Ads QS (1-10) 推定ロジック。入力: 広告文+LP+キーワード | 🔴 必須 | `tools_marketing.quality_score()` |
| 16.3 | **広告コピー A/B 生成** — `broadcast` 戦略で5バリアント並列生成 + LLMOスコア付き返却 | 🟡 重要 | 1クエリで完結 |
| 16.4 | **ROAS シミュレーター** — 信頼区間付き。`calculate_roi()` 拡張 | 🟡 重要 | モンテカルロ推定 or ベイズ区間 |
| 16.5 | **プロンプトインジェクション耐性** — Opus 4.8 の7%脆弱性に対して、入力サニタイズ + ツール呼び出し検証を実装 | 🟡 重要 | `serve/api.py` にインジェクション検知ミドルウェア |
| 16.6 | **ペルソナ × 広告マッチング** — RAGPipeline + `tools_marketing.py` の連携 | 🟢 任意 | ペルソナドキュメントから広告関連度スコア |

---

## Sprint 17: Mythos 公開対応 & 外販 API 完成 v0.20.0

> **Why**: Mythos 公開は数週間以内。公開後72時間以内に差分分析・ベンチマーク比較を公開し
> 「最速で技術分析したオープン実装」として存在感を確立する。

| task-id | 内容 | 優先度 | 詳細 |
| ------- | ---- | ------ | ---- |
| 17.1 | **Mythos 公開即時対応** — リリース後72h以内に `docs/mythos_vs_openmythos.md` 公開。アーキテクチャ差分・ベンチマーク比較・OpenMythos の優位点を記載 | 🔴 必須 | GitHub で話題化を狙う |
| 17.2 | **APIキー認証** — Bearer Token 認証。Opus 4.8 と同等以上のセキュリティ | 🔴 必須 | 401 ガード + ログ記録 |
| 17.3 | **Docker 本番イメージ** — Gunicorn + uvicorn workers | 🟡 重要 | `docker run` で `/health` が返る |
| 17.4 | **レート制限** — RPM/TPM 制限 + 429。Opus 4.8 API と同等の信頼性 | 🟡 重要 | `slowapi` or カスタムミドルウェア |
| 17.5 | **OpenAPI ドキュメント** — 全エンドポイントに `summary/description/example` | 🟢 任意 | Swagger UI が外販説明資料になる水準 |
| 17.6 | **PyPI v0.20.0 リリース** — `python -m build && twine upload` | 🟢 任意 | `pip install open-mythos` で動く |

---

## 外販チェックリスト（改訂版）

```
[✅] テスト: pytest 860+ PASS（torch 環境で確認済み）
[✅] デモ: examples/demo_seo_llmo.ipynb が Colab でゼロから動く
[✅] ドキュメント: README に「5分で動かす」手順あり
[ ] API: /health, /v1/seo/score, /v1/seo/generate が実際に応答する
[ ] 認証: Bearer Token なしのリクエストを 401 で弾く
[ ] パッケージ: pip install open-mythos でインストールできる
[ ] 差別化証明: 「Opus 4.8 APIより実務精度が高い」をベンチマークで示せる ← NEW
[ ] Mythos 対応: 公開後72h以内に差分分析を公開できる態勢 ← NEW
```

---

## 技術的負債

| 項目 | 場所 | 内容 |
| ---- | ---- | ---- |
| 文字単位トークナイザ | `generate_seo.py`, `thinking.py`, `react.py` | SentencePiece / tiktoken に替えるべき |
| HuggingFace tokenizer 統合 | `serve/api.py` state.tokenizer | 現在スタブ。実 tokenizer 接続で精度向上 |
| API リトライ機構 | `serve/api.py` | exponential backoff (max 3回) 未実装 |
| `speculative_decode()` の eps | `main.py:1540` | `1e-10` → `1e-8` に変更推奨（数値安定性） |

---

## バージョンロードマップ

| バージョン | Sprint | キーワード |
| ---------- | ------ | ---------- |
| v0.17.0 | Sprint 14 ✅ | バグ修正 + デモ + README + テスト品質強化 |
| v0.18.0 | Sprint 15 | **Opus 4.8 対抗 — 2-phase SEO + 日本語形態素 + ベンチマーク** |
| v0.19.0 | Sprint 16 | **Opus 4.8 対抗 — マルチエージェント + 広告ROI + injection耐性** |
| v0.20.0 | Sprint 17 | **Mythos 公開対応 + 外販 API 完成 🚀** |
