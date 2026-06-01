# OpenMythos — 課題集約・ロードマップ
> 最終更新: 2026-05-31 | 現在バージョン: **v0.19.0** (930+ PASS) | ruff エラー: **0**
> ブランチ: master | 最新コミット: `a679311`

## 今回セッション完了サマリー（2026-05-31）

| Sprint | バージョン | 主な成果 |
| ------ | ---------- | -------- |
| Sprint 14 | v0.17.0 | バグ修正2件・テスト品質強化9件・デモノートブック・README外販版 |
| Sprint 15 | v0.18.0 | 日本語形態素解析・重み付きKW密度・A/Bテスト・ドリフト検出・LLMOベンチマーク |
| Sprint 16 | v0.19.0 | SEOパイプライン・QS予測・広告バリアント生成・インジェクション耐性 |
| CI修正 | — | ruff 108エラー→0（TYPE_CHECKING追加・未使用変数削除・関数名重複修正） |

**テスト: 930+ PASS / Warning 2件（PyTorch CPU autocast 固有・無害）/ ruff: 0 errors**

---

## 現在地（完了済み）

| バージョン | Sprint | 主な成果 |
| ---------- | ------ | -------- |
| v0.17.0 ✅ | Sprint 14 | バグ修正2件・テスト品質強化9件・デモノートブック・README外販版 |
| v0.18.0 ✅ | Sprint 15 | 日本語形態素解析・重み付きKW密度・A/Bテスト・ドリフト検出・LLMOベンチマーク |
| v0.19.0 ✅ | Sprint 16 | SEOパイプライン・QS予測・広告バリアント生成・インジェクション耐性 |

---

## 競合状況（2026-05-31）

| 競合 | 弱点 | OpenMythos の対応 |
| ---- | ---- | ----------------- |
| Claude Opus 4.8 | コンテキストドリフト・injection 7%・日本語コスト高・汎用SEOのみ | 2-phase分離・InputGuard・オンプレ・形態素解析 — 全て実装済み |
| Claude Mythos | 数週間以内に一般公開予定。招待制（Project Glasswing）| 公開後72h以内に差分分析文書を公開する態勢を整える |
| GPT-5.5 | 文章品質で Opus 4.8 に劣後。マルチモーダルは中程度 | スコープ外 |
| Gemini 3.1 Pro | マルチモーダル最強。コスト効率 Opus の2.4倍 | テキスト特化に集中。コスト優位はオンプレで確保 |

---

## Sprint 17: Mythos 公開対応 & 外販 API 完成 v0.20.0

> **緊急度高**: Mythos は数週間以内に一般公開。公開後72h以内の対応が差別化のカギ。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 17.1 | **Mythos 公開即時対応** — 公開後72h以内に `docs/mythos_vs_openmythos.md` を作成。アーキテクチャ差分・ベンチマーク比較・OpenMythos の優位点を記載し GitHub で公開 | 🔴 必須 | ドキュメント公開・SNS告知 |
| 17.2 | **APIキー認証** — `serve/api.py` に Bearer Token 認証。`InputGuard` を middleware に組み込み | 🔴 必須 | 401 ガード + ログ記録 |
| 17.3 | **Docker 本番イメージ** — Gunicorn + uvicorn workers。`docker build && docker run` で `/health` が返る | 🟡 重要 | `docker-compose.yml` production 対応 |
| 17.4 | **レート制限** — エンドポイント別 RPM/TPM + 429。`slowapi` or カスタムミドルウェア | 🟡 重要 | 過負荷テスト PASS |
| 17.5 | **OpenAPI ドキュメント整備** — 全エンドポイントに `summary/description/example` | 🟢 任意 | `/docs` が外販説明資料になる水準 |
| 17.6 | **PyPI v0.20.0 リリース** — CHANGELOG + `python -m build` + `twine upload` | 🟢 任意 | `pip install open-mythos` で動く |

---

## Sprint 18: ファインチューニング & 実証 v0.21.0

> **Why**: 「Opus 4.8 より実務精度が高い」を社内 SEO データで数値証明する。ここが最大の差別化。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 18.1 | **社内 SEO データ変換** — `scripts/csv_to_jsonl.py --inspect` で列名確認 → `content_quality` タスク形式に変換 | 🔴 必須 | `data/seo_train.jsonl` 生成 |
| 18.2 | **ファインチューニング実行** — GCP T4 GPU で 300M モデルを社内データで FT。`scripts/pretrain_gcp.sh` ベース | 🔴 必須 | perplexity < 20 確認 |
| 18.3 | **Opus 4.8 API との精度比較** — 同一入力で LLMO スコア・人間評価を比較。「OpenMythos が自社タスクで上回る」を数値証明 | 🔴 必須 | 比較レポートを `benchmark/results/` に保存 |
| 18.4 | **A/B ルーター本番接続** — `serve/api.py` の A/B ルーターで OpenMythos と Claude API を 20% vs 80% 並走 | 🟡 重要 | 実トラフィックでの比較開始 |
| 18.5 | **ROAS シミュレーター** — `calculate_roi()` 拡張。モンテカルロ推定で信頼区間付き ROAS 予測 | 🟡 重要 | `roas_simulate(ad_spend, ctr, cvr, aov, n=1000)` |
| 18.6 | **ペルソナ × 広告マッチング** — `RAGPipeline` + `tools_marketing.py` 連携。ペルソナドキュメントから広告関連度スコア | 🟢 任意 | `rag.py` + `tools_marketing.py` 結合 |

---

## Sprint 19: スケールアップ & 外部公開 v0.22.0

> **Why**: 300M で精度証明できたら 1B へ。外部開発者への公開で OSS コミュニティを作る。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 19.1 | **1B モデルへのスケール** — `mythos_1b()` 設定で FT。300M との精度差をベンチマーク | 🔴 必須 | LLMO スコア 300M 比 +5pt 以上 |
| 19.2 | **Janome/Fugashi 本番統合** — `requirements.txt` に `janome>=0.5` を正式追加。CI でインストール確認 | 🟡 重要 | `pip install open-mythos[ja]` で動く |
| 19.3 | **Core Web Vitals シミュレーター** — `structured.py` に LCP/CLS の疑似スコア算出実装 | 🟡 重要 | コンテンツ長・画像数から LCP 推定 |
| 19.4 | **lm-eval ベンチマーク公開** — HellaSwag / ARC-Easy / WinoGrande を Mythos と比較して公開 | 🟢 任意 | `benchmark/results/` + README に掲載 |

---

## 外販チェックリスト

```
[✅] テスト: pytest 930+ PASS
[✅] デモ: examples/demo_seo_llmo.ipynb が Colab でゼロから動く
[✅] ドキュメント: README に「5分で動かす」手順あり
[✅] セキュリティ: InputGuard/OutputGuard によるインジェクション耐性
[✅] 広告: QS予測・広告バリアント生成・ROI計算
[✅] SEO: 2-phase ワークフロー・LLMO スコア・A/Bテスト
[ ] API: /health, /v1/seo/score, /v1/seo/generate が実際に応答 (Sprint 17)
[ ] 認証: Bearer Token なしを 401 で弾く (Sprint 17)
[ ] 差別化証明: 「Opus 4.8 より実務精度が高い」をベンチマークで示せる (Sprint 18)
[ ] Mythos 対応: 公開後72h以内に差分分析を公開できる態勢 (Sprint 17)
[ ] パッケージ: pip install open-mythos でインストールできる (Sprint 17)
```

---

## 技術的負債（忘れないリスト）

| 優先度 | 項目 | 場所 | 内容 |
| ------ | ---- | ---- | ---- |
| 🟡 | 文字単位トークナイザ | `thinking.py`, `react.py` | SentencePiece / tiktoken に替えるべき |
| 🟡 | HuggingFace tokenizer 統合 | `serve/api.py` | 現在スタブ。実 tokenizer 接続で精度向上 |
| 🟡 | API リトライ機構 | `serve/api.py` | exponential backoff (max 3回) 未実装 |
| 🟢 | `speculative_decode()` の eps | `main.py:1540` | `1e-10` → `1e-8` に変更推奨（数値安定性） |
| 🟢 | tools_marketing.py スタブ | 全関数 | 本番では SimilarWeb / SEMrush / Google Trends API に差し替え |

---

## バージョンロードマップ

| バージョン | Sprint | キーワード | 状態 |
| ---------- | ------ | ---------- | ---- |
| v0.17.0 | Sprint 14 | バグ修正・外販準備 | ✅ 完了 |
| v0.18.0 | Sprint 15 | 日本語形態素・LLMO深化 | ✅ 完了 |
| v0.19.0 | Sprint 16 | SEOパイプライン・QS・injection耐性 | ✅ 完了 |
| v0.20.0 | Sprint 17 | **Mythos対応・外販API完成** | 次回 |
| v0.21.0 | Sprint 18 | **FT実証・Opus 4.8比較・ROAS** | |
| v0.22.0 | Sprint 19 | **1Bスケール・OSS公開** | |
