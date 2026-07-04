# OpenMythos — 課題集約・ロードマップ
> 最終更新: 2026-06-05 | 現在バージョン: **v0.37.0** (1753 collected / 1400+ PASS) | ruff エラー: **0**
> ブランチ: master | 最新コミット: `f06fad4`

## Sprint 進捗サマリー（〜Sprint 34）

| Sprint | バージョン | 主な成果 |
| ------ | ---------- | -------- |
| Sprint 1〜16 | v0.1〜v0.19 | 推論エンジン・Training基盤・SEOパイプライン・QS予測・外販準備 |
| Sprint 17〜19 | v0.20〜v0.22 | API認証・Docker・レート制限・OpenAPI・LLMO強化・LLMOOptimizer |
| Sprint 20 | v0.23.0 | 討議型集合知 DebateOrchestrator (P1パターン) |
| Sprint 21 | v0.24.0 | KPI駆動自己改善 KPIAgent (P2パターン) |
| Sprint 22 | v0.25.0 | ボトルネック発見 ProfilerAgent (P3パターン) |
| Sprint 23 | v0.26.0 | 外部要因適応 ExternalSignalAgent (P4パターン) |
| Sprint 24〜25 | v0.28.0 | ミスから学習 ErrorMemory/MistakeGuard + SelfDistillLoop (P5〜P6) |
| Sprint 26〜29 | v0.29〜v0.32 | 長期記憶・アンサンブル・プロンプト進化・自律タスク計画 (P7〜P10) |
| Sprint 30 | v0.33.0 | GrowingAIOrchestrator — P1〜P10 統合オーケストレーター |
| Sprint 31 | v0.34.0 | GPU LoRA SFT 統合 (LoraTrainer / sft_backend) |
| Sprint 32 | v0.35.0 | エラーメモリ SQLite 永続化 + export API |
| Sprint 33 | v0.36.0 | LongTermMemory FAISS ANN インデックス |
| Sprint 34 | v0.37.0 | MistakeGuardMiddleware — 全 API エンドポイント透過チェック |

テスト: 1753 collected (1400+ PASS) / ruff: 0 errors

---

## 現在地（完了済み）

| バージョン | Sprint | 主な成果 |
| ---------- | ------ | -------- |
| v0.33.0 ✅ | Sprint 30 | GrowingAIOrchestrator (P1〜P10 統合) |
| v0.34.0 ✅ | Sprint 31 | GPU LoRA SFT 統合 |
| v0.35.0 ✅ | Sprint 32 | ErrorMemory SQLite 永続化 + export API |
| v0.36.0 ✅ | Sprint 33 | LongTermMemory FAISS ANN インデックス |
| v0.37.0 ✅ | Sprint 34 | MistakeGuardMiddleware 全 API 透過チェック |

---

## 外販チェックリスト

```text
[✅] テスト: pytest 1400+ PASS (1753 collected)
[✅] デモ: examples/demo_seo_llmo.ipynb が Colab でゼロから動く
[✅] ドキュメント: README に「5分で動かす」手順あり
[✅] セキュリティ: InputGuard/OutputGuard/MistakeGuardMiddleware インジェクション耐性
[✅] 広告: QS予測・広告バリアント生成・ROI計算
[✅] SEO: 2-phase ワークフロー・LLMO スコア・A/Bテスト
[✅] API: /health, /v1/seo/score, /v1/seo/generate, /v1/guard/stats が応答
[✅] 認証: Bearer Token なしを 401 で弾く
[✅] Docker: docker-compose.yml production 対応
[✅] ガード: MistakeGuardMiddleware が全エンドポイントを透過チェック
[ ] 差別化証明: 「Opus 4.8 より実務精度が高い」をベンチマークで示せる (Sprint 35)
[ ] FT実証: GCP T4 GPU + 社内 SEO データで LoRA FT 実行 (Sprint 35)
[ ] PyPI: pip install open-mythos でインストールできる (Sprint 36)
```

---

## Sprint 35: ファインチューニング実証 & Opus 4.8 比較 v0.38.0

> **Why**: LoraTrainer (Sprint 31) が完成済み。「Opus 4.8 より実務精度が高い」を社内 SEO データで数値証明する。ここが最大の差別化。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 35.1 | **社内 SEO データ変換** — `scripts/csv_to_jsonl.py --inspect` で列名確認 → `content_quality` タスク形式に変換 | 🔴 必須 | `data/seo_train.jsonl` 生成 |
| 35.2 | **LoRA FT 実行** — GCP T4 GPU で LoraTrainer を社内データで実行。`scripts/pretrain_gcp.sh` ベース | 🔴 必須 | perplexity < 20 確認 |
| 35.3 | **Opus 4.8 API との精度比較** — 同一入力で LLMO スコア・人間評価を比較。「OpenMythos が自社タスクで上回る」を数値証明 | 🔴 必須 | 比較レポートを `benchmark/results/` に保存 |
| 35.4 | **A/B ルーター本番接続** — `serve/api.py` の A/B ルーターで OpenMythos と Claude API を 20% vs 80% 並走 | 🟡 重要 | 実トラフィックでの比較開始 |
| 35.5 | **ROAS シミュレーター強化** — `calculate_roi()` 拡張。モンテカルロ推定で信頼区間付き ROAS 予測 | 🟡 重要 | `roas_simulate(ad_spend, ctr, cvr, aov, n=1000)` |

---

## Sprint 36: スケールアップ & OSS 公開 v0.39.0

> **Why**: FT 精度証明後、1B モデルへ。PyPI 公開で外部開発者を獲得しコミュニティを構築する。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 36.1 | **1B モデルへのスケール** — `mythos_1b()` 設定で FT。300M との精度差をベンチマーク | 🔴 必須 | LLMO スコア 300M 比 +5pt 以上 |
| 36.2 | **PyPI v0.38.0 リリース** — CHANGELOG + `python -m build` + `twine upload` | 🔴 必須 | `pip install open-mythos` で動く |
| 36.3 | **Janome/Fugashi 本番統合** — `requirements.txt` に `janome>=0.5` を正式追加。CI で確認 | 🟡 重要 | `pip install open-mythos[ja]` で動く |
| 36.4 | **lm-eval ベンチマーク公開** — HellaSwag / ARC-Easy / WinoGrande を Claude Opus 4.8 と比較して公開 | 🟡 重要 | `benchmark/results/` + README に掲載 |
| 36.5 | **Core Web Vitals シミュレーター** — `structured.py` に LCP/CLS 疑似スコア算出実装 | 🟢 任意 | コンテンツ長・画像数から LCP 推定 |

---

## Sprint 37: GrowingAI 深化 & 品質向上 v0.40.0

> **Why**: GrowingAIOrchestrator (Sprint 30) は P1〜P10 統合済み。各パターンの実精度向上と外部 API 連携で差別化を深める。

| task-id | 内容 | 優先度 | DoD |
| ------- | ---- | ------ | --- |
| 37.1 | **PatternSelector 精度向上** — 実トラフィックのログを元にキーワードマッチルールを改善 | 🔴 必須 | パターン誤選択率 < 5% |
| 37.2 | **DebateOrchestrator × LTM 統合** — 討議ログを LongTermMemoryAgent に自動蓄積 | 🟡 重要 | `store_debate_history=True` オプション追加 |
| 37.3 | **MistakeGuardMiddleware ダッシュボード** — `/v1/guard/stats` データを Grafana / カスタム UI で可視化 | 🟡 重要 | リアルタイムブロック率グラフ表示 |
| 37.4 | **SelfDistillLoop × LoRA SFT 実接続** — シミュレーションから実 GPU SFT に切り替え | 🟢 任意 | `sft_backend="lora"` で実 FT が走る |
| 37.5 | **tools_marketing.py 本番 API 接続** — SimilarWeb / SEMrush / Google Trends API に差し替え | 🟢 任意 | 実データで広告関連度スコア算出 |

---

## 技術的負債（忘れないリスト）

| 優先度 | 項目 | 場所 | 内容 |
| ------ | ---- | ---- | ---- |
| 🟡 | 文字単位トークナイザ | `thinking.py`, `react.py` | SentencePiece / tiktoken に替えるべき |
| 🟡 | HuggingFace tokenizer 統合 | `serve/api.py` | 現在スタブ。実 tokenizer 接続で精度向上 |
| 🟡 | API リトライ機構 | `serve/api.py` | exponential backoff (max 3回) 未実装 |
| 🟢 | `speculative_decode()` の eps | `main.py:1540` | `1e-10` → `1e-8` に変更推奨（数値安定性） |
| 🟢 | tools_marketing.py スタブ | 全関数 | 本番では SimilarWeb / SEMrush / Google Trends API に差し替え |
| 🟢 | FAISS faiss-cpu optional | `requirements.txt` | faiss-cpu を optional 依存として正式整理 |

---

## バージョンロードマップ

| バージョン | Sprint | キーワード | 状態 |
| ---------- | ------ | ---------- | ---- |
| v0.33.0 | Sprint 30 | GrowingAIOrchestrator (P1〜P10) | ✅ 完了 |
| v0.34.0 | Sprint 31 | GPU LoRA SFT 統合 | ✅ 完了 |
| v0.35.0 | Sprint 32 | ErrorMemory SQLite 永続化 | ✅ 完了 |
| v0.36.0 | Sprint 33 | LongTermMemory FAISS ANN | ✅ 完了 |
| v0.37.0 | Sprint 34 | MistakeGuardMiddleware 全 API 透過 | ✅ 完了 |
| v0.38.0 | Sprint 35 | **FT実証・Opus 4.8 比較・ROAS** | 次回 |
| v0.39.0 | Sprint 36 | **1Bスケール・PyPI公開・OSS展開** | |
| v0.40.0 | Sprint 37 | **GrowingAI深化・外部API本番接続** | |
