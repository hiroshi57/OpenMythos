# OpenMythos — 課題集約・ロードマップ

> 最終更新: 2026-07-04 | 現在バージョン: **v0.80.0** (4425 collected / 4400+ PASS) | ruff エラー: **0**
> ブランチ: feature/sprint71-city-map | 品質改革: Fable 5 全 Sprint レビュー完了

## Sprint 進捗サマリー（〜Sprint 77）

| Sprint | バージョン | 主な成果 |
| ------ | ---------- | -------- |
| 1〜16 | v0.1〜v0.19 | 推論エンジン・Training 基盤・SEO パイプライン・QS 予測・外販準備 |
| 17〜19 | v0.20〜v0.22 | API 認証・Docker・レート制限・OpenAPI・LLMO 強化 |
| 20〜30 | v0.23〜v0.33 | 育つ AI P1〜P10 + GrowingAIOrchestrator 統合 |
| 31〜35 | v0.34〜v0.38 | GPU LoRA SFT・SQLite・FAISS・Opus 比較ベンチマーク |
| 36〜43 | v0.39〜v0.46 | 1B スケール・PyPI・API 強化・Hermes Layer2 |
| 44〜52 | v0.47〜v0.55 | スキル統合 (VectorDB/HF Hub/推論 BE/研究/マルチモーダル/訓練/エージェント/データ/DevOps) |
| 53〜59 | v0.56〜v0.62 | セキュリティ統合・Assistants 互換・SSE・マルチプロバイダー・評価 FW・LLMO ダッシュボード・脆弱性スキャン |
| 60〜67 | v0.63〜v0.70 | 広告運用オートメーション (キャンペーン/コピー生成/A/B/予算/Fusion/異常検知) |
| 68〜70 | v0.71〜v0.73 | セキュリティインテル・時系列予測 TimesFM・予測アラート/Webhook/NLQ |
| 71〜77 | v0.74〜v0.80 | 都市地図ドメイン (断面図/比較/編集/経路/混雑/環境/災害/水質/騒音) |

テスト: 4425 collected (4400+ PASS) / ruff: 0 errors / CI: 全テストファイル実行

---

## 品質改革の記録 (2026-07-04, Fable 5)

| Phase | 内容 | 状態 |
| ----- | ---- | ---- |
| 0 | ruff 103→0、Sprint 72〜77 欠陥 6 件修正、全テスト triage (5 FAIL + 13 ERROR 修正) | ✅ |
| 1 | CI 全件実行化・pytest 設定・バージョン一元化 (`open_mythos.__version__`) | ✅ |
| 2 | api.py 9264 行モノリス → `serve/routers/` ドメイン分割 (ルート数 305 保存) | ✅ |
| 3 | ドキュメント整合 (本ファイル・CHANGELOG・Plans) | ✅ |

### serve/ 構成 (Phase 2 以降)

```text
serve/api.py                      — app 生成・コア推論・SEO/LLMO・ミドルウェア
serve/state.py                    — AppState シングルトン + ループ定数
serve/auth.py                     — 認証・レート制限
serve/routers/map.py              — 都市地図 (Sprint 71〜77)
serve/routers/ads.py              — 広告運用 (Sprint 60/63〜67/69/70)
serve/routers/llm_services.py     — Assistants/SSE/マルチプロバイダー/評価/LLMO-DB (54〜58)
serve/routers/skills_integrations.py — スキル統合 (44〜52)
serve/routers/growing_ai.py       — 育つ AI + A/B + ROAS + Hermes (18/20〜30/35/43)
serve/ab_router.py sla_router.py monitor.py — スタンドアロン (monitor は compose 独立サービス)
```

---

## 外販チェックリスト

```text
[✅] テスト: pytest 4400+ PASS (4425 collected) / CI 全件実行
[✅] デモ: examples/demo_seo_llmo.ipynb が Colab でゼロから動く
[✅] ドキュメント: README に「5分で動かす」手順あり
[✅] セキュリティ: InputGuard/OutputGuard/MistakeGuardMiddleware + 脆弱性スキャン + 脅威インテル
[✅] 広告: キャンペーン管理・A/B・予算最適化・異常検知・予測アラート
[✅] SEO: 2-phase ワークフロー・LLMO スコア・ダッシュボード・CEP 管理
[✅] API: 305 ルート / ドメイン別ルーター構成 / OpenAPI
[✅] 認証: Bearer Token なしを 401 で弾く
[✅] Docker: docker-compose.yml production 対応 (api + monitor)
[✅] 保守性: api.py 分割済み・ruff 0・バージョン単一ソース
[ ] 差別化証明: 「Opus 4.8 より実務精度が高い」ベンチマーク公開 (Sprint 35 結果の外部公開)
[ ] PyPI: v0.80.0 リリース (pip install open-mythos)
```

---

## 次期 Sprint 候補 (Sprint 78〜)

| Option | テーマ | 内容 | 優先度 |
| ------ | ------ | ---- | ------ |
| A | **mypy 段階導入** | ルーター分割済みの serve/ から型チェックを段階適用 | 🟡 |
| B | **テスト実行時間短縮** | 共有 TestClient fixture の全面適用・slow マーカー分離 (現状フル 32 分) | 🟡 |
| C | **PyPI v0.80.0 リリース** | 品質改革後の安定版を公開 | 🔴 |
| D | **都市地図フロントエンド** | SVG 断面図/アニメーションをブラウザ UI で提供 | 🟢 |
| E | **cloudbuild テストゲート** | Cloud Build にテスト実行ステップ追加・Cloud Run デプロイ | 🟢 |

---

## 技術的負債（忘れないリスト）

| 優先度 | 項目 | 場所 | 内容 |
| ------ | ---- | ---- | ---- |
| 🟡 | mypy 未導入 | 全体 | ルーター分割完了により段階導入の費用対効果が向上 (Sprint 78 候補 A) |
| 🟡 | ソース文字列検査テスト | tests/test_sprint20〜25 ほか | `_src()` が api.py+routers 連結を検査。挙動ベースの assert への移行が望ましい |
| 🟡 | 文字単位トークナイザ | `thinking.py`, `react.py` | SentencePiece / tiktoken に替えるべき |
| 🟡 | API リトライ機構 | `serve/api.py` | exponential backoff (max 3回) 未実装 |
| 🟢 | `speculative_decode()` の eps | `main.py:1540` | `1e-10` → `1e-8` に変更推奨（数値安定性） |
| 🟢 | tools_marketing.py スタブ | 全関数 | 本番では SimilarWeb / SEMrush / Google Trends API に差し替え |
| 🟢 | FAISS faiss-cpu optional | `requirements.txt` | faiss-cpu を optional 依存として正式整理 |
| 🟢 | CHANGELOG 0.42〜0.46 | CHANGELOG.md | Sprint 39〜43 の個別エントリ未記載 (要約のみ) |
