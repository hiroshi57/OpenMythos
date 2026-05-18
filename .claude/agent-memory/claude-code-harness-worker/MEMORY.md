# Memory Index
> OpenMythos プロジェクト — 最終更新: 2026-05-18

## project（プロジェクト状況）
- [Solo-Team Loop フレームワーク初期構築](solo-team-framework-setup.md) — THINK-ACT-LEARN ループ・エージェントロール・スクリプト一式を framework/ に構築完了
- [テスト 121/121 PASS 達成](completed-tasks.md) — freqs スライス修正・LTI float32 安定性修正・CI workflow 追加で全テスト緑化

## feedback（バグパターン・注意点）
- [freqs_cis は必ず [:T] スライスして渡す](freqs-slice-rule.md) — GQA/MLA/Block/RecurrentBlock は (T, head_dim//2) 形状を期待。max_seq_len の全長を渡すと apply_rope でブロードキャストエラー
- [LTI get_A() の float32 飽和](lti-float32-saturation.md) — 大きな lr で log_dt+log_A が -20 以下になると exp(-exp(-20))≈1.0 に丸まる。inner exp に .clamp(min=1e-6) が必要

## reference（外部リソース）
- [GitHub リポジトリ](https://github.com/hiroshi57/OpenMythos) — master ブランチ。CI は Actions タブで確認
- [Qiita 参考記事](https://qiita.com/sescore/items/170d695868d4bf7fb2ce) — OpenClaw×Claude Code 実践ガイド。4 ユースケース＋フレームワーク設計の起点
