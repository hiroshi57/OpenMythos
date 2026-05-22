---
name: completed-tasks
description: 完了したタスクの記録と学習ポイント
metadata:
  type: project
---

# 完了タスク記録


## top_p + decode_loops tests -- 2026-05-22

### 変更ファイル
- `open_mythos/main.py` — `OpenMythos.generate()` に `top_p` nucleus sampling 追加
- `open_mythos/hyperloop.py` — `HyperloopMythos.generate()` に `top_p` 追加
- `tests/test_main.py` — `TestDecodeLooops` (6テスト) + `TestTopPSampling` (5テスト)
- `tests/test_hyperloop.py` — `TestHyperloopDecodeOuterLoops` (4テスト) + `TestHyperloopTopP` (4テスト)

### 内容
- Option A (decode_loops) に対するテストが皆無だったため、shape/vocab/NaN/None等値を網羅
- top_p: `sorted_logits scatter` パターンで実装、top_k との組み合わせ可
- テスト: 188 → 207 PASS (+19)

---

## Option A: decode_loops 2-phase -- 2026-05-21 (commit ef85fa4)

### 変更ファイル
- `open_mythos/main.py` — `generate()` に `decode_loops` 引数追加
- `open_mythos/hyperloop.py` — `generate()` に `decode_outer_loops` 引数追加
- `tests/bench_vs_transformer.py` — `--decode-loops-sweep` オプション追加
- `benchmark_results/decode_speed_cpu_2026-05-21.txt`
- `benchmark_results/hyperloop_vs_flat_cpu_2026-05-21.txt`

### 結果
- prefill=4/decode=1: 3.65ms/step → 2.54x speedup vs full-depth (9.25ms)
- prefill=4/decode=2: 6.23ms/step → 1.48x speedup, ~99% quality

---

## Option B: 3-way benchmark -- 2026-05-21 (commit 6a64810, PR #1)

### 変更ファイル
- `tests/small_benchmark.py` — HyperloopMythos を第3モデルとして追加

### 内容
- `build_tiny_hyperloop_cfg()`: outer=2 x inner=2 (effective depth=4, flat と同等)
- `evaluate()`: HyperloopMythos 対応 (outer_loops sweep)
- argparse: `--hl-outer-loops` / `--hl-inner-loops` / `--hl-depth-sweep`
- main(): 3モデル並列学習・4カラムlog・summaryテーブル・depth sweep・cross-model比較
- Windows CP932 対応: Unicode 記号を全て ASCII に変換

### PR
- branch: `feature/hyperloop-benchmark`
- PR: https://github.com/hiroshi57/OpenMythos/pull/1

### 次のステップ
- 100 steps ベンチマーク実行 → `benchmark_results/hyperloop_vs_flat_cpu_<date>.txt` 保存
- PR マージ後に Option A (KV cache decode 最適化) へ

---

## 4-task sprint — 2026-05-19 (commit e03fed1)

### Task 1: Training script improvements
- 追加: perplexity ログ、ETA 推定、loguru ファイルシンク、eval step、loop curriculum
- ファイル: `training/3b_fine_web_edu.py`

### Task 2: MoDA × OpenMythos 統合テスト
- 新規: `tests/test_integration.py` (38 テスト)
- カバー: 形状・NaN・勾配・シリアライズ・generation・config 互換性

### Task 3: 新機能
- `open_mythos/moda.py`: `MoDAModel.generate()` 追加 (top-k + temperature)
- `open_mythos/hyperloop.py`: `HyperloopBlock` + `HyperloopMythos` (arXiv 2604.21254 ベース)
  - 2-level nested loop (outer × inner)、同等パラメータ数で指数的深さ
  - 両注入レベルで LTI スペクトル半径 < 1 保証
- `tests/test_hyperloop.py`: 29 テスト

### Task 4: ベンチマーク
- `benchmark_results/bench_vs_transformer_cpu_2026-05-19.txt`
  - OpenMythos: 43% 少ないパラメータ、短 seq で 2-7x 高速 prefill
- `benchmark_results/small_benchmark_tinystories_2026-05-19.txt`
  - Mythos が baseline より低い eval loss を達成
  - 深度外挿確認: n_loops=2 で n_loops=4 品質の 99% を回収

テスト: 188/188 PASS (元 121 から +67)

---

## 1.1.1 — 2026-05-18
- **説明**: open_mythos/main.py �� open_mythos/variants.py �̃J�o���b�W���|�[�g����
- **結果**: PASS
- **ログ**: .claude/state/logs/1.1.1-20260518-145204.log
