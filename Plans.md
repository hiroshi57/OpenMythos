# OpenMythos — Sprint Plans
> 最終更新: 2026-05-22 | ブランチ規約: `feature/<sprint>-<topic>`

---

## Sprint 1: HyperloopMythos & Inference Engine (完了)

> ブランチ: `feature/hyperloop-benchmark` → master merge pending

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 1.1.1 | 3-way benchmark (HyperloopMythos vs OpenMythos flat vs Baseline) | Worker | cc:完了 [6a64810] | (a) small_benchmark.py に3モデル追加 (b) benchmark_results/ に結果保存 |
| 1.1.2 | decode_loops 2-phase depth strategy (2.54x decode 高速化) | Worker | cc:完了 [ef85fa4] | (a) prefill=4/decode=1 実装 (b) 207 PASS |
| 1.1.3 | top_p nucleus sampling 実装 | Worker | cc:完了 [3ea6e68] | (a) top_p パラメータ追加 (b) decode_loops/decode_outer_loops テスト追加 (c) 207 PASS |
| 1.1.4 | generate_stream() + _sample_token() リファクタ | Worker | cc:完了 [fe7daba] | (a) ストリーミング生成 API 追加 (b) 220 PASS |
| 1.1.5 | speculative_decode() + _causal_mask cache_len 修正 | Worker | cc:完了 [0431b15] | (a) speculative decode 実装 (b) 227 PASS |
| 1.1.6 | lint 整備 (.gitattributes CRLF/LF 統一) | Worker | cc:完了 [040b174] | (a) ruff/black 全 pass (b) CI lint mismatch 解消 |
| 1.2.1 | feature/hyperloop-benchmark → master merge | Worker | cc:WIP | (a) 227 PASS on master (b) ローカル merge 完了 |
| 1.2.2 | GitHub push & CI 確認 | Worker | cc:TODO | (a) GitHub Actions green (b) PR close |

---

## Sprint 2: Inference 高度化 (Next)

> ブランチ予定: `feature/inference-v2`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 2.1.1 | beam search 実装 | Worker | cc:TODO | (a) beam_width パラメータ追加 (b) テスト追加 (c) 既存テスト全 PASS |
| 2.1.2 | temperature / repetition_penalty 強化 | Worker | cc:TODO | (a) repetition_penalty 実装 (b) テスト追加 |
| 2.1.3 | KV cache 最適化 (sliding window) | Worker | cc:TODO | (a) メモリ使用量 20% 削減 (b) ベンチマーク比較 |
| 2.2.1 | INT8 / FP16 量子化サポート | Worker | cc:TODO | (a) torch.quantization 対応 (b) ベンチマーク追加 |
| 2.2.2 | バッチ推論 API | Worker | cc:TODO | (a) generate_batch() 実装 (b) 並列実行テスト |

---

## Sprint 3: ドキュメント & エコシステム (Backlog)

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 3.1.1 | README.md 更新 (speculative decode / HyperloopMythos 追記) | Worker | cc:TODO | (a) 新機能セクション追加 (b) benchmark 結果グラフ追加 |
| 3.1.2 | docs/architecture.md 作成 | Worker | cc:TODO | (a) アーキテクチャ図 (b) コンポーネント説明 |
| 3.1.3 | PyPI パッケージ v0.6.0 リリース準備 | Worker | cc:TODO | (a) pyproject.toml version bump (b) CHANGELOG 更新 |
| 3.2.1 | GitHub Actions CI 強化 (coverage report) | Worker | cc:TODO | (a) pytest-cov 追加 (b) coverage badge |

---

## 進行中の作業メモ

### 現在のブランチ状態
- `feature/hyperloop-benchmark`: 227 PASS、master に 8 commit 先行
- master: `e03fed1` — 4-task sprint 完了時点

### 重要な技術的知見
- `freqs_cis` は必ず `[:T]` スライスして渡すこと (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` は `.clamp(min=1e-6)` が必要 (float32 飽和防止)
- decode_loops 2-phase: prefill=4 / decode=1 が最速 (2.54x)
