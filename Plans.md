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
| 1.2.1 | feature/hyperloop-benchmark → master merge | Worker | cc:完了 | (a) 227 PASS on master (b) ローカル merge 完了 |
| 1.2.2 | GitHub push & CI 確認 | Worker | cc:完了 | (a) GitHub Actions green (b) PR close |

---

## Sprint 2: Inference 高度化 (完了)

> ブランチ: `feature/inference-v2`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 2.1.1 | beam search 実装 | Worker | cc:完了 [905f761] | (a) beam_width パラメータ追加 (b) テスト追加 (c) 既存テスト全 PASS |
| 2.1.2 | temperature / repetition_penalty 強化 | Worker | cc:完了 [f2cb589] | (a) repetition_penalty 実装 (b) テスト追加 |
| 2.1.3 | KV cache 最適化 (sliding window) | Worker | cc:完了 [9946ef1] | (a) max_cache_len パラメータ (b) GQA/MLA 両対応 |
| 2.2.1 | INT8 / FP16 量子化サポート | Worker | cc:完了 [5119502] | (a) quantize("fp16"/"int8") 実装 (b) dtype-safe ACT/LTI修正 |
| 2.2.2 | バッチ推論 API | Worker | cc:完了 | (a) generate_batch() 実装 (b) 可変長プロンプト対応 |

---

## Sprint 3: ドキュメント & エコシステム (完了)

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 3.1.1 | README.md 更新 (speculative decode / HyperloopMythos 追記) | Worker | cc:完了 | (a) Inference API 表追加 (b) 全生成メソッドのコード例追加 |
| 3.1.2 | docs/architecture.md 作成 | Worker | cc:完了 | (a) コンポーネントマップ (b) アルゴリズム詳細 (c) 設定リファレンス |
| 3.1.3 | PyPI パッケージ v0.6.0 リリース準備 | Worker | cc:完了 | (a) pyproject.toml 0.5.0→0.6.0 (b) CHANGELOG.md 作成 |
| 3.2.1 | GitHub Actions CI 強化 (coverage report) | Worker | cc:完了 | (a) coverage XML 生成 (b) artifact upload |

---

## Sprint 4: Training 基盤強化 & エコシステム (完了)

> ブランチ: `master`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 4.1.1 | `mythos_nano` variant 追加 | Worker | cc:完了 | (a) variants.py + __init__.py 追加 (b) 257 PASS |
| 4.1.2 | training スクリプト整備 (gradient checkpointing, bf16) | Worker | cc:完了 | (a) TransformerBlock に gradient_checkpointing フラグ追加 (b) training スクリプトで有効化 (c) 257 PASS |
| 4.1.3 | test_training.py 追加 (nano config 1-step 訓練テスト) | Worker | cc:完了 | (a) CPU で 1-step 訓練が完走 (b) 8 tests PASS |
| 4.2.1 | MoDa テスト追加 (test_moda.py) | Worker | cc:完了 | (a) 54 tests PASS (既存ファイル確認済み) |
| 4.2.2 | MANIFEST.in + PyPI 公開準備 | Worker | cc:完了 | (a) MANIFEST.in 作成 (b) pip install --dry-run 確認 (c) CI/テストバッジ追加 |

---

## Sprint 5: Training 品質改善 & エコシステム拡張 (完了)

> ブランチ: `master`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 5.1.1 | `warmup_stable_decay` 3-phase LR スケジューラ | Worker | cc:完了 | (a) training スクリプトに実装 (b) warmup/plateau/decay 3フェーズ動作 |
| 5.1.2 | MuP 初期化 `init_mup(base_dim=256)` | Worker | cc:完了 | (a) OpenMythos に実装 (b) import math 追加 |
| 5.1.3 | GNS 監視 + 早期停止 | Worker | cc:完了 | (a) _gns_window deque 追加 (b) es_patience=5/es_min_delta=1e-3 実装 |
| 5.2.1 | CLI `mythos generate` / `mythos info` | Worker | cc:完了 | (a) open_mythos/cli.py 作成 (b) pyproject.toml scripts エントリ追加 |
| 5.2.2 | HF Hub 連携 `push_to_hub` / `from_pretrained` | Worker | cc:完了 | (a) OpenMythos に2メソッド追加 (b) HfApi/hf_hub_download 利用 |
| 5.3.1 | LoRA ファインチューニング API `enable_lora_finetuning` | Worker | cc:完了 | (a) enable_lora_finetuning() 実装 (b) trainable_parameters() 追加 |
| 5.4.1 | PyPI 正式リリース準備 + GitHub Release v0.7.0 | Worker | cc:完了 | (a) CHANGELOG.md 更新 (b) pyproject.toml 0.7.0 (c) Plans.md Sprint 5 追加 |
| 5.5.1 | Sprint 5 テスト追加 + commit + push | Worker | cc:完了 | (a) test_sprint5.py 作成 (b) 284 PASS (c) GitHub push |

---

## Sprint 6: 推論最適化 & 訓練高度化 & エージェント統合 & ベンチマーク (完了)

> ブランチ: `master`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 6.1.1 | `compile_model()` — torch.compile ラッパー（TransformerBlock/RecurrentBlock 個別コンパイル） | Worker | cc:完了 | (a) backend="eager" でWindows CPU対応 (b) forward 正常動作 |
| 6.1.2 | SDPA 最適化 — GQA/MLA fallback を `F.scaled_dot_product_attention` に置き換え | Worker | cc:完了 | (a) Flash Attn 2 / mem-efficient / math を自動選択 (b) 既存テスト全 PASS |
| 6.1.3 | KV cache ページング — `allocate_kv_cache()` / `free_kv_cache()` 実装 | Worker | cc:完了 | (a) kv_page_size/kv_max_pages を MythosConfig に追加 (b) eviction テスト PASS |
| 6.1.4 | Sprint 6.1 テスト + コミット + push | Worker | cc:完了 | (a) test_sprint6_inference.py 19 tests (b) 305 PASS |
| 6.2.1 | WandB/MLflow/TensorBoard ロギング — `TrainLogger` 統一インターフェース | Worker | cc:完了 | (a) logger_utils.py 作成 (b) none/wandb/mlflow/tensorboard + graceful fallback |
| 6.2.2 | argparse CLI フラグ — `--resume`, `--ckpt-dir`, `--logger`, `--run-name` 等 | Worker | cc:完了 | (a) _parse_args() 追加 (b) 定数を args でオーバーライド (c) TrainLogger 統合 |
| 6.2.3 | Sprint 6.2 テスト + コミット + push | Worker | cc:完了 | (a) test_sprint6_training.py 19 tests (b) 324 PASS |
| 6.3.1 | `OpenMythosLLM` — LangChain `BaseLLM` 互換アダプタ | Worker | cc:完了 | (a) agents.py 作成 (b) run/stream/from_variant/from_pretrained (c) langchain 未インストール時 graceful fallback |
| 6.3.2 | `MythosAgent` — Swarms `Agent` 互換ラッパー | Worker | cc:完了 | (a) run/stream_run/reset/history (b) system_prompt/multi-turn context (c) swarms 未インストール時 graceful |
| 6.3.3 | Sprint 6.3 テスト + コミット + push | Worker | cc:完了 | (a) test_sprint6_agents.py 23 tests (b) 347 PASS |
| 6.4.1 | `benchmark/perplexity.py` — WikiText-2/103 PPL 測定（sliding window, GPT-2 プロトコル） | Worker | cc:完了 | (a) evaluate_perplexity() 実装 (b) _load_corpus/_tokenize_corpus 実装 (c) CLI 完成 |
| 6.4.2 | `benchmark/throughput.py` — TTFT/TPOT/throughput/ピークメモリ計測 | Worker | cc:完了 | (a) measure_latency() / sweep_batch_sizes() 実装 (b) JSON 出力対応 |
| 6.4.3 | `benchmark/lm_eval_harness.py` — lm-evaluation-harness 統合 | Worker | cc:完了 | (a) MythosLMEvalWrapper 実装 (b) loglikelihood/rolling/generate_until (c) lm-eval 未インストール時 LAMBADA standalone fallback対応 |
| 6.4.4 | Sprint 6.4 テスト + コミット + push | Worker | cc:完了 | (a) test_sprint6_benchmark.py 33 tests (b) 380 PASS |

---

## Sprint 7: 本番 Serving テスト & データパイプライン & v0.12.0 (完了)

> ブランチ: `feature/sprint7-serving-data`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 7.1.1 | serve/ 統合テスト — FastAPI endpoints / A/B router / SLA router / monitor | Worker | cc:完了 | (a) test_sprint7_serve.py 作成 (b) httpx/TestClient でエンドポイント検証 (c) 既存テスト全 PASS |
| 7.1.2 | serve/ Docker ビルド確認 & docker-compose.yml 追加 | Worker | cc:完了 | (a) docker build 成功 (b) docker-compose.yml (api + monitor) 作成 |
| 7.2.1 | data パイプライン テスト — preprocess / csv_to_jsonl / eval_perplexity / finetune | Worker | cc:完了 | (a) test_sprint7_data.py 作成 (b) 各スクリプトの主要関数を単体テスト (c) 全テスト PASS |
| 7.2.2 | HuggingFace Datasets ストリーミング統合 — `stream_dataset()` + `preprocess_stream()` | Worker | cc:完了 | (a) scripts/preprocess.py にストリーミング対応追加 (b) メモリ効率テスト追加 |
| 7.3.1 | 分散推論サポート — `DataParallel` ラッパー + `model.to_distributed()` | Worker | cc:完了 | (a) OpenMythos に to_distributed() 追加 (b) CPU でのフォールバック動作テスト (c) 既存テスト全 PASS |
| 7.4.1 | PyPI v0.12.0 リリース準備 — pyproject.toml 0.11.0→0.12.0 + CHANGELOG 更新 | Worker | cc:完了 | (a) version bump (b) CHANGELOG.md Sprint 7 セクション追加 (c) serve/scripts 依存を extras に追加 |
| 7.5.1 | Sprint 7 テスト追加 + commit + push — master へ一括コミット & GitHub push | Worker | cc:完了 | (a) 全テスト PASS (目標 420+) (b) git push origin master (c) CI green |

---

## Sprint 8: 推論品質強化 & Fine-tuning パイプライン完成 (完了)

> ブランチ: `feature/sprint8-finetune-quality`

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 8.1.1 | `scripts/finetune.py` 完成 — LoRA fine-tuning エンドツーエンド実行 | Worker | cc:完了 | (a) --lora フラグ追加 (b) enable_lora_finetuning() + trainable_parameters() 統合 (c) テスト PASS |
| 8.1.2 | `scripts/eval_perplexity.py` 完成 — fine-tuned モデルの PPL 評価 | Worker | cc:完了 | (a) --checkpoint フラグ追加 (b) checkpoint ロード対応 (c) テスト PASS |
| 8.2.1 | Serve API 品質強化 — `serve/api.py` に `/v1/chat` エンドポイント追加 | Worker | cc:完了 | (a) OpenAI 互換 `/v1/chat/completions` (b) SSE ストリーミング対応 (c) テスト PASS |
| 8.2.2 | SLA ルーター精度モード拡張 — `serve/sla_router.py` に `ultra` モード追加 | Worker | cc:完了 | (a) ultra: loops=16, budget=2000 (b) 全タスク対応 (c) 既存テスト全 PASS |
| 8.3.1 | Sprint 8 テスト追加 + commit + push | Worker | cc:完了 | (a) test_sprint8.py 22 tests (b) 468 PASS (c) git push origin master |

---

## 進行中の作業メモ

### 現在のブランチ状態 (2026-05-25 更新)
- `master`: `b72c1da` — Sprint 7 全完了 (446 PASS, CI green)
- PR #2 (`feature/inference-v2`): 2026-05-25 close 済み（master に内包）
- Sprint 8 ブランチ: `feature/sprint8-finetune-quality` (予定)

### 重要な技術的知見
- `freqs_cis` は必ず `[:T]` スライスして渡すこと (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` は `.clamp(min=1e-6)` が必要 (float32 飽和防止)
- decode_loops 2-phase: prefill=4 / decode=1 が最速 (2.54x)
- stash pop 時に Plans.md でコンフリクト発生しやすい → `git checkout stash -- Plans.md` で解決
