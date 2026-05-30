# OpenMythos — 実装進捗

> 最終更新: 2026-05-30 | バージョン: v0.17.0 | テスト: 440+ PASS

---

## Sprint 進捗一覧

### Sprint 1: HyperloopMythos & Inference Engine ✅ 完了
- HyperloopMythos 2-phase inference engine (prefill/decode ループ分離)
- `generate_stream()` ストリーミング生成
- `speculative_decode()` 自己投機的デコード
- top_p nucleus sampling
- decode_loops 2-phase: 2.54× decode 高速化
- **227 PASS**

### Sprint 2: Inference 高度化 ✅ 完了
- `generate_beam()` beam search
- `generate_batch()` バッチ推論
- `repetition_penalty` / `max_cache_len` (sliding window KV cache)
- INT8 / FP16 量子化 (`model.quantize()`)
- **257 PASS**

### Sprint 3: ドキュメント & エコシステム ✅ 完了
- README.md 更新（全生成メソッド記載）
- `docs/architecture.md` 作成
- PyPI パッケージ v0.6.0 準備
- GitHub Actions CI coverage report
- **257 PASS**

### Sprint 4: Training 基盤強化 ✅ 完了
- `mythos_nano` variant 追加（vocab=1024, dim=128）
- gradient checkpointing / bf16 training
- `test_training.py` / `test_moda.py` 追加
- MANIFEST.in + PyPI 公開準備
- **265 PASS**

### Sprint 5: Training 品質改善 & エコシステム拡張 ✅ 完了
- `warmup_stable_decay()` 3-phase LR スケジューラ
- `init_mup(base_dim=256)` MuP 初期化
- GNS 監視 + 早期停止 (es_patience=5)
- CLI `mythos generate` / `mythos info`
- HF Hub 連携 `push_to_hub` / `from_pretrained`
- LoRA finetuning API `enable_lora_finetuning()`
- **286 PASS**

### Sprint 6.1: 推論最適化 ✅ 完了
- `compile_model()` — torch.compile（Windows CPU は `backend="eager"` fallback）
- GQA/MLA を `F.scaled_dot_product_attention` に置き換え
- KV cache ページング `allocate_kv_cache()` / `free_kv_cache()`
- **305 PASS**

### Sprint 6.2: 訓練スクリプト高度化 ✅ 完了
- `TrainLogger` — WandB/MLflow/TensorBoard/none 統一インターフェース (`open_mythos/logger_utils.py`)
- argparse CLI フラグ17種 (`--resume`, `--ckpt-dir`, `--logger` 等)
- **324 PASS**

### Sprint 6.3: エージェント統合 ✅ 完了
- `OpenMythosLLM` — LangChain `BaseLLM` 互換アダプタ (`open_mythos/agents.py`)
- `MythosAgent` — Swarms 互換ラッパー（system_prompt + 2-turn 履歴）
- **347 PASS**

### Sprint 6.4: ベンチマーク & 評価 ✅ 完了（2026-05-24）
- `benchmark/perplexity.py` — WikiText-2/103 sliding window PPL（GPT-2 プロトコル）
- `benchmark/throughput.py` — TTFT/TPOT/throughput/ピークメモリ計測
- `benchmark/lm_eval_harness.py` — `MythosLMEvalWrapper`（lm-evaluation-harness 統合）
  - `loglikelihood` / `loglikelihood_rolling` / `generate_until` 実装
  - `lm_eval` 未インストール時は LAMBADA standalone fallback
- `tests/test_sprint6_benchmark.py` — 33テスト
- **380 PASS** / v0.11.0

### Sprint 7: サービス統合 ✅ 完了（2026-05-26）

- `serve/api.py` に `OpenMythosLLM` / `MythosAgent` を統合
- 新タスクタイプ: `seo_content`, `llmo_optimize`, `ad_copy`, `persona_message`, `market_summary`
- `_TASK_SYSTEM_PROMPTS` — E-E-A-T / LLMO / PREP法 / PASONAの法則 対応の日本語プロンプト
- `TASK_LOOPS` — タスク別推奨ループ数（llmo_optimize: 8, seo_content: 6, ad_copy: 2）
- 新エンドポイント: `POST /generate`, `GET /generate/stream`, `POST /agent`, `DELETE /agent/{session_id}`
- セッション管理: `state.agents: dict[str, MythosAgent]`
- `/health` 拡張: `active_sessions`, `endpoints` フィールド追加
- `tests/test_sprint7_serve.py` — 40テスト
- **420 PASS** / v0.12.0

### Sprint 8: GPU訓練 & ベンチマーク & Cloud Runデプロイ ✅ 完了（2026-05-30）

- `scripts/pretrain.py` — FineWeb-Edu streaming 事前学習（bf16/GC/warmup-stable-decay）
- `scripts/pretrain_gcp.sh` — GCP T4 VM での tmux 実行スクリプト
- `benchmark/run_eval.py` — perplexity + lm-eval 一括実行・JSON保存・README更新
- `benchmark/results/` — 評価結果保存ディレクトリ
- `serve/deploy_cloudrun.sh` — Artifact Registry + Cloud Run デプロイ自動化
- `serve/cloudrun.env.example` — 環境変数サンプル
- `tests/test_sprint8_pretrain.py` — 20テスト
- **440+ PASS** / v0.17.0

---

## 次回の作業候補（Sprint 9）

1. **GPU 実訓練** — GCP T4 でプレトレイン実行 → perplexity < 20 確認・結果をbenchmark/results/に保存
2. **Cloud Run 本番デプロイ** — GCPプロジェクト作成後に deploy_cloudrun.sh 実行
3. **ベンチマーク結果テーブル** — README.md に実測値を反映
3. **本番デプロイ** — Docker化 + GCP Cloud Run / Kubernetes デプロイ

---

## 重要な技術的知見

- `freqs_cis` は必ず `[:T]` スライスして渡すこと（apply_rope ブロードキャストエラー防止）
- LTI `get_A()` の `log_dt + log_A` は `.clamp(min=1e-6)` が必要（float32 飽和防止）
- decode_loops 2-phase: prefill=4 / decode=1 が最速（2.54x）
- `torch.compile` は Windows CPU で Inductor 失敗 → `backend="eager"` fallback が必要
- MuP init: `nn.Embedding` と `nn.Linear` を別処理; weight tying は `data_ptr()` で検出
- KV cache eviction は decode ステップ（T=1）のみ。prefill（T>1）では evict しない
- LangChain BaseLLM: Pydantic 非対応の `nn.Module` は `object.__setattr__` で格納
- 空プロンプト → `ids = [0]` の BOS fallback が必要（RoPE crash 防止）
- CLI tokenizer: MythosTokenizer の出力も `vocab_size - 1` にクリップ必要（nano は 1024）
