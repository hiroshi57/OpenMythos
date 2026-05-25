# Changelog

All notable changes to OpenMythos are documented here.

---

## [0.12.0] — 2026-05-25

### Sprint 7: 本番 Serving テスト & データパイプライン & 分散推論

#### serve/ 統合テスト (`tests/test_sprint7_serve.py`)

- `serve.ab_router` — ルーティング決定性・A/B 集計・スキーマ検証 (6 tests)
- `serve.sla_router` — ループ数/バジェット解決・全タスク/モード網羅・設定更新 (7 tests)
- `serve.monitor` — PSI 計算・SQLite ログ・ドリフト検知・ベースライン設定 (9 tests)
- `serve.api` — MythosConfig 構築・TASK_LOOPS 網羅・Pydantic スキーマ検証 (8 tests)

#### データパイプライン テスト (`tests/test_sprint7_data.py`)

- `scripts.preprocess` — 4タスクのプロンプトビルダー・load_jsonl・split_dataset (20 tests)
- `scripts.csv_to_jsonl` — auto_detect_mapping 自動列推定 (5 tests)
- `stream_dataset` / `preprocess_stream` — ストリーミング統合 (8 tests)

#### HuggingFace Datasets ストリーミング統合 (`scripts/preprocess.py`)

- `stream_dataset(paths, chunk_size)` — JSONL を chunk 単位でメモリ効率よく読み込む
- `preprocess_stream(paths, task, chunk_size, tokenizer, max_length)` — ストリーム前処理

#### 分散推論サポート (`open_mythos/main.py`)

- `OpenMythos.to_distributed(device_ids, output_device)` — `nn.DataParallel` ラッパー
- CPU 環境では self をそのまま返す graceful fallback

Tests: 420+ PASS (up from 380)

---

## [0.11.0] — 2026-05-24

### Sprint 6.4: ベンチマーク & 評価 (`benchmark/`)

#### Perplexity 評価 (`benchmark/perplexity.py`)

- `evaluate_perplexity(model, token_ids, device, seq_len, stride, batch_size, n_loops)` — sliding window PPL (GPT-2 プロトコル)
- `_load_corpus(dataset_name, split)` — HuggingFace `datasets` 経由で WikiText-2/103 を取得
- `_tokenize_corpus(text, vocab_size)` — MythosTokenizer + byte-level fallback + vocab clip
- CLI: `--variant`, `--checkpoint`, `--dataset`, `--split`, `--seq-len`, `--stride`, `--n-loops`, `--max-tokens`

#### スループット & レイテンシ計測 (`benchmark/throughput.py`)

- `measure_latency(model, prompt_ids, device, max_new_tokens, batch_size, n_loops, warmup_iters)` — TTFT / TPOT / throughput / peak memory を `LatencyResult` dataclass で返却
- `sweep_batch_sizes(model, prompt_ids, device, batch_sizes, ...)` — バッチサイズ一覧をスイープ
- CUDA 同期付きタイミング (`torch.cuda.synchronize`)、CPU では `psutil` RSS フォールバック
- CLI: `--batch-sizes 1,2,4`, `--output-json` オプション

#### lm-evaluation-harness 統合 (`benchmark/lm_eval_harness.py`)

- `MythosLMEvalWrapper(LM)` — EleutherAI lm-eval の `LM` インターフェース実装
- `loglikelihood`, `loglikelihood_rolling`, `generate_until` の3メソッドを実装
- `lm_eval` 未インストール時は LAMBADA standalone fallback で精度評価可能
- `lm_eval.simple_evaluate()` 経由で HellaSwag / ARC / WinoGrande 等を実行可能

Tests: 380 PASS (up from 347)

---

## [0.10.0] — 2026-05-24

### Sprint 6.3: エージェント統合 (`open_mythos/agents.py`)

#### OpenMythosLLM — LangChain 互換アダプタ

- `OpenMythosLLM(model, device, max_new_tokens, temperature, top_k, top_p)`
- `run(prompt) -> str` — LangChain 不要のスタンドアロン呼び出し
- `stream(prompt) -> Iterator[str]` — トークン単位ストリーミング
- LangChain `BaseLLM` プロトコル実装: `_generate()` / `_stream()` / `_llm_type`
- `from_variant(variant, checkpoint, device)` / `from_pretrained(repo_id_or_path)`
- `langchain-core` 未インストール時は `run()` / `stream()` のみ利用可能、パイプライン API は `ImportError`
- 空プロンプト → BOS fallback で RoPE クラッシュを回避

#### MythosAgent — Swarms 互換ラッパー

- `run(task) -> str` / `stream_run(task) -> Iterator[str]`
- `__call__(task)` — Swarms パイプラインへの drop-in 対応
- `system_prompt` + 直近 2 ターン会話履歴を自動コンテキスト構築
- `reset()` で会話履歴をクリア
- `from_variant()` / `from_pretrained()` コンストラクタ
- `swarms` 未インストール時もスタンドアロンで動作

Tests: 347 PASS (up from 324)

---

## [0.9.0] — 2026-05-24

### Sprint 6.2: 訓練スクリプト高度化

#### TrainLogger — 統一実験ロギング (`open_mythos/logger_utils.py`)

- `TrainLogger(backend, run_name, project, config, log_dir)` — WandB / MLflow / TensorBoard / none を統一 API で切り替え
- 全バックエンドで `log(metrics, step)` / `log_artifact(path)` / `finish()` を提供
- `wandb` / `mlflow` / `tensorboard` 未インストール時は警告のみで `"none"` に graceful fallback
- コンテキストマネージャ対応 (`with TrainLogger(...) as log:`)

#### argparse CLI フラグ

- `_parse_args()` 追加で全ハイパーパラメータを CLI から上書き可能に
- `--resume PATH` — 特定チェックポイントを指定再開（省略時は `--ckpt-dir` の最新を自動選択）
- `--ckpt-dir` / `--ckpt-every` / `--keep-last` — チェックポイント管理
- `--logger {none,wandb,mlflow,tensorboard}` / `--run-name` / `--project` / `--log-dir`
- `--seq-len` / `--micro-batch` / `--lr` / `--wd` / `--warmup-steps` / `--eval-every` / `--no-grad-ckpt`
- `TrainLogger` をトレーニングループに統合（train/eval メトリクスを step 単位で記録）

Tests: 324 PASS (up from 305)

---

## [0.8.0] — 2026-05-24

### Sprint 6.1: 推論最適化

#### torch.compile 統合

- `OpenMythos.compile_model(mode, fullgraph, dynamic, backend)` — `TransformerBlock` / `RecurrentBlock` を個別にコンパイル（再帰ループのグラフ断絶を回避）
- `backend="eager"` にフォールバックして Windows CPU でも動作
- `PyTorch < 2.0` では自動スキップ

#### SDPA 最適化（GQA / MLA）

- GQA / MLA の手動 attention を `F.scaled_dot_product_attention` に置き換え
- Flash Attention 2 / memory-efficient / math backend を自動選択
- `flash_attn` 未インストール環境でも高速化（PyTorch 2.x SDPA カーネル）

#### KV cache ページング

- `MythosConfig` に `kv_page_size: int = 64` / `kv_max_pages: int = 0` を追加
- `OpenMythos.allocate_kv_cache(max_seq_len, device)` — ページ上限付きキャッシュ辞書を生成
- `OpenMythos.free_kv_cache(cache)` — レイヤーテンソルを解放してメタキーを保持（再利用可能）
- decode ステップで window 上限超過時に古いトークンを evict

Tests: 305 PASS (up from 286)

---

## [0.7.0] — 2026-05-24

### Sprint 5: Training 品質改善 & エコシステム拡張

#### Training quality

- `warmup_stable_decay(step, warmup, stable_end, total, max_lr, min_lr)` — 3-phase LR schedule (linear warmup → flat plateau → cosine decay, DeepSeek-V3 style)
- `OpenMythos.init_mup(base_dim=256)` — Maximal Update Parametrization; optimal LR transfers across model widths
- Gradient Noise Scale (GNS) monitoring in training loop: `gns = grad_norm² / micro_batch`, rolling window logged per step
- Early stopping: `es_patience=5`, `es_min_delta=1e-3`; halts after consecutive eval steps without improvement

#### CLI

- `mythos generate` — autoregressive generation from prompt (supports `--checkpoint`, `--variant`, `--stream`, `--temperature`, `--top-k`, `--top-p`, `--device`)
- `mythos info` — print param count and config summary for any variant
- Registered as `mythos` console script in `pyproject.toml`

#### Hugging Face Hub integration

- `OpenMythos.push_to_hub(repo_id, token)` — upload weights + config to the Hub
- `OpenMythos.from_pretrained(repo_id_or_path, token, map_location)` — load from local path, directory, or Hub repo

#### LoRA finetuning API

- `OpenMythos.enable_lora_finetuning()` — freeze all params except `LoRAAdapter` weights; returns `self` for chaining
- `OpenMythos.trainable_parameters()` — yield only grad-requiring params (for optimizer construction)

Tests: 286 PASS (up from 265)

---

## [0.6.0] — 2026-05-23

### Sprint 2: Inference 高度化

#### New generation methods

- `generate_beam()` — beam search with configurable `beam_width` and `length_penalty` (B=1)
- `generate_batch()` — parallel generation for variable-length prompt lists via left-padding
- `repetition_penalty` parameter added to `generate()` and `generate_stream()`
- `max_cache_len` (sliding window KV cache) parameter added to `generate()` and `generate_stream()`

#### Quantization

- `model.quantize("fp16")` — in-place FP16 cast; dtype-safe through ACT, LTI, and MoEFFN
- `model.quantize("int8")` — dynamic INT8 quantization of all `nn.Linear` layers

#### Internal fixes

- `MoEFFN.forward`: `weight` and `expert_out` now cast to `flat.dtype` for FP16/BF16 safety
- `RecurrentBlock`: `cumulative_p` and boolean masks cast to `h.dtype`
- `LTIInjection.forward`: `A` and `B` cast to `h.dtype`
- `ACTHalting.forward`: output cast to `h.dtype`

Tests: 257 PASS (up from 227)

---

## [0.5.0] — 2026-05-21

### Sprint 1: HyperloopMythos & Inference Engine

- `HyperloopMythos` two-phase inference engine (prefill `n_loops`, decode `decode_loops`)
- `generate_stream()` — streaming autoregressive generation
- `speculative_decode()` — self-speculative decoding with depth-based draft/target split
- `top_p` nucleus sampling in `_sample_token`
- `decode_loops` two-phase depth strategy: 2.54× decode speedup
- `_causal_mask` cache_len fix for incremental decoding
- LTI `get_A()` `.clamp(min=1e-6)` fix for float32 saturation
- `freqs_cis[:T]` slice fix for apply_rope broadcast errors
- Lint: ruff/black pass, `.gitattributes` CRLF/LF unification

Tests: 227 PASS

---

## [0.1.0] — 2026-05-01

Initial release: OpenMythos RDT Phase 0–4

- `OpenMythos` model with GQA / MLA attention
- `MoEFFN` fine-grained mixture of experts
- `LTIInjection` with spectral radius < 1 guarantee
- `ACTHalting` adaptive computation time
- `LoRAAdapter` depth-wise LoRA
- `RecurrentBlock` with loop index embedding
- `HyperloopMythos` v0.1 (basic)
- Pre-configured variants `mythos_1b` … `mythos_1t`
- `MythosTokenizer`
- Training script `training/3b_fine_web_edu.py`
