# OpenMythos — Sprint Plans
> 最終更新: 2026-05-29 | ブランチ規約: `feature/<sprint>-<topic>`
> Sprint 1〜9 のアーカイブ: `docs/archive/sprint-plans-1-9.md`

---

## 完了済み Sprint サマリー

| Sprint | 内容 | テスト | バージョン | commit |
|--------|------|--------|-----------|--------|
| 1 | HyperloopMythos / Inference Engine | 227 PASS | — | 6a64810 |
| 2 | Inference 高度化 (beam/quant/KV cache) | — | — | 9946ef1 |
| 3 | ドキュメント & エコシステム | — | v0.6.0 | — |
| 4 | Training 基盤 / MoDa / variants | 257 PASS | — | — |
| 5 | Training 品質 / LoRA / HF Hub / CLI | 284 PASS | v0.7.0 | — |
| 6 | 推論最適化 / TrainLogger / Agents / Benchmark | 380 PASS | — | — |
| 7 | Serving テスト / データパイプライン | 420+ PASS | v0.12.0 | — |
| 8 | Fine-tuning / /v1/chat / SLA ultra | 468 PASS | — | — |
| 9 | Marketing eval / A/B 検定 / batch API | 508 PASS | v0.13.0 | — |

---

## Sprint 10: LLMO生成 & Extended Thinking & Structured Output & v0.14.0 (完了)

> ブランチ: `feature/sprint10-llmo-thinking` → master merge 済み (PR #8)

| task-id | 説明 | 状態 |
|---------|------|------|
| 10.1.1 | `open_mythos/llmo.py` — LLMOScorer (entity_density / answer_directness / citability) | cc:完了 [ae264dd] |
| 10.1.2 | `scripts/generate_seo.py` — SEO/LLMO最適化コンテンツ生成 (3スタイル) | cc:完了 [ae264dd] |
| 10.1.3 | `serve/api.py` — `/v1/seo/score` & `/v1/seo/generate` | cc:完了 [ae264dd] |
| 10.2.1 | `open_mythos/thinking.py` — Extended Thinking (per-loop 内部状態エクスポート) | cc:完了 [ae264dd] |
| 10.2.2 | `serve/api.py` — `/v1/thinking` + ChatRequest `thinking` フラグ | cc:完了 [ae264dd] |
| 10.3.1 | `open_mythos/structured.py` — JSON mode / Structured Output | cc:完了 [ae264dd] |
| 10.3.2 | `scripts/train_dpo.py` — DPO fine-tuning | cc:完了 [ae264dd] |
| 10.4.1 | PyPI v0.14.0 | cc:完了 [ae264dd] |
| 10.5.1 | test_sprint10.py 52 tests — 560 PASS | cc:完了 [ae264dd] |

---

## Sprint 11: Tool Use / Long Context / RAG & v0.15.0 (完了)

> ブランチ: `feature/sprint11-tools-longctx-rag` → master merge 済み (PR #9)

| task-id | 説明 | 状態 |
|---------|------|------|
| 11.1.1 | `open_mythos/tools.py` — ToolRegistry / @tool / ToolCall / ToolResult | cc:完了 [d557cd5] |
| 11.1.2 | `open_mythos/tools_marketing.py` — search_competitor / calculate_roi / fetch_trend / score_content | cc:完了 [d557cd5] |
| 11.1.3 | `serve/api.py` — `/v1/tools`, `/v1/tools/call`, `/v1/tools/batch` | cc:完了 [d557cd5] |
| 11.2.1 | `open_mythos/rope_extension.py` — YaRN Dynamic NTK-aware RoPE (32K対応) | cc:完了 [d557cd5] |
| 11.2.2 | Long Context 推論テスト — extend_model_context() | cc:完了 [d557cd5] |
| 11.3.1 | `open_mythos/rag.py` — VectorStore / RAGPipeline (numpy + FAISS opt) | cc:完了 [d557cd5] |
| 11.3.2 | `serve/api.py` — `/v1/rag/index`, `/v1/rag` | cc:完了 [d557cd5] |
| 11.4.1 | PyPI v0.15.0 | cc:完了 [d557cd5] |
| 11.5.1 | test_sprint11.py 104 tests — 664 PASS | cc:完了 [d557cd5] |

---

## Sprint 12: ReAct エージェントループ & プロンプトキャッシュ & 会話メモリ & v0.16.0 (完了)

> ブランチ: `feature/sprint12-react-cache-memory`
> 戦略: A) ReAct Agent Loop B) Prompt Prefix Cache C) Conversation Memory / Session API

| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 12.1.1 | `open_mythos/react.py` — ReActAgent (Think→Act→Observe ループ + format_agent_trace) | Worker | cc:完了 | (a) AgentStep/AgentResult (b) ループ実装 (c) テスト PASS |
| 12.1.2 | `serve/api.py` に `/v1/agent/run` エンドポイント追加 | Worker | cc:完了 | (a) AgentRunRequest/Response (b) max_iterations 制御 (c) テスト PASS |
| 12.2.1 | `open_mythos/prefix_cache.py` — PromptPrefixCache (LRU prefill キャッシュ) | Worker | cc:完了 | (a) cache_prefix() (b) generate_with_cache() (c) hit_rate 統計 (d) テスト PASS |
| 12.3.1 | `open_mythos/conversation.py` — ConversationMemory + SessionStore | Worker | cc:完了 | (a) add_turn() / to_context_string() (b) 自動圧縮 (c) セッション管理 (d) テスト PASS |
| 12.3.2 | `serve/api.py` に `/v1/sessions/*` エンドポイント追加 | Worker | cc:完了 | (a) POST/GET/DELETE /v1/sessions (b) POST turns (c) GET context (d) テスト PASS |
| 12.4.1 | PyPI v0.16.0 — pyproject.toml 0.15.0→0.16.0 + CHANGELOG Sprint 12 追加 | Worker | cc:完了 | (a) version bump (b) CHANGELOG 追加 |
| 12.5.1 | Sprint 12 テスト追加 + commit + push | Worker | cc:完了 | (a) test_sprint12.py 65 tests (b) 729 PASS (c) git push |

---

## 進行中の作業メモ

### 現在のブランチ状態 (2026-05-29 更新)
- `master`: `5671840` — Sprint 1〜11 全完了 (664 PASS / v0.15.0)
- `feature/sprint12-react-cache-memory`: Sprint 12 全完了 (729 PASS / v0.16.0) — master merge 待ち

### 重要な技術的知見
- `freqs_cis` は必ず `[:T]` スライスして渡すこと (apply_rope ブロードキャストエラー防止)
- LTI `get_A()` の `log_dt + log_A` は `.clamp(min=1e-6)` が必要 (float32 飽和防止)
- decode_loops 2-phase: prefill=4 / decode=1 が最速 (2.54x)
- stash pop 時に Plans.md でコンフリクト発生しやすい → `git checkout stash -- Plans.md` で解決
