"""
serve/routers/skills_integrations.py — スキル統合ドメイン API (Sprint 44〜52)

Vector Store + Instructor 構造化抽出 / HuggingFace Hub / 推論バックエンド /
研究ツール / マルチモーダル / 訓練最適化 / エージェントフレームワーク /
データ・検索ツール / DevOps・クラウド統合。
serve/api.py のモノリスから分割 (認証は app 全体の verify_api_key に委譲)。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from serve.auth import verify_api_key

router = APIRouter()

# ============================================================================
# Sprint 44 — Vector Store API  +  Instructor 構造化抽出 API
# ============================================================================

from open_mythos.skills.vector_store import (  # noqa: E402
    VectorDocument as _VectorDocument,
    VectorStoreConfig as _VectorStoreConfig,
    ChromaStore as _ChromaStore,
    QdrantStore as _QdrantStore,
    PineconeStore as _PineconeStore,
    FaissStore as _FaissStore,
)
from open_mythos.skills.instructor_extract import (  # noqa: E402
    ExtractionSchema as _ExtractionSchema,
    InstructorExtractor as _InstructorExtractor,
)
# ── ストアレジストリ (backend, collection) → store instance ─────────────────
_vs_registry: dict = {}


def _get_or_create_store(backend: str, collection: str, dim: int = 768):
    """バックエンドとコレクション名でストアを取得または生成する。"""
    key = (backend, collection)
    if key not in _vs_registry:
        cfg = _VectorStoreConfig(collection=collection, dim=dim)
        if backend == "faiss":
            _vs_registry[key] = _FaissStore(cfg)
        elif backend == "chroma":
            _vs_registry[key] = _ChromaStore(cfg)
        elif backend == "qdrant":
            _vs_registry[key] = _QdrantStore(cfg)
        elif backend == "pinecone":
            _vs_registry[key] = _PineconeStore(cfg)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
    return _vs_registry[key]


# ── リクエストモデル ────────────────────────────────────────────────────────────

class _VsDocInput(BaseModel):
    id: str = ""
    vector: List[float]
    text: str = ""
    metadata: dict = Field(default_factory=dict)


class _VsUpsertRequest(BaseModel):
    backend: str = "faiss"
    collection: str = "default"
    documents: List[_VsDocInput]


class _VsQueryRequest(BaseModel):
    backend: str = "faiss"
    collection: str = "default"
    vector: List[float]
    top_k: int = 5
    filter: Optional[dict] = None


class _VsDeleteRequest(BaseModel):
    backend: str = "faiss"
    collection: str = "default"
    ids: List[str]


class _FieldDef(BaseModel):
    name: str
    type: str = "str"


class _ExtractRequest(BaseModel):
    text: str = ""
    schema_name: str
    fields: List[_FieldDef]
    max_retries: int = 3


# ── Vector Store エンドポイント ─────────────────────────────────────────────────

@router.post(
    "/v1/vector-store/upsert",
    tags=["vector-store"],
    summary="Vector Store — ドキュメント upsert (Sprint 44)",
    dependencies=[Depends(verify_api_key)],
)
def vs_upsert(req: _VsUpsertRequest):
    """指定バックエンド・コレクションへドキュメントを挿入/更新する。"""
    store = _get_or_create_store(req.backend, req.collection)
    docs = [
        _VectorDocument(id=d.id, vector=d.vector, text=d.text, metadata=d.metadata)
        for d in req.documents
    ]
    count = store.upsert(docs)
    return {"upserted": count, "backend": req.backend, "collection": req.collection}


@router.post(
    "/v1/vector-store/query",
    tags=["vector-store"],
    summary="Vector Store — 近傍検索 (Sprint 44)",
    dependencies=[Depends(verify_api_key)],
)
def vs_query(req: _VsQueryRequest):
    """クエリベクターに近い上位 top_k ドキュメントを返す。"""
    store = _get_or_create_store(req.backend, req.collection)
    results = store.query(req.vector, top_k=req.top_k, filter=req.filter)
    hits = [
        {
            "id":       doc.id,
            "text":     doc.text,
            "metadata": doc.metadata,
            "score":    float(score),
        }
        for doc, score in results
    ]
    return {"hits": hits, "backend": req.backend, "collection": req.collection}


@router.post(
    "/v1/vector-store/delete",
    tags=["vector-store"],
    summary="Vector Store — ドキュメント削除 (Sprint 44)",
    dependencies=[Depends(verify_api_key)],
)
def vs_delete(req: _VsDeleteRequest):
    """指定 ID のドキュメントを削除する。"""
    store = _get_or_create_store(req.backend, req.collection)
    deleted = store.delete(req.ids)
    return {"deleted": deleted, "backend": req.backend, "collection": req.collection}


@router.get(
    "/v1/vector-store/count",
    tags=["vector-store"],
    summary="Vector Store — ドキュメント数取得 (Sprint 44)",
    dependencies=[Depends(verify_api_key)],
)
def vs_count(backend: str = "faiss", collection: str = "default"):
    """コレクションのドキュメント総数を返す。"""
    store = _get_or_create_store(backend, collection)
    return {"count": store.count(), "backend": backend, "collection": collection}


# ── Instructor 構造化抽出エンドポイント ─────────────────────────────────────────

@router.post(
    "/v1/extract",
    tags=["instructor"],
    summary="Instructor 構造化抽出 (Sprint 44)",
    dependencies=[Depends(verify_api_key)],
)
def extract(req: _ExtractRequest):
    """テキストからスキーマ定義に従って JSON を抽出する。"""
    schema = _ExtractionSchema(
        name=req.schema_name,
        fields={f.name: f.type for f in req.fields},
    )
    extractor = _InstructorExtractor(max_retries=req.max_retries)
    result = extractor.extract(req.text, schema)
    return {
        "success":     result.success,
        "data":        result.data,
        "schema_name": result.schema_name,
        "error":       result.error,
        "retries":     result.retries,
    }


@router.post(
    "/v1/extract/prompt",
    tags=["instructor"],
    summary="Instructor 構造化抽出プロンプト生成 (Sprint 44)",
    dependencies=[Depends(verify_api_key)],
)
def extract_prompt(req: _ExtractRequest):
    """スキーマから LLM 向け抽出プロンプトを生成して返す。"""
    schema = _ExtractionSchema(
        name=req.schema_name,
        fields={f.name: f.type for f in req.fields},
    )
    extractor = _InstructorExtractor()
    prompt = extractor.build_prompt(schema)
    return {
        "prompt":      prompt,
        "schema":      schema.to_json_schema(),
        "schema_name": req.schema_name,
    }


# ============================================================================
# Sprint 45 — HuggingFace Hub 統合 API
# ============================================================================

from open_mythos.skills.hf_hub import (  # noqa: E402
    HFHubClient as _HFHubClient,
    FastTokenizer as _FastTokenizer,
    LoRAConfig as _LoRAConfig,
    PEFTAdapter as _PEFTAdapter,
    EvalTask as _EvalTask,
    LMEvaluator as _LMEvaluator,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────

class _HFSearchModelsRequest(BaseModel):
    query: str
    task: str = ""
    limit: int = 10


class _HFSearchDatasetsRequest(BaseModel):
    query: str
    limit: int = 10


class _TokenizeRequest(BaseModel):
    text: str
    model: str = "gpt2"
    max_length: Optional[int] = None
    truncation: bool = False


class _PEFTEstimateRequest(BaseModel):
    total_params: int
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class _LMEvalRequest(BaseModel):
    tasks: List[str]
    model: str = "mock"
    batch_size: int = 1


# ── エンドポイント ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/hf/search/models",
    tags=["huggingface"],
    summary="HuggingFace Hub — モデル検索 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def hf_search_models(req: _HFSearchModelsRequest):
    """クエリに一致する HF Hub モデルを検索する。"""
    client = _HFHubClient()
    results = client.search_models(req.query, task=req.task, limit=req.limit)
    return {
        "results": [
            {
                "model_id": r.model_id,
                "task":     r.task,
                "downloads": r.downloads,
                "likes":    r.likes,
                "tags":     r.tags,
                "private":  r.private,
            }
            for r in results
        ],
        "query": req.query,
        "count": len(results),
    }


@router.post(
    "/v1/hf/search/datasets",
    tags=["huggingface"],
    summary="HuggingFace Hub — データセット検索 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def hf_search_datasets(req: _HFSearchDatasetsRequest):
    """クエリに一致する HF Hub データセットを検索する。"""
    client = _HFHubClient()
    results = client.search_datasets(req.query, limit=req.limit)
    return {
        "results": [
            {
                "dataset_id":       r.dataset_id,
                "task_categories":  r.task_categories,
                "downloads":        r.downloads,
            }
            for r in results
        ],
        "query": req.query,
        "count": len(results),
    }


@router.get(
    "/v1/hf/model/{model_id:path}",
    tags=["huggingface"],
    summary="HuggingFace Hub — モデル情報取得 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def hf_model_info(model_id: str):
    """指定 model_id のモデル情報を返す。"""
    client = _HFHubClient()
    info = client.get_model_info(model_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id}")
    return {
        "model_id": info.model_id,
        "task":     info.task,
        "downloads": info.downloads,
        "likes":    info.likes,
        "tags":     info.tags,
        "private":  info.private,
    }


@router.post(
    "/v1/tokenize",
    tags=["huggingface"],
    summary="FastTokenizer — テキストをトークン化 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def tokenize(req: _TokenizeRequest):
    """テキストをトークン ID に変換する。"""
    tok = _FastTokenizer(req.model)
    tok._native = False  # テスト環境でも安全にフォールバックへ
    result = tok.encode(req.text, max_length=req.max_length, truncation=req.truncation)
    return {
        "tokens":    result.tokens,
        "n_tokens":  result.n_tokens,
        "truncated": result.truncated,
        "model":     req.model,
    }


@router.post(
    "/v1/peft/estimate",
    tags=["peft"],
    summary="PEFT/LoRA — 訓練可能パラメータ概算 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def peft_estimate(req: _PEFTEstimateRequest):
    """LoRA 設定に基づいて訓練可能パラメータ数を概算する。"""
    cfg = _LoRAConfig(
        r=req.r,
        lora_alpha=req.lora_alpha,
        target_modules=req.target_modules,
    )
    adapter = _PEFTAdapter(cfg)
    est = adapter.estimate_trainable_params(req.total_params)
    return est


@router.post(
    "/v1/lm-eval",
    tags=["lm-eval"],
    summary="LM Evaluation Harness — モデル評価 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def lm_eval(req: _LMEvalRequest):
    """指定タスクでモデルを評価し、スコアを返す。"""
    evaluator = _LMEvaluator(req.model)
    tasks = [_EvalTask(name=t) for t in req.tasks]
    results = evaluator.evaluate(tasks, batch_size=req.batch_size)
    return {
        "results": [
            {"task": r.task, "metric": r.metric, "value": r.value, "stderr": r.stderr}
            for r in results
        ],
        "model":  req.model,
        "count":  len(results),
    }


@router.get(
    "/v1/lm-eval/tasks",
    tags=["lm-eval"],
    summary="LM Evaluation Harness — 利用可能タスク一覧 (Sprint 45)",
    dependencies=[Depends(verify_api_key)],
)
def lm_eval_list_tasks():
    """利用可能な評価タスクの一覧を返す。"""
    evaluator = _LMEvaluator()
    return {"tasks": evaluator.list_tasks()}


# ============================================================================
# Sprint 46 — 推論バックエンド統合 API
# ============================================================================

from open_mythos.skills.inference_backends import (  # noqa: E402
    AttentionConfig as _AttentionConfig,
    FlashAttentionOptimizer as _FlashAttentionOptimizer,
    GuidanceTemplate as _GuidanceTemplate,
    GuidanceGenerator as _GuidanceGenerator,
    TRTConfig as _TRTConfig,
    TRTLLMBackend as _TRTLLMBackend,
    WhisperTranscriber as _WhisperTranscriber,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────

class _AttentionBenchmarkRequest(BaseModel):
    seq_len: int = 512
    batch_size: int = 1
    backend: str = "auto"


class _GuidanceGenerateRequest(BaseModel):
    template: str
    variables: dict = Field(default_factory=dict)
    max_tokens: int = 256


class _RegexGrammarRequest(BaseModel):
    pattern: str


class _ChoiceGrammarRequest(BaseModel):
    choices: List[str]


class _TRTGenerateRequest(BaseModel):
    input_ids: List[List[int]]
    max_new_tokens: int = 64
    temperature: float = 1.0


class _WhisperTranscribeRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None
    task: str = "transcribe"
    model: str = "base"


# ── エンドポイント ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/attention/benchmark",
    tags=["inference"],
    summary="Flash Attention — バックエンド選択とベンチマーク (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def attention_benchmark(req: _AttentionBenchmarkRequest):
    """利用可能な最速アテンション実装を検出し、ベンチマークを実行する。"""
    cfg = _AttentionConfig(backend=req.backend)
    opt = _FlashAttentionOptimizer(cfg)
    bench = opt.benchmark(seq_len=req.seq_len, batch_size=req.batch_size)
    return {
        "backend":         bench.backend,
        "seq_len":         bench.seq_len,
        "batch_size":      bench.batch_size,
        "latency_ms":      bench.latency_ms,
        "memory_mb":       bench.memory_mb,
        "speedup_vs_eager": bench.speedup_vs_eager,
    }


@router.post(
    "/v1/guidance/generate",
    tags=["guidance"],
    summary="Guidance — 制約付きテキスト生成 (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def guidance_generate(req: _GuidanceGenerateRequest):
    """テンプレートと変数から制約付きテキストを生成する。"""
    gen = _GuidanceGenerator()
    tpl = _GuidanceTemplate(
        template=req.template,
        variables=req.variables,
        max_tokens=req.max_tokens,
    )
    result = gen.generate(tpl)
    return {
        "text":        result.text,
        "variables":   result.variables,
        "tokens_used": result.tokens_used,
        "success":     result.success,
    }


@router.post(
    "/v1/guidance/grammar/regex",
    tags=["guidance"],
    summary="Guidance — 正規表現 grammar 生成 (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def guidance_grammar_regex(req: _RegexGrammarRequest):
    """正規表現パターンから Guidance grammar 文字列を生成する。"""
    gen = _GuidanceGenerator()
    return {"grammar": gen.build_regex_grammar(req.pattern), "pattern": req.pattern}


@router.post(
    "/v1/guidance/grammar/choice",
    tags=["guidance"],
    summary="Guidance — 選択肢 grammar 生成 (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def guidance_grammar_choice(req: _ChoiceGrammarRequest):
    """選択肢リストから Guidance select grammar を生成する。"""
    gen = _GuidanceGenerator()
    return {"grammar": gen.build_choice_grammar(req.choices), "choices": req.choices}


@router.post(
    "/v1/trt/generate",
    tags=["tensorrt-llm"],
    summary="TensorRT-LLM — バッチ推論 (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def trt_generate(req: _TRTGenerateRequest):
    """TensorRT-LLM バックエンドでバッチ推論を実行する。"""
    backend = _TRTLLMBackend(_TRTConfig())
    result = backend.generate(req.input_ids, max_new_tokens=req.max_new_tokens,
                              temperature=req.temperature)
    return {
        "texts":      result.texts,
        "output_ids": result.output_ids,
        "latency_ms": result.latency_ms,
    }


@router.post(
    "/v1/whisper/transcribe",
    tags=["whisper"],
    summary="Whisper — 音声書き起こし (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def whisper_transcribe(req: _WhisperTranscribeRequest):
    """音声ファイルをテキストに書き起こす。"""
    w = _WhisperTranscriber(model_name=req.model)
    result = w.transcribe(req.audio_path, language=req.language, task=req.task)
    return {
        "text":      result.text,
        "language":  result.language,
        "segments":  result.segments,
        "duration_s": result.duration_s,
        "model":     result.model,
    }


@router.post(
    "/v1/whisper/detect-language",
    tags=["whisper"],
    summary="Whisper — 言語検出 (Sprint 46)",
    dependencies=[Depends(verify_api_key)],
)
def whisper_detect_language(req: _WhisperTranscribeRequest):
    """音声ファイルの言語を検出し、確率マップを返す。"""
    w = _WhisperTranscriber(model_name=req.model)
    probs = w.detect_language(req.audio_path)
    return {"probabilities": probs, "audio_path": req.audio_path}


# ============================================================================
# Sprint 47 — 研究ツール統合 API
# ============================================================================

from open_mythos.skills.research_tools import (  # noqa: E402
    ArxivSearcher as _ArxivSearcher,
    DSPySignature as _DSPySignature,
    DSPyOptimizer as _DSPyOptimizer,
    WebSearcher as _WebSearcher,
    JupyterKernelClient as _JupyterKernelClient,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────

class _ArxivSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    category: str = ""


class _DSPyPredictRequest(BaseModel):
    signature_name: str
    inputs: dict
    outputs: dict
    instructions: str = ""


class _WebSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    region: str = "wt-wt"


class _JupyterExecuteRequest(BaseModel):
    code: str
    timeout: float = 30.0


# ── エンドポイント ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/arxiv/search",
    tags=["research"],
    summary="arXiv — 論文検索 (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def arxiv_search(req: _ArxivSearchRequest):
    """キーワードで arXiv 論文を検索する。"""
    searcher = _ArxivSearcher(max_results=req.max_results)
    papers = searcher.search(req.query, max_results=req.max_results, category=req.category)
    return {
        "papers": [
            {
                "arxiv_id":   p.arxiv_id,
                "title":      p.title,
                "authors":    p.authors,
                "abstract":   p.abstract,
                "categories": p.categories,
                "published":  p.published,
                "url":        p.url,
            }
            for p in papers
        ],
        "query": req.query,
        "count": len(papers),
    }


@router.get(
    "/v1/arxiv/paper/{arxiv_id:path}",
    tags=["research"],
    summary="arXiv — ID 指定で論文取得 (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def arxiv_get_paper(arxiv_id: str):
    """arXiv ID から論文情報を取得する。"""
    searcher = _ArxivSearcher()
    paper = searcher.get_by_id(arxiv_id)
    if paper is None:
        return {"paper": None, "arxiv_id": arxiv_id}
    return {
        "paper": {
            "arxiv_id":   paper.arxiv_id,
            "title":      paper.title,
            "authors":    paper.authors,
            "abstract":   paper.abstract,
            "url":        paper.url,
        },
        "arxiv_id": arxiv_id,
    }


@router.post(
    "/v1/dspy/predict",
    tags=["dspy"],
    summary="DSPy — プログラマブルプロンプト予測 (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def dspy_predict(req: _DSPyPredictRequest):
    """DSPy シグネチャに従って予測を実行する。"""
    sig = _DSPySignature(
        name=req.signature_name,
        inputs={k: str(v) for k, v in req.inputs.items()},
        outputs={k: str(v) for k, v in req.outputs.items()},
        instructions=req.instructions,
    )
    opt = _DSPyOptimizer()
    pred = opt.predict(sig, req.inputs)
    return {
        "outputs": pred.outputs,
        "rationale": pred.rationale,
        "success": pred.success,
    }


@router.post(
    "/v1/dspy/chain-of-thought",
    tags=["dspy"],
    summary="DSPy — Chain-of-Thought プロンプト生成 (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def dspy_chain_of_thought(req: _DSPyPredictRequest):
    """シグネチャから Chain-of-Thought プロンプトを生成する。"""
    sig = _DSPySignature(
        name=req.signature_name,
        inputs={k: str(v) for k, v in req.inputs.items()},
        outputs={k: str(v) for k, v in req.outputs.items()},
        instructions=req.instructions,
    )
    opt = _DSPyOptimizer()
    prompt = opt.build_chain_of_thought(sig)
    return {"prompt": prompt, "signature_name": req.signature_name}


@router.post(
    "/v1/search/web",
    tags=["search"],
    summary="Web 検索 (DuckDuckGo fallback) (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def search_web(req: _WebSearchRequest):
    """Web を検索し、結果リストを返す。"""
    searcher = _WebSearcher(max_results=req.max_results)
    results = searcher.search(req.query, region=req.region, max_results=req.max_results)
    return {
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
            for r in results
        ],
        "query": req.query,
        "count": len(results),
    }


@router.post(
    "/v1/search/news",
    tags=["search"],
    summary="ニュース検索 (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def search_news(req: _WebSearchRequest):
    """ニュースを検索し、結果リストを返す。"""
    searcher = _WebSearcher(max_results=req.max_results)
    results = searcher.news(req.query, max_results=req.max_results)
    return {
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
            for r in results
        ],
        "query": req.query,
        "count": len(results),
    }


@router.post(
    "/v1/jupyter/execute",
    tags=["jupyter"],
    summary="Jupyter — コード実行 (Sprint 47)",
    dependencies=[Depends(verify_api_key)],
)
def jupyter_execute(req: _JupyterExecuteRequest):
    """Python コードを Jupyter カーネルで実行する (in-process fallback)。"""
    jk = _JupyterKernelClient()
    jk._native = False  # サーバー環境では exec fallback を使用
    result = jk.execute(req.code, timeout=req.timeout)
    return {
        "stdout":          result.stdout,
        "stderr":          result.stderr,
        "outputs":         result.outputs,
        "execution_count": result.execution_count,
        "success":         result.success,
        "error_name":      result.error_name,
    }


# ============================================================================
# Sprint 48 — マルチモーダル統合 API
# ============================================================================

from open_mythos.skills.multimodal import (  # noqa: E402
    CLIPModel as _CLIPModel,
    VisionChatMessage as _VisionChatMessage,
    LLaVAModel as _LLaVAModel,
    DiffusionRequest as _DiffusionRequest,
    StableDiffusionGenerator as _StableDiffusionGenerator,
    SegmentRequest as _SegmentRequest,
    SAMSegmenter as _SAMSegmenter,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────

class _CLIPEncodeTextRequest(BaseModel):
    texts: List[str]
    model: str = "openai/clip-vit-base-patch32"


class _CLIPEncodeImageRequest(BaseModel):
    image_b64: str
    model: str = "openai/clip-vit-base-patch32"


class _CLIPClassifyRequest(BaseModel):
    image_b64: str
    labels: List[str]
    model: str = "openai/clip-vit-base-patch32"


class _VisionChatMessageInput(BaseModel):
    role: str = "user"
    text: str
    image_b64: Optional[str] = None


class _LLaVAChatRequest(BaseModel):
    messages: List[_VisionChatMessageInput]
    max_new_tokens: int = 256
    model: str = "llava-hf/llava-1.5-7b-hf"


class _DiffusionGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1


class _SAMSegmentRequest(BaseModel):
    image_b64: str
    points: List[List[int]] = Field(default_factory=list)
    boxes: List[List[int]] = Field(default_factory=list)
    multimask: bool = True


# ── エンドポイント ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/clip/encode/text",
    tags=["multimodal"],
    summary="CLIP — テキスト埋め込み (Sprint 48)",
    dependencies=[Depends(verify_api_key)],
)
def clip_encode_text(req: _CLIPEncodeTextRequest):
    """テキストリストを CLIP ベクトルに変換する。"""
    clip = _CLIPModel(req.model)
    clip._native = False
    embs = clip.encode_text(req.texts)
    return {
        "embeddings": [
            {"vector": e.vector, "modality": e.modality, "dim": e.dim}
            for e in embs
        ],
        "count": len(embs),
    }


@router.post(
    "/v1/clip/encode/image",
    tags=["multimodal"],
    summary="CLIP — 画像埋め込み (Sprint 48)",
    dependencies=[Depends(verify_api_key)],
)
def clip_encode_image(req: _CLIPEncodeImageRequest):
    """Base64 画像を CLIP ベクトルに変換する。"""
    clip = _CLIPModel(req.model)
    clip._native = False
    emb = clip.encode_image_b64(req.image_b64)
    return {
        "embedding": {"vector": emb.vector, "modality": emb.modality, "dim": emb.dim}
    }


# ============================================================================
# Sprint 49 — 訓練最適化統合 API
# ============================================================================

from open_mythos.skills.training_optimization import (  # noqa: E402
    LightningTrainerConfig as _LightningConfig,
    LightningTrainer as _LightningTrainer,
    FSDPConfig as _FSDPConfig,
    FSDPWrapper as _FSDPWrapper,
    SimPOConfig as _SimPOConfig,
    SimPOTrainer as _SimPOTrainer,
    SAEConfig as _SAEConfig,
    SparseAutoencoder as _SparseAutoencoder,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────

class _LightningFitRequest(BaseModel):
    max_epochs: int = 3
    accelerator: str = "auto"
    precision: str = "32-true"
    gradient_clip_val: float = 1.0


class _FSDPEstimateRequest(BaseModel):
    total_params: int
    world_size: int = 1
    sharding_strategy: str = "FULL_SHARD"


class _SimPOLossRequest(BaseModel):
    chosen_logprobs: List[float]
    rejected_logprobs: List[float]
    beta: float = 2.0
    gamma: float = 0.5


class _SAEForwardRequest(BaseModel):
    activations: List[List[float]]
    d_in: int = 512
    d_sae: int = 2048
    k: int = 32


class _SAEEstimateRequest(BaseModel):
    model_dim: int
    expansion: int = 4


# ── エンドポイント ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/training/lightning/fit",
    tags=["training"],
    summary="PyTorch Lightning — 訓練実行 (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def lightning_fit(req: _LightningFitRequest):
    """Lightning トレーナーでモデルを訓練する (fallback)。"""
    cfg = _LightningConfig(
        max_epochs=req.max_epochs,
        accelerator=req.accelerator,
        precision=req.precision,
        gradient_clip_val=req.gradient_clip_val,
    )
    tr = _LightningTrainer(cfg)
    result = tr.fit(model=None)
    return {
        "final_loss":     result.final_loss,
        "epochs_trained": result.epochs_trained,
        "best_val_loss":  result.best_val_loss,
        "train_time_s":   result.train_time_s,
    }


@router.post(
    "/v1/training/lightning/config",
    tags=["training"],
    summary="PyTorch Lightning — 設定取得 (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def lightning_config(req: _LightningFitRequest):
    """Lightning トレーナー設定辞書を返す。"""
    cfg = _LightningConfig(
        max_epochs=req.max_epochs,
        accelerator=req.accelerator,
        precision=req.precision,
    )
    return _LightningTrainer(cfg).build_config_dict()


@router.post(
    "/v1/training/fsdp/estimate",
    tags=["training"],
    summary="FSDP — メモリ使用量推定 (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def fsdp_estimate(req: _FSDPEstimateRequest):
    """FSDP シャーディング時のメモリ使用量を推定する。"""
    cfg = _FSDPConfig(sharding_strategy=req.sharding_strategy)
    fw = _FSDPWrapper(cfg)
    info = fw.estimate_memory(req.total_params, world_size=req.world_size)
    return {
        "shard_count":          info.shard_count,
        "local_params":         info.local_params,
        "total_params":         info.total_params,
        "memory_per_shard_mb":  info.memory_per_shard_mb,
    }


@router.post(
    "/v1/training/simpo/compute-loss",
    tags=["training"],
    summary="SimPO — 損失計算 (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def simpo_compute_loss(req: _SimPOLossRequest):
    """SimPO 損失を計算する。"""
    cfg = _SimPOConfig(beta=req.beta, gamma=req.gamma)
    tr = _SimPOTrainer(cfg)
    loss = tr.compute_loss(req.chosen_logprobs, req.rejected_logprobs)
    return {"loss": loss, "beta": req.beta, "gamma": req.gamma}


@router.post(
    "/v1/training/simpo/train-step",
    tags=["training"],
    summary="SimPO — 1 ステップ訓練 (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def simpo_train_step(req: _SimPOLossRequest):
    """SimPO の 1 ステップを実行し、報酬統計を返す。"""
    cfg = _SimPOConfig(beta=req.beta, gamma=req.gamma)
    tr = _SimPOTrainer(cfg)
    result = tr.train_step(req.chosen_logprobs, req.rejected_logprobs)
    return result


@router.post(
    "/v1/training/sae/forward",
    tags=["training"],
    summary="Sparse Autoencoder — forward pass (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def sae_forward(req: _SAEForwardRequest):
    """SAE の forward を実行し、再構成損失とスパース度を返す。"""
    import torch as _torch
    cfg = _SAEConfig(d_in=req.d_in, d_sae=req.d_sae, k=req.k)
    sae = _SparseAutoencoder(cfg)
    x = _torch.tensor(req.activations, dtype=_torch.float32)
    result = sae.forward(x)
    return {
        "recon_loss":       result.get("recon_loss", 0.0),
        "l0_sparsity":      result.get("l0_sparsity", 0.0),
        "l1_sparsity":      result.get("l1_sparsity", 0.0),
    }


@router.post(
    "/v1/training/sae/estimate-config",
    tags=["training"],
    summary="Sparse Autoencoder — 推奨設定生成 (Sprint 49)",
    dependencies=[Depends(verify_api_key)],
)
def sae_estimate_config(req: _SAEEstimateRequest):
    """モデル次元から SAE 推奨設定を生成する。"""
    sae = _SparseAutoencoder(_SAEConfig())
    cfg = sae.estimate_config(req.model_dim, expansion=req.expansion)
    return {"d_in": cfg.d_in, "d_sae": cfg.d_sae, "k": cfg.k}


@router.post(
    "/v1/clip/classify",
    tags=["multimodal"],
    summary="CLIP — ゼロショット画像分類 (Sprint 48)",
    dependencies=[Depends(verify_api_key)],
)
def clip_classify(req: _CLIPClassifyRequest):
    """画像をラベルリストでゼロショット分類する。"""
    clip = _CLIPModel(req.model)
    clip._native = False
    scores = clip.zero_shot_classify(req.image_b64, req.labels)
    return {"scores": scores, "labels": req.labels}


@router.post(
    "/v1/llava/chat",
    tags=["multimodal"],
    summary="LLaVA — 視覚言語チャット (Sprint 48)",
    dependencies=[Depends(verify_api_key)],
)
def llava_chat(req: _LLaVAChatRequest):
    """画像とテキストを含む会話に応答する。"""
    model = _LLaVAModel(req.model)
    model._native = False
    messages = [
        _VisionChatMessage(role=m.role, text=m.text, image_b64=m.image_b64)
        for m in req.messages
    ]
    result = model.chat(messages, max_new_tokens=req.max_new_tokens)
    return {
        "response":    result.response,
        "model":       result.model,
        "tokens_used": result.tokens_used,
    }


@router.post(
    "/v1/diffusion/generate",
    tags=["multimodal"],
    summary="Stable Diffusion — 画像生成 (Sprint 48)",
    dependencies=[Depends(verify_api_key)],
)
def diffusion_generate(req: _DiffusionGenerateRequest):
    """テキストプロンプトから画像を生成する。"""
    gen = _StableDiffusionGenerator()
    gen._native = False
    diff_req = _DiffusionRequest(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
    )
    result = gen.generate(diff_req)
    return {
        "image_b64": result.image_b64,
        "prompt":    result.prompt,
        "seed":      result.seed,
        "steps":     result.steps,
        "width":     result.width,
        "height":    result.height,
    }


@router.post(
    "/v1/sam/segment",
    tags=["multimodal"],
    summary="SAM — セグメンテーション (Sprint 48)",
    dependencies=[Depends(verify_api_key)],
)
def sam_segment(req: _SAMSegmentRequest):
    """画像をセグメンテーションし、マスクリストを返す。"""
    sam = _SAMSegmenter()
    seg_req = _SegmentRequest(
        image_b64=req.image_b64,
        points=[tuple(p) for p in req.points],  # type: ignore
        boxes=[tuple(b) for b in req.boxes],     # type: ignore
        multimask=req.multimask,
    )
    result = sam.segment(seg_req)
    return {
        "masks": [
            {"mask_b64": m.mask_b64, "score": m.score, "area": m.area}
            for m in result.masks
        ],
        "n_masks": result.n_masks,
    }


# ============================================================================
# Sprint 50 — エージェントフレームワーク強化 API
# ============================================================================

from open_mythos.skills.agent_framework import (  # noqa: E402
    SubAgentTask as _SubAgentTask,
    SubAgentOrchestrator as _SubAgentOrchestrator,
    TDDAgent as _TDDAgent,
    BugReport as _BugReport,
    SystematicDebugger as _SystematicDebugger,
    DarwinianEvolver as _DarwinianEvolver,
    ParallelJob as _ParallelJob,
    ParallelCLIRunner as _ParallelCLIRunner,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────


class _SubAgentPlanRequest(BaseModel):
    goal: str
    n_subtasks: int = 3


class _SubAgentTaskItem(BaseModel):
    task_id: str
    description: str
    priority: int = 1
    timeout_s: float = 60.0


class _SubAgentRunRequest(BaseModel):
    tasks: List[_SubAgentTaskItem]


class _TDDCycleRequest(BaseModel):
    goal: str


class _TDDSessionRequest(BaseModel):
    goals: List[str]


class _DebugRequest(BaseModel):
    description: str
    context: str = ""
    stack_trace: str = ""
    observed: str = ""
    expected: str = ""


class _EvolveRequest(BaseModel):
    genome_dim: int = 4
    n_generations: int = 20
    population_size: int = 30
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    genome_bounds_lo: float = -1.0
    genome_bounds_hi: float = 1.0


class _CLIJobItem(BaseModel):
    job_id: str
    command: str
    cwd: str = "."
    timeout_s: float = 30.0


class _CLIRunRequest(BaseModel):
    jobs: List[_CLIJobItem]


# ── エンドポイント ─────────────────────────────────────────────────────────────


@router.post(
    "/v1/agent/subagent/plan",
    tags=["agent"],
    summary="サブエージェント — タスク計画生成 (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def subagent_plan(req: _SubAgentPlanRequest):
    """ゴールからサブタスクリストを生成する。"""
    orch = _SubAgentOrchestrator()
    tasks = orch.plan(req.goal, n_subtasks=req.n_subtasks)
    return {
        "tasks": [
            {"task_id": t.task_id, "description": t.description, "priority": t.priority}
            for t in tasks
        ]
    }


@router.post(
    "/v1/agent/subagent/run",
    tags=["agent"],
    summary="サブエージェント — タスク実行 (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def subagent_run(req: _SubAgentRunRequest):
    """サブタスクリストを実行し 2-stage レビュー結果を返す。"""
    orch = _SubAgentOrchestrator()
    tasks = [
        _SubAgentTask(
            task_id=t.task_id,
            description=t.description,
            priority=t.priority,
            timeout_s=t.timeout_s,
        )
        for t in req.tasks
    ]
    results = orch.run(tasks)
    return {
        "results": [
            {
                "task_id": r.task_id,
                "output": r.output,
                "success": r.success,
                "review_passed": r.review_passed,
                "latency_ms": r.latency_ms,
                "reviewer_feedback": r.reviewer_feedback,
            }
            for r in results
        ]
    }


@router.post(
    "/v1/agent/tdd/cycle",
    tags=["agent"],
    summary="TDD エージェント — 1 サイクル実行 (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def tdd_cycle(req: _TDDCycleRequest):
    """Red-Green-Refactor 1 サイクルを実行する。"""
    agent = _TDDAgent()
    cycle = agent.run_cycle(req.goal)
    return {
        "phase": cycle.phase,
        "test_code": cycle.test_code,
        "impl_code": cycle.impl_code,
        "test_passed": cycle.test_passed,
        "notes": cycle.notes,
    }


@router.post(
    "/v1/agent/tdd/session",
    tags=["agent"],
    summary="TDD エージェント — セッション実行 (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def tdd_session(req: _TDDSessionRequest):
    """複数ゴールに対して TDD セッションを実行する。"""
    agent = _TDDAgent()
    session = agent.run_session(req.goals)
    return {
        "goal": session.goal,
        "total_tests_written": session.total_tests_written,
        "total_tests_passing": session.total_tests_passing,
        "pass_rate": session.pass_rate,
        "cycles": [
            {
                "phase": c.phase,
                "test_passed": c.test_passed,
                "notes": c.notes,
            }
            for c in session.cycles
        ],
    }


@router.post(
    "/v1/agent/debug",
    tags=["agent"],
    summary="体系的デバッガー — 4 フェーズデバッグ (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def agent_debug(req: _DebugRequest):
    """バグレポートを受け取り 4 フェーズで体系的にデバッグする。"""
    dbg = _SystematicDebugger()
    bug = _BugReport(
        description=req.description,
        context=req.context,
        stack_trace=req.stack_trace,
        observed=req.observed,
        expected=req.expected,
    )
    session = dbg.debug(bug)
    return {
        "root_cause": session.root_cause,
        "fix_suggestion": session.fix_suggestion,
        "verified": session.verified,
        "steps": [
            {
                "phase": s.phase,
                "action": s.action,
                "finding": s.finding,
                "hypothesis": s.hypothesis,
                "confidence": s.confidence,
            }
            for s in session.steps
        ],
    }


@router.post(
    "/v1/agent/evolve",
    tags=["agent"],
    summary="Darwinian Evolver — 進化的最適化 (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def agent_evolve(req: _EvolveRequest):
    """遺伝的アルゴリズムで最適解を探索する。"""

    def _fitness(genome: List[float]) -> float:  # type: ignore[name-defined]
        # デフォルト適応度: 各次元の二乗和の負（最小化→最大化）
        return -sum(g * g for g in genome)

    evolver = _DarwinianEvolver(
        population_size=req.population_size,
        mutation_rate=req.mutation_rate,
        crossover_rate=req.crossover_rate,
    )
    result = evolver.evolve(
        fitness_fn=_fitness,
        genome_dim=req.genome_dim,
        n_generations=req.n_generations,
        genome_bounds=(req.genome_bounds_lo, req.genome_bounds_hi),
    )
    return {
        "best_fitness": result.best_fitness,
        "generations_run": result.generations_run,
        "population_size": result.population_size,
        "fitness_history": result.fitness_history,
        "best_genome": result.best_individual.genome,
    }


@router.post(
    "/v1/agent/cli/run",
    tags=["agent"],
    summary="並列 CLI ランナー — コマンド実行 (Sprint 50)",
    dependencies=[Depends(verify_api_key)],
)
def agent_cli_run(req: _CLIRunRequest):
    """複数の CLI コマンドを並列実行する。"""
    runner = _ParallelCLIRunner()
    jobs = [
        _ParallelJob(
            job_id=j.job_id,
            command=j.command,
            cwd=j.cwd,
            timeout_s=j.timeout_s,
        )
        for j in req.jobs
    ]
    results = runner.run(jobs)
    return {
        "results": [
            {
                "job_id": r.job_id,
                "command": r.command,
                "returncode": r.returncode,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "duration_s": r.duration_s,
                "success": r.success,
            }
            for r in results
        ]
    }


# ============================================================================
# Sprint 51 — データ・検索ツール統合 API
# ============================================================================

from typing import Any  # noqa: E402 (Any not yet imported at module level)

from open_mythos.skills.data_tools import (  # noqa: E402
    SearXNGSearcher as _SearXNGSearcher,
    DomainIntelligence as _DomainIntelligence,
    NemoCurator as _NemoCurator,
    CurationRule as _CurationRule,
    CodeWikiGenerator as _CodeWikiGenerator,
    APIDebugger as _APIDebugger,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────


class _SearXNGRequest(BaseModel):
    query: str
    categories: Optional[List[str]] = None
    engines: Optional[List[str]] = None
    max_results: int = 10
    base_url: str = "https://searxng.org"


class _DomainLookupRequest(BaseModel):
    domain: str
    check_ssl: bool = False


class _CurationRuleItem(BaseModel):
    name: str
    description: str
    field: str = "text"
    min_length: int = 10
    max_length: int = 100_000
    deduplicate: bool = False
    language: str = ""


class _DataCurateRequest(BaseModel):
    documents: List[Dict[str, Any]]
    rules: List[_CurationRuleItem] = []


class _CodeWikiRequest(BaseModel):
    source_code: str
    module_name: str = "module"
    title: str = "Code Reference"


class _APIRestRequest(BaseModel):
    url: str
    method: str = "GET"
    headers: Dict[str, str] = {}
    body: Optional[str] = None
    timeout_s: float = 10.0


class _APIGraphQLRequest(BaseModel):
    url: str
    query: str
    variables: Dict[str, Any] = {}
    headers: Dict[str, str] = {}
    timeout_s: float = 10.0


# ── エンドポイント ─────────────────────────────────────────────────────────────


@router.post(
    "/v1/search/searxng",
    tags=["data"],
    summary="SearXNG — プライバシー重視メタ検索 (Sprint 51)",
    dependencies=[Depends(verify_api_key)],
)
def searxng_search(req: _SearXNGRequest):
    """SearXNG を使ってプライバシー重視の検索を行う。"""
    searcher = _SearXNGSearcher(base_url=req.base_url)
    results = searcher.search(
        req.query,
        categories=req.categories,
        engines=req.engines,
        max_results=req.max_results,
    )
    return {
        "query": req.query,
        "results": [
            {"title": r.title, "url": r.url, "content": r.content,
             "engine": r.engine, "score": r.score}
            for r in results
        ],
        "total": len(results),
    }


@router.post(
    "/v1/domain/lookup",
    tags=["data"],
    summary="ドメインインテリジェンス — WHOIS/DNS/SSL 収集 (Sprint 51)",
    dependencies=[Depends(verify_api_key)],
)
def domain_lookup(req: _DomainLookupRequest):
    """ドメインの IP / NS / MX 情報を収集する。"""
    di = _DomainIntelligence()
    info = di.lookup(req.domain)
    result: dict = {
        "domain": info.domain,
        "ip": info.ip,
        "ns_records": info.ns_records,
        "mx_records": info.mx_records,
    }
    if req.check_ssl:
        ssl_info = di.check_ssl(req.domain)
        result["ssl"] = ssl_info
    return result


@router.post(
    "/v1/data/curate",
    tags=["data"],
    summary="NeMo Curator — データキュレーション (Sprint 51)",
    dependencies=[Depends(verify_api_key)],
)
def data_curate(req: _DataCurateRequest):
    """ドキュメントリストにキュレーションルールを適用する。"""
    rules = [
        _CurationRule(
            name=r.name,
            description=r.description,
            field=r.field,
            min_length=r.min_length,
            max_length=r.max_length,
            deduplicate=r.deduplicate,
            language=r.language,
        )
        for r in req.rules
    ]
    curator = _NemoCurator(rules=rules)
    kept, result = curator.curate(req.documents)
    return {
        "total_input": result.total_input,
        "total_output": result.total_output,
        "removed_count": result.removed_count,
        "duplicate_count": result.duplicate_count,
        "rule_stats": result.rule_stats,
        "kept_documents": kept,
    }


@router.post(
    "/v1/code/wiki",
    tags=["data"],
    summary="Code Wiki — ソースコード解析・Markdown 生成 (Sprint 51)",
    dependencies=[Depends(verify_api_key)],
)
def code_wiki(req: _CodeWikiRequest):
    """Python ソースコードを解析して Markdown リファレンスを生成する。"""
    gen = _CodeWikiGenerator()
    symbols = gen.analyze_source(req.source_code, module_name=req.module_name)
    wiki = gen.generate(symbols, title=req.title)
    return {
        "title": wiki.title,
        "markdown": wiki.markdown,
        "n_symbols": wiki.n_symbols,
        "symbols": [
            {"name": s.name, "kind": s.kind, "module": s.module,
             "signature": s.signature, "line": s.line}
            for s in wiki.symbols
        ],
    }


@router.post(
    "/v1/api/rest",
    tags=["data"],
    summary="API デバッガー — REST リクエスト実行 (Sprint 51)",
    dependencies=[Depends(verify_api_key)],
)
def api_rest(req: _APIRestRequest):
    """REST API を実行してレスポンスを解析する。"""
    dbg = _APIDebugger(timeout=req.timeout_s)
    result = dbg.call_rest(
        url=req.url,
        method=req.method,
        headers=req.headers,
        body=req.body,
    )
    analysis = dbg.inspect_response(result)
    return {
        "url": result.url,
        "method": result.method,
        "status_code": result.status_code,
        "success": result.success,
        "duration_ms": result.duration_ms,
        "valid_json": analysis.get("valid_json", False),
        "body_length": analysis.get("body_length", 0),
    }


@router.post(
    "/v1/api/graphql",
    tags=["data"],
    summary="API デバッガー — GraphQL クエリ実行 (Sprint 51)",
    dependencies=[Depends(verify_api_key)],
)
def api_graphql(req: _APIGraphQLRequest):
    """GraphQL エンドポイントにクエリを送信してレスポンスを解析する。"""
    dbg = _APIDebugger(timeout=req.timeout_s)
    result = dbg.call_graphql(
        url=req.url,
        query=req.query,
        variables=req.variables,
        headers=req.headers,
    )
    analysis = dbg.inspect_response(result)
    return {
        "url": result.url,
        "status_code": result.status_code,
        "success": result.success,
        "duration_ms": result.duration_ms,
        "valid_json": analysis.get("valid_json", False),
        "body_length": analysis.get("body_length", 0),
    }


# ============================================================================
# Sprint 52 — DevOps・クラウド統合 API
# ============================================================================

from open_mythos.skills.devops_cloud import (  # noqa: E402
    ModalFunctionConfig as _ModalConfig,
    ModalRunner as _ModalRunner,
    DockerManager as _DockerManager,
    WatchRule as _WatchRule,
    FileWatcher as _FileWatcher,
    SLiMeConfig as _SLiMeConfig,
    SLiMeModel as _SLiMeModel,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────


class _ModalRunRequest(BaseModel):
    name: str
    # code フィールドは廃止 — eval による RCE を防止するため operation + args に変更
    operation: str = "echo"   # echo | sum | count | noop
    args: List[Any] = []
    gpu: str = ""
    memory: int = 512
    timeout: int = 300


class _ModalStubRequest(BaseModel):
    name: str
    gpu: str = ""
    memory: int = 512
    timeout: int = 300
    app_name: str = "openmythos"


class _DockerContainersRequest(BaseModel):
    all: bool = False


class _DockerBuildRequest(BaseModel):
    dockerfile_path: str
    tag: str
    context: str = "."


class _WatchRuleItem(BaseModel):
    path: str
    pattern: str = "**/*"
    event_types: List[str] = ["created", "modified", "deleted"]
    recursive: bool = True
    debounce_ms: int = 500


class _WatchConfigRequest(BaseModel):
    rules: List[_WatchRuleItem] = []


class _SLiMeFitRequest(BaseModel):
    data: List[List[float]]
    n_components: int = 64
    alpha: float = 0.01
    normalize: bool = True


# ── エンドポイント ─────────────────────────────────────────────────────────────


@router.post(
    "/v1/modal/run",
    tags=["devops"],
    summary="Modal — クラウド関数実行 (Sprint 52)",
    dependencies=[Depends(verify_api_key)],
)
def modal_run(req: _ModalRunRequest):
    """定義済み操作をクラウド (または fallback) で実行する。eval() は使用しない。"""
    cfg = _ModalConfig(name=req.name, gpu=req.gpu, memory=req.memory, timeout=req.timeout)
    runner = _ModalRunner()
    # 安全な定義済み操作のみ許可 (RCE 防止)
    _ops = {
        "echo":  lambda *a: list(a),
        "sum":   lambda *a: sum(float(x) for x in a),
        "count": lambda *a: len(a),
        "noop":  lambda *a: None,
    }
    fn = _ops.get(req.operation, _ops["noop"])
    result = runner.run_function(fn, cfg, *req.args)
    return {
        "output": result.output,
        "run_id": result.run_id,
        "duration_s": result.duration_s,
        "gpu_used": result.gpu_used,
        "success": result.success,
        "error": result.error,
    }


@router.post(
    "/v1/modal/stub",
    tags=["devops"],
    summary="Modal — スタブコード生成 (Sprint 52)",
    dependencies=[Depends(verify_api_key)],
)
def modal_stub(req: _ModalStubRequest):
    """Modal 関数のスタブコードを生成する。"""
    cfg = _ModalConfig(name=req.name, gpu=req.gpu, memory=req.memory, timeout=req.timeout)
    runner = _ModalRunner(app_name=req.app_name)
    stub = runner.build_stub(cfg)
    return {"stub_code": stub, "app_name": req.app_name, "function_name": req.name}


@router.post(
    "/v1/docker/containers",
    tags=["devops"],
    summary="Docker — コンテナ一覧 (Sprint 52)",
    dependencies=[Depends(verify_api_key)],
)
def docker_containers(req: _DockerContainersRequest):
    """実行中（または全）コンテナの一覧を返す。"""
    dm = _DockerManager()
    containers = dm.list_containers(all=req.all)
    return {
        "containers": [
            {
                "container_id": c.container_id,
                "name": c.name,
                "image": c.image,
                "status": c.status,
                "ports": c.ports,
                "created": c.created,
            }
            for c in containers
        ],
        "count": len(containers),
    }


@router.post(
    "/v1/docker/build",
    tags=["devops"],
    summary="Docker — イメージビルド (Sprint 52)",
    dependencies=[Depends(verify_api_key)],
)
def docker_build(req: _DockerBuildRequest):
    """Docker イメージをビルドする。"""
    dm = _DockerManager()
    result = dm.build(req.dockerfile_path, req.tag, req.context)
    return {
        "image_id": result.image_id,
        "tag": result.tag,
        "build_time_s": result.build_time_s,
        "size_mb": result.size_mb,
        "success": result.success,
        "error": result.error,
    }


@router.post(
    "/v1/watch/config",
    tags=["devops"],
    summary="FileWatcher — 監視設定生成 (Sprint 52)",
    dependencies=[Depends(verify_api_key)],
)
def watch_config(req: _WatchConfigRequest):
    """ファイル監視ルールを設定して構成情報を返す。"""
    rules = [
        _WatchRule(
            path=r.path,
            pattern=r.pattern,
            event_types=r.event_types,
            recursive=r.recursive,
            debounce_ms=r.debounce_ms,
        )
        for r in req.rules
    ]
    fw = _FileWatcher(rules=rules)
    config = fw.build_config()
    return config


@router.post(
    "/v1/slime/fit",
    tags=["devops"],
    summary="SLiMe — スパース特徴学習 (Sprint 52)",
    dependencies=[Depends(verify_api_key)],
)
def slime_fit(req: _SLiMeFitRequest):
    """SLiMe でスパース特徴を学習する。"""
    cfg = _SLiMeConfig(
        n_components=req.n_components,
        alpha=req.alpha,
        normalize=req.normalize,
    )
    model = _SLiMeModel(cfg)
    result = model.fit(req.data)
    return {
        "components": result.components,
        "sparsity": result.sparsity,
        "reconstruction_error": result.reconstruction_error,
        "n_iter": result.n_iter,
    }



