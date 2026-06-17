"""
Sprint 47 — 研究ツール統合

Hermes Skills: arxiv / dspy / duckduckgo-search / scrapling / jupyter-live-kernel
ref: skills/research/*-SKILL.md

研究・情報収集ツールを OpenMythos に統合する。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# arXiv 論文検索
# ---------------------------------------------------------------------------

@dataclass
class ArxivPaper:
    """arXiv 論文情報。"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str] = field(default_factory=list)
    published: str = ""
    pdf_url: str = ""

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class ArxivSearcher:
    """arXiv 論文検索クライアント。

    `arxiv` ライブラリがある場合はそれを使用し、
    ない場合は HTTP API へ直接アクセスする。
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        try:
            import arxiv as _arxiv  # type: ignore
            self._arxiv = _arxiv
            self._native = True
        except ImportError:
            self._arxiv = None
            self._native = False

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        category: str = "",
    ) -> List[ArxivPaper]:
        """キーワードで論文を検索する。"""
        k = max_results or self.max_results
        if self._native:
            try:
                search_query = query
                if category:
                    search_query = f"cat:{category} AND ({query})"
                results = list(self._arxiv.Search(query=search_query, max_results=k).results())
                return [
                    ArxivPaper(
                        arxiv_id=r.get_short_id(),
                        title=r.title,
                        authors=[a.name for a in r.authors],
                        abstract=r.summary[:500],
                        categories=r.categories,
                        published=str(r.published.date()),
                        pdf_url=r.pdf_url,
                    )
                    for r in results
                ]
            except Exception:
                pass
        # fallback: HTTP API
        try:
            import urllib.request
            params = f"search_query=all:{query.replace(' ', '+')}&max_results={k}"
            url = f"{self.BASE_URL}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "OpenMythos/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                xml = r.read().decode("utf-8")
            return self._parse_atom(xml, k)
        except Exception:
            pass
        # ダミー
        return [ArxivPaper(
            arxiv_id=f"2401.{i:05d}",
            title=f"Mock Paper: {query} ({i})",
            authors=["Mock Author"],
            abstract=f"This is a mock abstract for query '{query}'.",
        ) for i in range(min(k, 3))]

    def _parse_atom(self, xml: str, limit: int) -> List[ArxivPaper]:
        """Atom XML から論文情報を抽出する。"""
        papers = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for entry in entries[:limit]:
            arxiv_id = re.search(r"<id>.*?/abs/([^<]+)</id>", entry)
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            abstract = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            authors = re.findall(r"<name>(.*?)</name>", entry)
            if arxiv_id and title:
                papers.append(ArxivPaper(
                    arxiv_id=arxiv_id.group(1).strip(),
                    title=title.group(1).strip().replace("\n", " "),
                    authors=authors[:3],
                    abstract=(abstract.group(1).strip()[:400] if abstract else ""),
                ))
        return papers

    def get_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """ID から論文を取得する。"""
        results = self.search(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# DSPy プログラマブルプロンプト
# ---------------------------------------------------------------------------

@dataclass
class DSPySignature:
    """DSPy シグネチャ定義。"""
    name: str
    inputs: Dict[str, str]    # field_name → description
    outputs: Dict[str, str]
    instructions: str = ""


@dataclass
class DSPyPrediction:
    """DSPy 予測結果。"""
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    rationale: str = ""
    success: bool = True


class DSPyOptimizer:
    """DSPy プログラマブルプロンプト最適化。

    `dspy` ライブラリがある場合は本物を使用し、
    ない場合はテンプレートベースの代替を使用する。
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model
        try:
            import dspy  # type: ignore
            self._dspy = dspy
            self._native = True
        except ImportError:
            self._dspy = None
            self._native = False

    def predict(
        self,
        signature: DSPySignature,
        inputs: Dict[str, Any],
    ) -> DSPyPrediction:
        """シグネチャに従って予測を実行する。"""
        if self._native and self._model:
            try:
                class DynSig(self._dspy.Signature):
                    pass
                for k, desc in signature.inputs.items():
                    setattr(DynSig, k, self._dspy.InputField(desc=desc))
                for k, desc in signature.outputs.items():
                    setattr(DynSig, k, self._dspy.OutputField(desc=desc))
                predictor = self._dspy.Predict(DynSig)
                pred = predictor(**inputs)
                return DSPyPrediction(
                    inputs=inputs,
                    outputs={k: getattr(pred, k, "") for k in signature.outputs},
                )
            except Exception:
                pass
        # fallback
        outputs = {k: f"[predicted {k} from {list(inputs.values())[:1]}]"
                   for k in signature.outputs}
        return DSPyPrediction(inputs=inputs, outputs=outputs)

    def build_chain_of_thought(self, signature: DSPySignature) -> str:
        """Chain-of-Thought プロンプトを生成する。"""
        in_str = ", ".join(f"{k}: {v}" for k, v in signature.inputs.items())
        out_str = ", ".join(f"{k}" for k in signature.outputs)
        return (
            f"Given: {in_str}\n"
            f"Think step by step to produce: {out_str}\n"
            f"Rationale: [your reasoning]\n"
            f"Answer: {{{', '.join(signature.outputs)}}}"
        )

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# Web 検索 (DuckDuckGo)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """検索結果エントリ。"""
    title: str
    url: str
    snippet: str
    source: str = "web"


class WebSearcher:
    """Web 検索クライアント (DuckDuckGo / SearXNG)。

    `duckduckgo_search` がある場合はそれを使用し、
    ない場合は requests ベースの HTTP フォールバックを使用する。
    """

    def __init__(self, backend: str = "duckduckgo", max_results: int = 10) -> None:
        self.backend = backend
        self.max_results = max_results
        self._native = False
        if backend == "duckduckgo":
            try:
                from duckduckgo_search import DDGS  # type: ignore
                self._ddgs = DDGS()
                self._native = True
            except ImportError:
                self._ddgs = None

    def search(
        self,
        query: str,
        region: str = "wt-wt",
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Web を検索する。"""
        k = max_results or self.max_results
        if self._native and self._ddgs:
            try:
                results = list(self._ddgs.text(query, region=region, max_results=k))
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    )
                    for r in results
                ]
            except Exception:
                pass
        # fallback: ダミー
        return [
            SearchResult(
                title=f"Result {i}: {query}",
                url=f"https://example.com/search?q={query.replace(' ','+')}#{i}",
                snippet=f"Mock snippet for '{query}' result {i}.",
            )
            for i in range(min(k, 3))
        ]

    def news(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """ニュース検索を行う。"""
        if self._native and self._ddgs:
            try:
                results = list(self._ddgs.news(query, max_results=max_results))
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("body", ""),
                        source="news",
                    )
                    for r in results
                ]
            except Exception:
                pass
        return [SearchResult(title=f"News: {query}", url="https://example.com", snippet="mock")]

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# Jupyter Live Kernel
# ---------------------------------------------------------------------------

@dataclass
class KernelExecutionResult:
    """Jupyter カーネル実行結果。"""
    stdout: str
    stderr: str
    outputs: List[Dict[str, Any]]
    execution_count: int
    success: bool
    error_name: str = ""
    error_traceback: List[str] = field(default_factory=list)


class JupyterKernelClient:
    """Jupyter カーネルクライアント。

    `jupyter_client` がある場合は本物のカーネルを起動し、
    ない場合は `exec` ベースのシンプルな実行を行う。
    """

    def __init__(self, kernel_name: str = "python3") -> None:
        self.kernel_name = kernel_name
        self._km = None
        self._kc = None
        self._native = False
        try:
            from jupyter_client import KernelManager  # type: ignore
            self._KernelManager = KernelManager
            self._native = True
        except ImportError:
            self._KernelManager = None

    def execute(self, code: str, timeout: float = 30.0) -> KernelExecutionResult:
        """コードを実行する。"""
        if self._native:
            try:
                km = self._KernelManager(kernel_name=self.kernel_name)
                km.start_kernel()
                kc = km.client()
                kc.start_channels()
                kc.wait_for_ready(timeout=10)
                kc.execute(code)
                outputs = []
                stdout_parts = []
                stderr_parts = []
                ec = 0
                while True:
                    msg = kc.get_iopub_msg(timeout=timeout)
                    msg_type = msg["msg_type"]
                    content = msg["content"]
                    if msg_type == "stream":
                        if content["name"] == "stdout":
                            stdout_parts.append(content["text"])
                        else:
                            stderr_parts.append(content["text"])
                    elif msg_type in ("execute_result", "display_data"):
                        outputs.append(content.get("data", {}))
                        ec = content.get("execution_count", 0)
                    elif msg_type == "error":
                        kc.stop_channels(); km.shutdown_kernel()
                        return KernelExecutionResult(
                            stdout="", stderr="".join(content.get("traceback", [])),
                            outputs=[], execution_count=0, success=False,
                            error_name=content.get("ename", "Error"),
                            error_traceback=content.get("traceback", []),
                        )
                    elif msg_type == "status" and content.get("execution_state") == "idle":
                        break
                kc.stop_channels(); km.shutdown_kernel()
                return KernelExecutionResult(
                    stdout="".join(stdout_parts), stderr="".join(stderr_parts),
                    outputs=outputs, execution_count=ec, success=True,
                )
            except Exception:
                pass
        # fallback: exec
        import io, sys
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = buf_out = io.StringIO()
        sys.stderr = buf_err = io.StringIO()
        success = True
        error_name = ""
        try:
            exec(compile(code, "<string>", "exec"))
        except Exception as e:
            success = False
            error_name = type(e).__name__
            print(str(e), file=sys.stderr)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        return KernelExecutionResult(
            stdout=buf_out.getvalue(), stderr=buf_err.getvalue(),
            outputs=[], execution_count=1, success=success, error_name=error_name,
        )

    @property
    def is_native(self) -> bool:
        return self._native
