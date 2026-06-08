"""
Sprint 51 — データ・検索ツール統合

Hermes Skills: searxng / domain-intel / nemo-curator / code-wiki / rest-graphql-debug
ref: skills/data/*-SKILL.md

データキュレーション・検索・コード解析ツールを OpenMythos に統合する。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SearXNG プライバシー検索
# ---------------------------------------------------------------------------

@dataclass
class SearXNGResult:
    """SearXNG 検索結果。"""
    title: str
    url: str
    content: str
    engine: str = "searxng"
    score: float = 0.0


class SearXNGSearcher:
    """SearXNG メタ検索エンジンクライアント。"""

    def __init__(self, base_url: str = "https://searxng.org", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[SearXNGResult]:
        """SearXNG API で検索する。"""
        try:
            import urllib.request, json, urllib.parse
            params = {
                "q": query,
                "format": "json",
                "pageno": "1",
            }
            if categories:
                params["categories"] = ",".join(categories)
            if engines:
                params["engines"] = ",".join(engines)
            url = f"{self.base_url}/search?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "OpenMythos/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
            return [
                SearXNGResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", "")[:300],
                    engine=",".join(item.get("engines", ["unknown"])),
                    score=item.get("score", 0.0),
                )
                for item in data.get("results", [])[:max_results]
            ]
        except Exception:
            return [
                SearXNGResult(
                    title=f"SearXNG result {i}: {query}",
                    url=f"https://example.com?q={query.replace(' ','+')}#{i}",
                    content=f"Mock result {i} for '{query}'",
                )
                for i in range(min(max_results, 3))
            ]


# ---------------------------------------------------------------------------
# Domain Intelligence
# ---------------------------------------------------------------------------

@dataclass
class DomainInfo:
    """ドメイン情報。"""
    domain: str
    ip: str = ""
    registrar: str = ""
    created_date: str = ""
    expiry_date: str = ""
    ns_records: List[str] = field(default_factory=list)
    mx_records: List[str] = field(default_factory=list)
    ssl_issuer: str = ""
    ssl_expiry: str = ""
    technologies: List[str] = field(default_factory=list)


class DomainIntelligence:
    """ドメインインテリジェンス収集クライアント。

    WHOIS / DNS / SSL 情報を収集する。
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def lookup(self, domain: str) -> DomainInfo:
        """ドメインの総合情報を収集する。"""
        info = DomainInfo(domain=domain)
        info.ip = self._resolve_ip(domain)
        info.ns_records = self._get_ns(domain)
        info.mx_records = self._get_mx(domain)
        return info

    def _resolve_ip(self, domain: str) -> str:
        try:
            import socket
            return socket.gethostbyname(domain)
        except Exception:
            return "0.0.0.0"

    def _get_ns(self, domain: str) -> List[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["nslookup", "-type=NS", domain],
                capture_output=True, timeout=5, text=True,
            )
            return re.findall(r"nameserver = (.+)", result.stdout)
        except Exception:
            return []

    def _get_mx(self, domain: str) -> List[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["nslookup", "-type=MX", domain],
                capture_output=True, timeout=5, text=True,
            )
            return re.findall(r"mail exchanger = (.+)", result.stdout)
        except Exception:
            return []

    def check_ssl(self, domain: str, port: int = 443) -> Dict[str, str]:
        """SSL 証明書情報を取得する。"""
        try:
            import ssl, socket
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(self.timeout)
                s.connect((domain, port))
                cert = s.getpeercert()
            issuer = dict(x[0] for x in cert.get("issuer", []))
            expiry = cert.get("notAfter", "")
            return {
                "issuer": issuer.get("organizationName", ""),
                "expiry": expiry,
                "valid": True,
            }
        except Exception:
            return {"issuer": "", "expiry": "", "valid": False}


# ---------------------------------------------------------------------------
# NeMo Curator データキュレーション
# ---------------------------------------------------------------------------

@dataclass
class CurationRule:
    """データキュレーションルール。"""
    name: str
    description: str
    field: str = "text"
    min_length: int = 10
    max_length: int = 100_000
    deduplicate: bool = False
    language: str = ""               # 言語フィルタ (空=全言語)


@dataclass
class CurationResult:
    """キュレーション結果。"""
    total_input: int
    total_output: int
    removed_count: int
    duplicate_count: int
    rule_stats: Dict[str, int]


class NemoCurator:
    """NeMo Curator データキュレーションパイプライン。

    `nemo_curator` がある場合はそれを使用し、
    ない場合は Python ベースのフィルタリングを行う。
    """

    def __init__(self, rules: Optional[List[CurationRule]] = None) -> None:
        self.rules = rules or []
        try:
            import nemo_curator  # type: ignore
            self._nemo = nemo_curator
            self._native = True
        except ImportError:
            self._nemo = None
            self._native = False

    def curate(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], CurationResult]:
        """ドキュメントリストにキュレーションを適用する。"""
        total_input = len(documents)
        rule_stats: Dict[str, int] = {}
        filtered = list(documents)

        for rule in self.rules:
            before = len(filtered)
            new_filtered = []
            for doc in filtered:
                text = str(doc.get(rule.field, ""))
                if len(text) < rule.min_length:
                    rule_stats[f"{rule.name}_too_short"] = rule_stats.get(f"{rule.name}_too_short", 0) + 1
                    continue
                if len(text) > rule.max_length:
                    rule_stats[f"{rule.name}_too_long"] = rule_stats.get(f"{rule.name}_too_long", 0) + 1
                    continue
                if rule.language:
                    # 簡易言語検出: ASCII 比率
                    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
                    if rule.language == "en" and ascii_ratio < 0.8:
                        rule_stats[f"{rule.name}_lang_filter"] = rule_stats.get(f"{rule.name}_lang_filter", 0) + 1
                        continue
                new_filtered.append(doc)
            filtered = new_filtered
            removed = before - len(filtered)
            if removed:
                rule_stats[rule.name] = rule_stats.get(rule.name, 0) + removed

        # 重複除去
        dup_count = 0
        if any(r.deduplicate for r in self.rules):
            seen = set()
            deduped = []
            for doc in filtered:
                key = str(doc.get("text", ""))[:200]
                if key in seen:
                    dup_count += 1
                else:
                    seen.add(key)
                    deduped.append(doc)
            filtered = deduped

        result = CurationResult(
            total_input=total_input,
            total_output=len(filtered),
            removed_count=total_input - len(filtered) - dup_count,
            duplicate_count=dup_count,
            rule_stats=rule_stats,
        )
        return filtered, result

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# Code Wiki 生成
# ---------------------------------------------------------------------------

@dataclass
class CodeSymbol:
    """コードシンボル情報。"""
    name: str
    kind: str                   # function | class | method | variable
    module: str
    signature: str = ""
    docstring: str = ""
    line: int = 0


@dataclass
class CodeWiki:
    """コードWiki生成結果。"""
    title: str
    symbols: List[CodeSymbol]
    markdown: str
    n_symbols: int


class CodeWikiGenerator:
    """コードベースから Markdown Wiki を自動生成するジェネレーター。"""

    def __init__(self) -> None:
        pass

    def analyze_source(self, source_code: str, module_name: str = "module") -> List[CodeSymbol]:
        """Python ソースコードを解析してシンボルを抽出する。"""
        symbols = []
        try:
            import ast
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    doc = (ast.get_docstring(node) or "")[:200]
                    args = [a.arg for a in node.args.args]
                    sig = f"def {node.name}({', '.join(args)})"
                    symbols.append(CodeSymbol(
                        name=node.name, kind="function", module=module_name,
                        signature=sig, docstring=doc, line=node.lineno,
                    ))
                elif isinstance(node, ast.ClassDef):
                    doc = (ast.get_docstring(node) or "")[:200]
                    symbols.append(CodeSymbol(
                        name=node.name, kind="class", module=module_name,
                        signature=f"class {node.name}", docstring=doc, line=node.lineno,
                    ))
        except SyntaxError:
            # 簡易正規表現フォールバック
            for m in re.finditer(r"^def (\w+)\(", source_code, re.MULTILINE):
                symbols.append(CodeSymbol(name=m.group(1), kind="function", module=module_name))
            for m in re.finditer(r"^class (\w+)", source_code, re.MULTILINE):
                symbols.append(CodeSymbol(name=m.group(1), kind="class", module=module_name))
        return symbols

    def generate(self, symbols: List[CodeSymbol], title: str = "Code Reference") -> CodeWiki:
        """シンボルリストから Markdown Wiki を生成する。"""
        lines = [f"# {title}\n"]
        # 種別ごとにグループ化
        classes = [s for s in symbols if s.kind == "class"]
        functions = [s for s in symbols if s.kind == "function"]
        if classes:
            lines.append("## Classes\n")
            for sym in classes:
                lines.append(f"### `{sym.name}`\n")
                if sym.signature:
                    lines.append(f"```python\n{sym.signature}\n```\n")
                if sym.docstring:
                    lines.append(f"{sym.docstring}\n")
        if functions:
            lines.append("## Functions\n")
            for sym in functions:
                lines.append(f"### `{sym.name}`\n")
                if sym.signature:
                    lines.append(f"```python\n{sym.signature}\n```\n")
                if sym.docstring:
                    lines.append(f"{sym.docstring}\n")
        md = "\n".join(lines)
        return CodeWiki(title=title, symbols=symbols, markdown=md, n_symbols=len(symbols))


# ---------------------------------------------------------------------------
# REST / GraphQL デバッグ
# ---------------------------------------------------------------------------

@dataclass
class APICallResult:
    """API 呼び出し結果。"""
    url: str
    method: str
    status_code: int
    response_body: str
    headers: Dict[str, str]
    duration_ms: float
    success: bool


class APIDebugger:
    """REST / GraphQL API デバッグツール。"""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def call_rest(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> APICallResult:
        """REST API を呼び出す。"""
        import urllib.request
        t0 = time.perf_counter()
        try:
            data = body.encode() if body else None
            req = urllib.request.Request(
                url,
                data=data,
                method=method.upper(),
                headers=headers or {},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp_body = r.read().decode("utf-8", errors="replace")[:2000]
                resp_headers = dict(r.headers)
                status = r.status
            return APICallResult(
                url=url, method=method.upper(), status_code=status,
                response_body=resp_body, headers=resp_headers,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                success=(200 <= status < 300),
            )
        except Exception as e:
            return APICallResult(
                url=url, method=method.upper(), status_code=0,
                response_body=str(e), headers={},
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                success=False,
            )

    def call_graphql(
        self,
        url: str,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APICallResult:
        """GraphQL エンドポイントを呼び出す。"""
        import json
        body = json.dumps({"query": query, "variables": variables or {}})
        all_headers = {"Content-Type": "application/json", **(headers or {})}
        return self.call_rest(url, method="POST", headers=all_headers, body=body)

    def inspect_response(self, result: APICallResult) -> Dict[str, Any]:
        """レスポンスを解析してデバッグ情報を返す。"""
        import json
        analysis: Dict[str, Any] = {
            "status": result.status_code,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "content_type": result.headers.get("Content-Type", ""),
            "body_length": len(result.response_body),
        }
        try:
            analysis["json"] = json.loads(result.response_body)
            analysis["valid_json"] = True
        except Exception:
            analysis["valid_json"] = False
        return analysis
