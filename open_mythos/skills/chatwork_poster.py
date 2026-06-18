"""
Sprint 67 — Chatwork 共有投稿自動化

URL を入力して「投稿」ボタンを押すと、以下を自動生成して Chatwork に投稿する::

    ＜事業領域/サービス＞
    件名
    URL
    80文字の要約

旧フロー (＜共有＞ / 件名 / URL の 3 行手動投稿) を置き換える。

処理フロー:
  1. PageFetcher   — URL のページを取得し、件名 (title) / 説明 / 本文を抽出
  2. Summarizer    — 本文を 80 文字に要約 (LLM、無ければ rule-based フォールバック)
  3. DomainClassifier — 事業領域/サービスを 7 区分に分類 (キーワード / LLM)
  4. ChatworkClient — Chatwork API へ投稿 (トークン無ければ preview のみ)

オブジェクト:
  BusinessDomain          : 事業領域/サービスの enum (7 区分)
  PageContent             : 取得したページ内容
  SharePost               : 投稿 1 件 (.to_chatwork_message() で 4 行整形)
  PageFetcher             : URL → PageContent
  Summarizer              : PageContent → 80 文字要約
  DomainClassifier        : PageContent → BusinessDomain
  ChatworkClient          : Chatwork API クライアント
  ChatworkShareEngine     : 上記を束ねるオーケストレーター
  ChatworkShareEngineFactory : from_env / from_mock

使用例::
    engine = ChatworkShareEngineFactory.from_env()
    result = engine.share("https://example.com/article")
    print(result["message"])      # 投稿された 4 行テキスト
    print(result["posted"])       # True なら実投稿済み
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum 層
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BusinessDomain(str, Enum):
    """事業領域 / サービス区分 (7 区分)。"""
    CONSULTING     = "コンサルティング"
    DX_MARKETING   = "DX/マーケティング"
    AD_CREATIVE    = "広告/クリエイティブ"
    SEO_PRODUCTION = "SEO/製作"
    SNS            = "SNS"
    LLMO           = "LLMO"
    AI_LLMO        = "AI/LLMO"


# 分類用キーワード辞書 (優先度順に評価)。値の語が title/desc/本文に出るとスコア加算。
_DOMAIN_KEYWORDS: Dict[BusinessDomain, List[str]] = {
    BusinessDomain.AI_LLMO: [
        "生成ai", "生成 ai", "llm", "chatgpt", "gpt", "claude", "gemini",
        "機械学習", "ディープラーニング", "deep learning", "人工知能",
        "ai活用", "ai 活用", "プロンプト", "rag", "ファインチューニング",
    ],
    BusinessDomain.LLMO: [
        "llmo", "aio", "ai検索", "ai 検索", "ai overview", "aioverview",
        "回答エンジン", "answer engine", "生成エンジン最適化", "geo",
        "aiに引用", "ai に引用", "参照率", "引用率", "言及率",
    ],
    BusinessDomain.SEO_PRODUCTION: [
        "seo", "検索順位", "検索エンジン", "被リンク", "内部対策", "外部対策",
        "コンテンツseo", "サイト制作", "ホームページ制作", "web制作",
        "コーディング", "lp制作", "サイト構築", "cms",
    ],
    BusinessDomain.SNS: [
        "sns", "インスタ", "instagram", "x(twitter)", "twitter", "tiktok",
        "youtube", "facebook", "line", "インフルエンサー", "ショート動画",
        "ugc", "フォロワー", "バズ",
    ],
    BusinessDomain.AD_CREATIVE: [
        "広告", "リスティング", "ディスプレイ広告", "運用型広告", "クリエイティブ",
        "バナー", "動画広告", "広告運用", "google広告", "yahoo広告",
        "meta広告", "コピーライティング", "デザイン", "クリック率", "roas",
    ],
    BusinessDomain.DX_MARKETING: [
        "dx", "デジタルトランスフォーメーション", "マーケティング", "マーケ",
        "ma", "crm", "mawebサイト", "リード", "顧客体験", "cx",
        "データ活用", "業務効率化", "自動化", "オウンドメディア", "メルマガ",
    ],
    BusinessDomain.CONSULTING: [
        "コンサル", "コンサルティング", "経営", "戦略", "事業計画",
        "支援", "伴走", "kpi設計", "組織", "新規事業", "業務改善",
    ],
}

# 全候補に該当しなかった場合のデフォルト
_DEFAULT_DOMAIN = BusinessDomain.DX_MARKETING

# 要約の最大文字数 (要件: 80 文字)
DEFAULT_SUMMARY_CHARS = 80

# 既定の投稿先 Chatwork ルーム (https://www.chatwork.com/#!rid336261448)
DEFAULT_ROOM_ID = "336261448"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PageContent:
    """取得したページ内容。"""
    url:         str
    title:       str   = ""
    description: str   = ""
    text:        str   = ""
    fetched_at:  float = 0.0
    ok:          bool  = True
    error:       str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "text": self.text[:500],
            "fetched_at": self.fetched_at,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class SharePost:
    """Chatwork に投稿する共有 1 件。"""
    domain:  BusinessDomain
    subject: str
    url:     str
    summary: str

    def to_chatwork_message(self) -> str:
        """4 行整形: ＜事業領域/サービス＞ / 件名 / URL / 80文字要約。"""
        return (
            f"＜{self.domain.value}＞\n"
            f"{self.subject}\n"
            f"{self.url}\n"
            f"{self.summary}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "subject": self.subject,
            "url": self.url,
            "summary": self.summary,
            "message": self.to_chatwork_message(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML パース ユーティリティ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_TAG_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|template)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_ANY = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    return _WS.sub(" ", _html.unescape(s)).strip()


def _meta_content(htmltext: str, *, prop: str = "", name: str = "") -> str:
    """meta[property=...] または meta[name=...] の content を取り出す。"""
    attr, val = ("property", prop) if prop else ("name", name)
    # content が property より前後どちらにあっても拾えるよう 2 パターン試す
    patterns = [
        rf'<meta[^>]*\b{attr}=["\']{re.escape(val)}["\'][^>]*\bcontent=["\']([^"\']*)["\']',
        rf'<meta[^>]*\bcontent=["\']([^"\']*)["\'][^>]*\b{attr}=["\']{re.escape(val)}["\']',
    ]
    for pat in patterns:
        m = re.search(pat, htmltext, re.IGNORECASE)
        if m:
            return _clean_text(m.group(1))
    return ""


def parse_html(url: str, htmltext: str) -> PageContent:
    """生 HTML から title / description / 本文を抽出する。"""
    # title: og:title > <title> > 最初の <h1>
    title = _meta_content(htmltext, prop="og:title")
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", htmltext, re.IGNORECASE | re.DOTALL)
        if m:
            title = _clean_text(m.group(1))
    if not title:
        m = re.search(r"<h1[^>]*>(.*?)</h1>", htmltext, re.IGNORECASE | re.DOTALL)
        if m:
            title = _clean_text(_TAG_ANY.sub("", m.group(1)))

    # description: meta description > og:description
    description = (
        _meta_content(htmltext, name="description")
        or _meta_content(htmltext, prop="og:description")
    )

    # 本文: script/style を除去 → 全タグ除去 → 空白正規化
    body = _TAG_SCRIPT_STYLE.sub(" ", htmltext)
    body = _TAG_ANY.sub(" ", body)
    text = _clean_text(body)

    return PageContent(
        url=url,
        title=title,
        description=description,
        text=text,
        fetched_at=time.time(),
        ok=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PageFetcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PageFetcher:
    """URL からページを取得し PageContent を返す。

    `fetcher` を注入すると HTTP を行わずそのコールバックの返す HTML を使う
    (テスト用 / 外部依存なし)。
    """

    _UA = "OpenMythos-ChatworkPoster/1.0 (+https://github.com/)"

    def __init__(
        self,
        fetcher: Optional[Callable[[str], str]] = None,
        timeout: int = 15,
        max_bytes: int = 2_000_000,
    ) -> None:
        self._fetcher = fetcher
        self.timeout = timeout
        self.max_bytes = max_bytes

    def _http_get(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self._UA})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            raw = resp.read(self.max_bytes)
            charset = resp.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return raw.decode("utf-8", errors="replace")

    def fetch(self, url: str) -> PageContent:
        if not re.match(r"^https?://", url.strip(), re.IGNORECASE):
            return PageContent(url=url, ok=False, error="invalid_url")
        try:
            htmltext = self._fetcher(url) if self._fetcher else self._http_get(url)
        except urllib.error.URLError as e:  # pragma: no cover - ネットワーク依存
            return PageContent(url=url, ok=False, error=f"fetch_error: {e}")
        except Exception as e:  # noqa: BLE001
            return PageContent(url=url, ok=False, error=f"fetch_error: {e}")
        return parse_html(url, htmltext)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summarizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _truncate(text: str, max_chars: int) -> str:
    """max_chars 以内に収める。超過時は末尾を … にする。"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1].rstrip() + "…"


class Summarizer:
    """ページ本文を指定文字数に要約する。

    LLM (MultiProviderRouter) が利用可能なら使い、
    無ければ rule-based (説明文 / 冒頭文の抜粋) にフォールバックする。
    いずれの場合も `max_chars` を厳守する。
    """

    def __init__(
        self,
        router: Any = None,
        max_chars: int = DEFAULT_SUMMARY_CHARS,
    ) -> None:
        self.router = router
        self.max_chars = max_chars

    def _llm_available(self) -> bool:
        if self.router is None:
            return False
        try:
            return bool(self.router.available_providers())
        except Exception:  # noqa: BLE001
            return False

    def _rule_based(self, content: PageContent) -> str:
        base = content.description.strip() or content.text.strip()
        if not base:
            base = content.title.strip()
        # 文単位で max_chars に収まるよう足していく
        sentences = re.split(r"(?<=[。.!?！？])", base)
        out = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(out) + len(s) > self.max_chars:
                break
            out += s
        if not out:
            out = base
        return _truncate(out, self.max_chars)

    def summarize(self, content: PageContent) -> str:
        if self._llm_available():
            try:
                return self._llm_summarize(content)
            except Exception:  # noqa: BLE001
                pass  # フォールバック
        return self._rule_based(content)

    def _llm_summarize(self, content: PageContent) -> str:
        from open_mythos.skills.llm_providers import LLMRequest

        source = (content.description + " " + content.text).strip()[:2000]
        system = (
            "あなたは日本語の編集者です。与えられた記事を、"
            f"日本語で必ず{self.max_chars}文字以内に要約してください。"
            "前置き・記号・改行は不要で、要約本文のみを出力します。"
        )
        prompt = f"タイトル: {content.title}\n本文: {source}\n\n{self.max_chars}文字以内の要約:"
        resp = self.router.complete(
            LLMRequest(prompt=prompt, system=system, max_tokens=200, temperature=0.3)
        )
        summary = _clean_text(resp.text)
        if not summary:
            return self._rule_based(content)
        return _truncate(summary, self.max_chars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DomainClassifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DomainClassifier:
    """ページ内容を 7 区分の事業領域/サービスに分類する。

    キーワードマッチでスコアリングし、最高スコアの区分を返す。
    title の一致は本文より重く評価する。同点・該当なしは既定区分。
    """

    def __init__(self, router: Any = None) -> None:
        self.router = router

    def scores(self, content: PageContent) -> Dict[BusinessDomain, int]:
        title = content.title.lower()
        haystack = (content.title + " " + content.description + " " + content.text).lower()
        result: Dict[BusinessDomain, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                k = kw.lower()
                if k in title:
                    score += 3
                if k in haystack:
                    score += 1
            result[domain] = score
        return result

    def classify(self, content: PageContent) -> BusinessDomain:
        scores = self.scores(content)
        best = max(scores.values()) if scores else 0
        if best <= 0:
            return _DEFAULT_DOMAIN
        # _DOMAIN_KEYWORDS の定義順 (優先度順) でタイブレーク
        for domain in _DOMAIN_KEYWORDS:
            if scores[domain] == best:
                return domain
        return _DEFAULT_DOMAIN


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChatworkClient
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatworkClient:
    """Chatwork API クライアント (rooms/{id}/messages へ投稿)。

    `token` 未設定かつ `poster` 未注入なら投稿不可 (is_available() == False)。
    `poster` を注入するとテストで HTTP を回避できる。
    """

    API_BASE = "https://api.chatwork.com/v2"

    def __init__(
        self,
        token: Optional[str] = None,
        poster: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        timeout: int = 15,
        api_base: Optional[str] = None,
    ) -> None:
        self.token = token
        self._poster = poster
        self.timeout = timeout
        self.api_base = api_base or self.API_BASE

    def is_available(self) -> bool:
        return bool(self._poster) or bool(self.token)

    def post_message(self, room_id: str, body: str) -> Dict[str, Any]:
        if self._poster is not None:
            return self._poster(room_id, body)
        if not self.token:
            raise RuntimeError("CHATWORK_API_TOKEN が未設定です")
        url = f"{self.api_base}/rooms/{room_id}/messages"
        data = urllib.parse.urlencode({"body": body}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "X-ChatWorkToken": self.token,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"raw": raw}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChatworkShareEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatworkShareEngine:
    """URL → 4 行整形 → Chatwork 投稿 のオーケストレーター。"""

    def __init__(
        self,
        fetcher: PageFetcher,
        summarizer: Summarizer,
        classifier: DomainClassifier,
        client: ChatworkClient,
        default_room_id: str = DEFAULT_ROOM_ID,
    ) -> None:
        self.fetcher = fetcher
        self.summarizer = summarizer
        self.classifier = classifier
        self.client = client
        self.default_room_id = default_room_id

    def build_post(
        self,
        url: str,
        *,
        domain_override: Optional[BusinessDomain] = None,
        subject_override: Optional[str] = None,
    ) -> SharePost:
        """URL から SharePost を組み立てる (投稿はしない)。"""
        content = self.fetcher.fetch(url)
        if not content.ok:
            raise ValueError(f"ページ取得に失敗しました: {content.error} ({url})")

        subject = (subject_override or content.title or url).strip()
        summary = self.summarizer.summarize(content)
        domain = domain_override or self.classifier.classify(content)
        return SharePost(domain=domain, subject=subject, url=url, summary=summary)

    def share(
        self,
        url: str,
        *,
        room_id: Optional[str] = None,
        dry_run: bool = False,
        domain_override: Optional[BusinessDomain] = None,
        subject_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """URL を投稿する。

        dry_run=True もしくは Chatwork トークン未設定の場合は投稿せず
        プレビュー (posted=False) を返す。
        """
        post = self.build_post(
            url,
            domain_override=domain_override,
            subject_override=subject_override,
        )
        room = room_id or self.default_room_id
        message = post.to_chatwork_message()

        result: Dict[str, Any] = {
            "posted": False,
            "room_id": room,
            "message": message,
            "post": post.to_dict(),
        }

        if dry_run:
            result["reason"] = "dry_run"
            return result
        if not self.client.is_available():
            result["reason"] = "no_token"
            return result

        api_resp = self.client.post_message(room, message)
        result["posted"] = True
        result["chatwork_response"] = api_resp
        if isinstance(api_resp, dict) and "message_id" in api_resp:
            result["message_id"] = api_resp["message_id"]
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatworkShareEngineFactory:
    """ChatworkShareEngine の生成口。"""

    @staticmethod
    def from_env(
        *,
        max_chars: int = DEFAULT_SUMMARY_CHARS,
        room_id: Optional[str] = None,
    ) -> ChatworkShareEngine:
        """環境変数からエンジンを構築する。

        - CHATWORK_API_TOKEN : 投稿用トークン (無ければ preview のみ)
        - CHATWORK_ROOM_ID   : 投稿先ルーム (既定: 336261448)
        - ANTHROPIC/OPENAI_API_KEY : LLM 要約 (無ければ rule-based)
        """
        router = None
        try:
            from open_mythos.skills.llm_providers import MultiProviderRouter
            router = MultiProviderRouter.from_env()
        except Exception:  # noqa: BLE001
            router = None

        token = os.getenv("CHATWORK_API_TOKEN")
        room = room_id or os.getenv("CHATWORK_ROOM_ID") or DEFAULT_ROOM_ID
        return ChatworkShareEngine(
            fetcher=PageFetcher(),
            summarizer=Summarizer(router=router, max_chars=max_chars),
            classifier=DomainClassifier(router=router),
            client=ChatworkClient(token=token),
            default_room_id=room,
        )

    @staticmethod
    def from_mock(
        *,
        fetcher: Optional[Callable[[str], str]] = None,
        poster: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        router: Any = None,
        max_chars: int = DEFAULT_SUMMARY_CHARS,
        room_id: str = DEFAULT_ROOM_ID,
    ) -> ChatworkShareEngine:
        """テスト用。HTTP を行わず注入したコールバックを使う。"""
        return ChatworkShareEngine(
            fetcher=PageFetcher(fetcher=fetcher or (lambda u: "")),
            summarizer=Summarizer(router=router, max_chars=max_chars),
            classifier=DomainClassifier(router=router),
            client=ChatworkClient(poster=poster or (lambda r, b: {"message_id": "mock-1"})),
            default_room_id=room_id,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# テスト用モック
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _MockLLMRouter:
    """LLM 要約パスの単体テスト用ルーター。"""

    def __init__(self, text: str = "これはモック要約です。") -> None:
        self._text = text

    def available_providers(self) -> List[str]:
        return ["mock"]

    def complete(self, req: Any) -> Any:
        from open_mythos.skills.llm_providers import LLMResponse
        return LLMResponse(text=self._text, provider_used="mock", model="mock")
