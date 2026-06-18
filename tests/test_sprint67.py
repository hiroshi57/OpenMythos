"""
Sprint 67 — Chatwork 共有投稿自動化 テスト

対象: open_mythos/skills/chatwork_poster.py
  - HTML パース (title / description / 本文)
  - Summarizer (rule-based / LLM / 80文字厳守)
  - DomainClassifier (7 区分のキーワード分類)
  - ChatworkClient (token / poster の可用性)
  - ChatworkShareEngine (4 行整形 / dry_run / no_token / 実投稿)
"""
from __future__ import annotations

import pytest

from open_mythos.skills.chatwork_poster import (
    BusinessDomain,
    PageContent,
    SharePost,
    PageFetcher,
    Summarizer,
    DomainClassifier,
    ChatworkClient,
    ChatworkShareEngine,
    ChatworkShareEngineFactory,
    parse_html,
    _truncate,
    _MockLLMRouter,
    DEFAULT_ROOM_ID,
    DEFAULT_SUMMARY_CHARS,
)


# ━━━ サンプル HTML ━━━

_HTML_AI = """
<html><head>
  <title>生成AIで広告コピーを量産する方法 | OpenMythos</title>
  <meta name="description" content="ChatGPTやClaudeなどのLLMを使い、広告コピーを効率的に量産する手法を解説します。プロンプト設計のコツも紹介。">
  <meta property="og:title" content="生成AIで広告コピーを量産する方法">
</head><body>
  <script>var x = 1;</script>
  <h1>生成AIコピー量産</h1>
  <p>本文テキストです。機械学習を活用したマーケティング自動化について述べます。</p>
</body></html>
"""

_HTML_SEO = """
<html><head>
  <title>SEO内部対策の基本：検索順位を上げるサイト制作</title>
</head><body>
  <p>被リンクとコンテンツSEOで検索エンジンの評価を高めるホームページ制作の話。</p>
</body></html>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML パース
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_parse_html_extracts_title_via_og():
    c = parse_html("https://x.test/a", _HTML_AI)
    assert c.ok
    assert c.title == "生成AIで広告コピーを量産する方法"  # og:title 優先


def test_parse_html_title_fallback_to_title_tag():
    c = parse_html("https://x.test/b", _HTML_SEO)
    assert c.title == "SEO内部対策の基本：検索順位を上げるサイト制作"


def test_parse_html_extracts_description():
    c = parse_html("https://x.test/a", _HTML_AI)
    assert "LLM" in c.description
    assert "広告コピー" in c.description


def test_parse_html_strips_script_and_tags():
    c = parse_html("https://x.test/a", _HTML_AI)
    assert "var x" not in c.text
    assert "<p>" not in c.text
    assert "本文テキストです" in c.text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _truncate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_truncate_under_limit_unchanged():
    assert _truncate("短い文", 80) == "短い文"


def test_truncate_over_limit_adds_ellipsis():
    s = "あ" * 100
    out = _truncate(s, 80)
    assert len(out) == 80
    assert out.endswith("…")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summarizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_summarizer_rule_based_uses_description():
    c = parse_html("https://x.test/a", _HTML_AI)
    s = Summarizer(router=None, max_chars=80).summarize(c)
    assert s
    assert len(s) <= 80


def test_summarizer_respects_max_chars():
    long_desc = "テスト。" * 100
    c = PageContent(url="u", title="t", description=long_desc, text="")
    s = Summarizer(router=None, max_chars=80).summarize(c)
    assert len(s) <= 80


def test_summarizer_llm_path_used_when_available():
    c = parse_html("https://x.test/a", _HTML_AI)
    router = _MockLLMRouter("LLMが生成した要約テキストです。")
    s = Summarizer(router=router, max_chars=80).summarize(c)
    assert s == "LLMが生成した要約テキストです。"


def test_summarizer_llm_output_truncated_to_max():
    c = parse_html("https://x.test/a", _HTML_AI)
    router = _MockLLMRouter("あ" * 200)
    s = Summarizer(router=router, max_chars=80).summarize(c)
    assert len(s) == 80


def test_summarizer_falls_back_when_no_providers():
    class _Empty:
        def available_providers(self):
            return []
    c = parse_html("https://x.test/a", _HTML_AI)
    s = Summarizer(router=_Empty(), max_chars=80).summarize(c)
    assert s  # rule-based で生成される
    assert len(s) <= 80


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DomainClassifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_classifier_detects_ai_llmo():
    c = parse_html("https://x.test/a", _HTML_AI)
    assert DomainClassifier().classify(c) == BusinessDomain.AI_LLMO


def test_classifier_detects_seo():
    c = parse_html("https://x.test/b", _HTML_SEO)
    assert DomainClassifier().classify(c) == BusinessDomain.SEO_PRODUCTION


def test_classifier_detects_sns():
    c = PageContent(url="u", title="Instagram運用でフォロワーを増やすSNS戦略", text="")
    assert DomainClassifier().classify(c) == BusinessDomain.SNS


def test_classifier_detects_consulting():
    c = PageContent(url="u", title="経営戦略コンサルティングで新規事業を支援", text="")
    assert DomainClassifier().classify(c) == BusinessDomain.CONSULTING


def test_classifier_defaults_when_no_match():
    c = PageContent(url="u", title="今日のランチ日記", text="美味しいラーメンを食べた")
    assert DomainClassifier().classify(c) == BusinessDomain.DX_MARKETING


def test_classifier_title_weighted_over_body():
    scores = DomainClassifier().scores(
        PageContent(url="u", title="SNSマーケティング", text="")
    )
    assert scores[BusinessDomain.SNS] >= 3  # title 一致は +3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SharePost 整形
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_sharepost_message_format_four_lines():
    post = SharePost(
        domain=BusinessDomain.AI_LLMO,
        subject="件名サンプル",
        url="https://x.test/a",
        summary="要約サンプル",
    )
    lines = post.to_chatwork_message().split("\n")
    assert len(lines) == 4
    assert lines[0] == "＜AI/LLMO＞"
    assert lines[1] == "件名サンプル"
    assert lines[2] == "https://x.test/a"
    assert lines[3] == "要約サンプル"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChatworkClient
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_client_unavailable_without_token():
    assert ChatworkClient().is_available() is False


def test_client_available_with_token():
    assert ChatworkClient(token="abc").is_available() is True


def test_client_post_raises_without_token():
    with pytest.raises(RuntimeError):
        ChatworkClient().post_message("1", "body")


def test_client_uses_injected_poster():
    captured = {}
    def poster(room, body):
        captured["room"] = room
        captured["body"] = body
        return {"message_id": "42"}
    client = ChatworkClient(poster=poster)
    res = client.post_message("336261448", "hello")
    assert res["message_id"] == "42"
    assert captured["room"] == "336261448"
    assert captured["body"] == "hello"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChatworkShareEngine (E2E, mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _engine_with_posts(posts: list, html: str = _HTML_AI, has_token: bool = True):
    fetcher = PageFetcher(fetcher=lambda u: html)
    summarizer = Summarizer(router=None, max_chars=80)
    classifier = DomainClassifier()
    if has_token:
        client = ChatworkClient(poster=lambda r, b: posts.append((r, b)) or {"message_id": "m1"})
    else:
        client = ChatworkClient()  # no token, no poster
    return ChatworkShareEngine(fetcher, summarizer, classifier, client)


def test_engine_build_post_full_flow():
    engine = _engine_with_posts([])
    post = engine.build_post("https://x.test/a")
    assert post.domain == BusinessDomain.AI_LLMO
    assert post.subject == "生成AIで広告コピーを量産する方法"
    assert post.url == "https://x.test/a"
    assert len(post.summary) <= 80


def test_engine_share_posts_when_available():
    posts: list = []
    engine = _engine_with_posts(posts, has_token=True)
    res = engine.share("https://x.test/a")
    assert res["posted"] is True
    assert res["message_id"] == "m1"
    assert len(posts) == 1
    assert posts[0][1] == res["message"]  # 投稿本文一致


def test_engine_share_dry_run_does_not_post():
    posts: list = []
    engine = _engine_with_posts(posts, has_token=True)
    res = engine.share("https://x.test/a", dry_run=True)
    assert res["posted"] is False
    assert res["reason"] == "dry_run"
    assert posts == []


def test_engine_share_no_token_returns_preview():
    engine = _engine_with_posts([], has_token=False)
    res = engine.share("https://x.test/a")
    assert res["posted"] is False
    assert res["reason"] == "no_token"
    assert res["message"].startswith("＜AI/LLMO＞")


def test_engine_domain_override():
    engine = _engine_with_posts([])
    post = engine.build_post("https://x.test/a", domain_override=BusinessDomain.SNS)
    assert post.domain == BusinessDomain.SNS


def test_engine_subject_override():
    engine = _engine_with_posts([])
    post = engine.build_post("https://x.test/a", subject_override="手動件名")
    assert post.subject == "手動件名"


def test_engine_fetch_failure_raises():
    engine = _engine_with_posts([])
    with pytest.raises(ValueError):
        engine.build_post("not-a-url")


def test_engine_uses_default_room():
    posts: list = []
    engine = _engine_with_posts(posts, has_token=True)
    res = engine.share("https://x.test/a")
    assert res["room_id"] == DEFAULT_ROOM_ID


def test_engine_custom_room():
    posts: list = []
    engine = _engine_with_posts(posts, has_token=True)
    res = engine.share("https://x.test/a", room_id="999")
    assert res["room_id"] == "999"
    assert posts[0][0] == "999"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_factory_from_mock_e2e():
    posted: list = []
    engine = ChatworkShareEngineFactory.from_mock(
        fetcher=lambda u: _HTML_AI,
        poster=lambda r, b: posted.append((r, b)) or {"message_id": "mk"},
    )
    res = engine.share("https://x.test/a")
    assert res["posted"] is True
    assert res["message_id"] == "mk"
    assert posted[0][0] == DEFAULT_ROOM_ID


def test_factory_from_env_no_token_preview(monkeypatch):
    monkeypatch.delenv("CHATWORK_API_TOKEN", raising=False)
    engine = ChatworkShareEngineFactory.from_env()
    # ネットワークを避けるため fetcher を差し替え
    engine.fetcher = PageFetcher(fetcher=lambda u: _HTML_AI)
    res = engine.share("https://x.test/a")
    assert res["posted"] is False
    assert res["reason"] == "no_token"


def test_factory_from_env_default_room(monkeypatch):
    monkeypatch.delenv("CHATWORK_ROOM_ID", raising=False)
    engine = ChatworkShareEngineFactory.from_env()
    assert engine.default_room_id == DEFAULT_ROOM_ID
