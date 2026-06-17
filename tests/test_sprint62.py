"""
Sprint 62 — LLM コピー生成強化 テスト (56 tests)

対象:
  open_mythos/skills/llm_copy_generator.py:
    CopyGenerationConfig
    CopyGenerationResult
    LLMCopyPromptBuilder
    LLMCopyParser
    LLMCopyGenerator
    LLMCopyGeneratorFactory
"""
from __future__ import annotations

import json
import pytest

from open_mythos.skills.campaign_manager import (
    AdChannel, AdCopy, AdObjective,
)
from open_mythos.skills.llm_copy_generator import (
    CopyGenerationConfig,
    CopyGenerationResult,
    LLMCopyPromptBuilder,
    LLMCopyParser,
    LLMCopyGenerator,
    LLMCopyGeneratorFactory,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CopyGenerationConfig
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCopyGenerationConfig:
    def test_defaults(self):
        cfg = CopyGenerationConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens  == 512
        assert cfg.max_retries == 2
        assert cfg.fallback_on_error is True
        assert cfg.preferred_provider is None

    def test_custom(self):
        cfg = CopyGenerationConfig(temperature=0.3, max_tokens=256, max_retries=1)
        assert cfg.temperature == 0.3
        assert cfg.max_tokens  == 256
        assert cfg.max_retries == 1

    def test_preferred_provider(self):
        cfg = CopyGenerationConfig(preferred_provider="claude")
        assert cfg.preferred_provider == "claude"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CopyGenerationResult
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_result(provider="rule-based", fallback=False) -> CopyGenerationResult:
    copy = AdCopy(
        id="test-id",
        headline="テスト見出し",
        body="テスト本文です。",
        cta="今すぐチェック",
    )
    return CopyGenerationResult(copy=copy, provider_used=provider, fallback_used=fallback)


class TestCopyGenerationResult:
    def test_to_dict_keys(self):
        r = _make_result()
        d = r.to_dict()
        assert "copy" in d
        assert "provider_used" in d
        assert "latency_ms" in d
        assert "fallback_used" in d

    def test_provider_used(self):
        r = _make_result(provider="llm:claude")
        assert r.to_dict()["provider_used"] == "llm:claude"

    def test_fallback_used_false(self):
        r = _make_result(fallback=False)
        assert r.fallback_used is False

    def test_fallback_used_true(self):
        r = _make_result(fallback=True)
        assert r.fallback_used is True

    def test_latency_rounded(self):
        r = _make_result()
        r.latency_ms = 123.456789
        assert r.to_dict()["latency_ms"] == 123.46


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyPromptBuilder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLLMCopyPromptBuilder:
    def setup_method(self):
        self.builder = LLMCopyPromptBuilder()

    def test_build_user_prompt_contains_scenario(self):
        prompt = self.builder.build_user_prompt(
            "コスト削減したい", AdObjective.CONVERSION, AdChannel.SEARCH, "TestBrand"
        )
        assert "コスト削減したい" in prompt

    def test_build_user_prompt_contains_brand(self):
        prompt = self.builder.build_user_prompt(
            "テスト", AdObjective.AWARENESS, AdChannel.SOCIAL, "MyBrand"
        )
        assert "MyBrand" in prompt

    def test_build_user_prompt_contains_objective(self):
        prompt = self.builder.build_user_prompt(
            "テスト", AdObjective.CONVERSION, AdChannel.SEARCH, "Brand"
        )
        assert "コンバージョン" in prompt

    def test_build_user_prompt_contains_channel(self):
        prompt = self.builder.build_user_prompt(
            "テスト", AdObjective.AWARENESS, AdChannel.EMAIL, "Brand"
        )
        assert "メール" in prompt

    def test_build_user_prompt_extra(self):
        prompt = self.builder.build_user_prompt(
            "テスト", AdObjective.AWARENESS, AdChannel.SEARCH, "Brand",
            extra={"discount": "50%"}
        )
        assert "50%" in prompt

    def test_system_prompt_not_empty(self):
        assert len(self.builder.system_prompt) > 0

    def test_system_prompt_contains_json(self):
        assert "JSON" in self.builder.system_prompt

    def test_all_objectives_in_prompt(self):
        for obj in AdObjective:
            prompt = self.builder.build_user_prompt("t", obj, AdChannel.SEARCH, "B")
            assert len(prompt) > 0

    def test_all_channels_in_prompt(self):
        for ch in AdChannel:
            prompt = self.builder.build_user_prompt("t", AdObjective.AWARENESS, ch, "B")
            assert len(prompt) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyParser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VALID_JSON = json.dumps({
    "headline": "LLM生成見出し",
    "body": "LLMが生成した広告本文です。品質が高い。",
    "cta": "今すぐ試す",
    "tags": ["LLM", "広告"],
})

class TestLLMCopyParser:
    def setup_method(self):
        self.parser = LLMCopyParser()

    def test_parse_valid_json(self):
        result = self.parser.parse(_VALID_JSON, AdChannel.SEARCH, AdObjective.AWARENESS)
        assert result["headline"] == "LLM生成見出し"
        assert result["body"]     == "LLMが生成した広告本文です。品質が高い。"
        assert result["cta"]      == "今すぐ試す"
        assert result["tags"]     == ["LLM", "広告"]

    def test_parse_code_block_json(self):
        raw = f"```json\n{_VALID_JSON}\n```"
        result = self.parser.parse(raw, AdChannel.SEARCH, AdObjective.AWARENESS)
        assert result["headline"] == "LLM生成見出し"

    def test_parse_code_block_no_lang(self):
        raw = f"```\n{_VALID_JSON}\n```"
        result = self.parser.parse(raw, AdChannel.SEARCH, AdObjective.AWARENESS)
        assert result["headline"] == "LLM生成見出し"

    def test_parse_regex_fallback(self):
        raw = '"headline": "正規表現見出し", "body": "本文テスト", "cta": "確認する"'
        result = self.parser.parse(
            raw, AdChannel.SEARCH, AdObjective.AWARENESS,
            fallback_headline="fb"
        )
        assert result["headline"] == "正規表現見出し"

    def test_parse_fallback_on_invalid(self):
        result = self.parser.parse(
            "全く解析できないテキスト",
            AdChannel.SEARCH, AdObjective.AWARENESS,
            fallback_headline="フォールバック",
            fallback_body="フォールバック本文",
            fallback_cta="詳しく見る",
        )
        assert result["headline"] == "フォールバック"

    def test_parse_headline_truncated(self):
        long_json = json.dumps({"headline": "あ" * 50, "body": "b", "cta": "c", "tags": []})
        result = self.parser.parse(long_json, AdChannel.SEARCH, AdObjective.AWARENESS)
        assert len(result["headline"]) <= 40

    def test_parse_tags_list(self):
        result = self.parser.parse(_VALID_JSON, AdChannel.SEARCH, AdObjective.AWARENESS)
        assert isinstance(result["tags"], list)

    def test_parse_empty_response(self):
        result = self.parser.parse(
            "", AdChannel.SEARCH, AdObjective.AWARENESS,
            fallback_headline="empty", fallback_body="body", fallback_cta="cta"
        )
        assert result["headline"] == "empty"

    def test_parse_non_list_tags_normalized(self):
        data = json.dumps({"headline": "h", "body": "b", "cta": "c", "tags": "not-a-list"})
        result = self.parser.parse(data, AdChannel.SEARCH, AdObjective.AWARENESS)
        assert isinstance(result["tags"], list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyGenerator — rule-based モード
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLLMCopyGeneratorRuleBased:
    def setup_method(self):
        self.gen = LLMCopyGeneratorFactory.rule_based(brand="TestBrand")

    def test_generate_returns_ad_copy(self):
        copy = self.gen.generate_from_scenario("コスト削減したい")
        assert isinstance(copy, AdCopy)

    def test_provider_used_rule_based(self):
        result = self.gen.generate_with_meta("テスト")
        assert result.provider_used == "rule-based"

    def test_fallback_used_false_when_no_llm(self):
        result = self.gen.generate_with_meta("テスト")
        assert result.fallback_used is False

    def test_has_llm_false_when_no_router(self):
        assert self.gen.has_llm is False

    def test_generate_batch_with_meta(self):
        results = self.gen.generate_batch_with_meta(
            "テスト", channels=[AdChannel.SEARCH, AdChannel.EMAIL]
        )
        assert len(results) == 2
        assert all(isinstance(r, CopyGenerationResult) for r in results)

    def test_backward_compat_generate_batch(self):
        copies = self.gen.generate_batch("テスト")
        assert len(copies) == 2
        assert all(isinstance(c, AdCopy) for c in copies)

    def test_headline_not_empty(self):
        copy = self.gen.generate_from_scenario("シナリオ")
        assert len(copy.headline) > 0

    def test_objective_preserved(self):
        result = self.gen.generate_with_meta("テスト", objective=AdObjective.CONVERSION)
        assert result.copy.objective == AdObjective.CONVERSION

    def test_channel_preserved(self):
        result = self.gen.generate_with_meta("テスト", channel=AdChannel.VIDEO)
        assert result.copy.channel == AdChannel.VIDEO


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyGenerator — LLM モック
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MOCK_RESPONSE = json.dumps({
    "headline": "AI生成の見出し",
    "body": "AI が生成した高品質な広告コピーです。",
    "cta": "今すぐ申し込む",
    "tags": ["AI", "テスト"],
})


class TestLLMCopyGeneratorWithMock:
    def setup_method(self):
        self.gen = LLMCopyGeneratorFactory.from_mock(
            responses=[_MOCK_RESPONSE],
            brand="MockBrand",
        )

    def test_has_llm_true_with_mock(self):
        assert self.gen.has_llm is True

    def test_generate_returns_ad_copy(self):
        copy = self.gen.generate_from_scenario("テスト")
        assert isinstance(copy, AdCopy)

    def test_headline_from_llm(self):
        copy = self.gen.generate_from_scenario("テスト")
        assert copy.headline == "AI生成の見出し"

    def test_body_from_llm(self):
        copy = self.gen.generate_from_scenario("テスト")
        assert copy.body == "AI が生成した高品質な広告コピーです。"

    def test_cta_from_llm(self):
        copy = self.gen.generate_from_scenario("テスト")
        assert copy.cta == "今すぐ申し込む"

    def test_tags_from_llm(self):
        copy = self.gen.generate_from_scenario("テスト")
        assert "AI" in copy.tags

    def test_provider_used_is_llm(self):
        result = self.gen.generate_with_meta("テスト")
        assert result.provider_used.startswith("llm:")

    def test_fallback_false_when_llm_succeeds(self):
        result = self.gen.generate_with_meta("テスト")
        assert result.fallback_used is False

    def test_raw_response_stored(self):
        result = self.gen.generate_with_meta("テスト")
        assert len(result.raw_response) > 0

    def test_multiple_calls_use_responses_cyclically(self):
        gen = LLMCopyGeneratorFactory.from_mock(
            responses=[_MOCK_RESPONSE, json.dumps({
                "headline": "2回目見出し", "body": "2回目本文テストです。", "cta": "確認する", "tags": []
            })],
        )
        r1 = gen.generate_with_meta("テスト1")
        r2 = gen.generate_with_meta("テスト2")
        assert r1.copy.headline != r2.copy.headline


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyGenerator — フォールバック動作
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ErrorRouter:
    """常にエラーを投げるモックルーター"""
    def available_providers(self):
        return ["error-mock"]
    def complete(self, req, preferred=None):
        raise RuntimeError("意図的なエラー")


class TestLLMCopyGeneratorFallback:
    def test_fallback_on_error_true(self):
        gen = LLMCopyGenerator(
            brand="Brand",
            config=CopyGenerationConfig(fallback_on_error=True, max_retries=1),
            router=_ErrorRouter(),
        )
        result = gen.generate_with_meta("テスト")
        assert result.provider_used == "rule-based"
        assert result.fallback_used is True

    def test_fallback_error_in_raw_response(self):
        gen = LLMCopyGenerator(
            brand="Brand",
            config=CopyGenerationConfig(fallback_on_error=True, max_retries=1),
            router=_ErrorRouter(),
        )
        result = gen.generate_with_meta("テスト")
        assert "LLM error" in result.raw_response

    def test_no_fallback_raises(self):
        gen = LLMCopyGenerator(
            brand="Brand",
            config=CopyGenerationConfig(fallback_on_error=False, max_retries=1),
            router=_ErrorRouter(),
        )
        with pytest.raises(RuntimeError):
            gen.generate_with_meta("テスト")

    def test_copy_still_valid_after_fallback(self):
        gen = LLMCopyGenerator(
            brand="Brand",
            config=CopyGenerationConfig(fallback_on_error=True, max_retries=1),
            router=_ErrorRouter(),
        )
        result = gen.generate_with_meta("テスト")
        assert isinstance(result.copy, AdCopy)
        assert len(result.copy.headline) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyGeneratorFactory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLLMCopyGeneratorFactory:
    def test_rule_based_returns_generator(self):
        gen = LLMCopyGeneratorFactory.rule_based()
        assert isinstance(gen, LLMCopyGenerator)

    def test_rule_based_no_llm(self):
        gen = LLMCopyGeneratorFactory.rule_based()
        assert gen.has_llm is False

    def test_from_mock_returns_generator(self):
        gen = LLMCopyGeneratorFactory.from_mock(responses=[_MOCK_RESPONSE])
        assert isinstance(gen, LLMCopyGenerator)

    def test_from_mock_has_llm(self):
        gen = LLMCopyGeneratorFactory.from_mock(responses=[_MOCK_RESPONSE])
        assert gen.has_llm is True

    def test_from_mock_brand_set(self):
        gen = LLMCopyGeneratorFactory.from_mock(responses=[_MOCK_RESPONSE], brand="MyBrand")
        assert gen.brand == "MyBrand"

    def test_from_env_returns_generator(self):
        # API キーがない環境でも構築できる
        gen = LLMCopyGeneratorFactory.from_env(brand="TestBrand")
        assert isinstance(gen, LLMCopyGenerator)

    def test_from_mock_with_config(self):
        cfg = CopyGenerationConfig(temperature=0.3)
        gen = LLMCopyGeneratorFactory.from_mock(
            responses=[_MOCK_RESPONSE], config=cfg
        )
        assert gen._config.temperature == 0.3
