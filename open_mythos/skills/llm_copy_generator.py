"""
Sprint 62 — LLM コピー生成強化

既存のルールベース CopyGenerator を LLM API 連携に拡張する。
API キーがない場合は自動的にルールベース生成にフォールバックする。

オブジェクト:
  CopyGenerationConfig  : LLM 生成設定 (temperature/max_tokens/provider/retry)
  CopyGenerationResult  : 生成結果 + メタデータ (provider/latency/fallback_used)
  LLMCopyPromptBuilder  : CEP シナリオ → LLM プロンプト構築
  LLMCopyParser         : LLM レスポンス → AdCopy フィールド抽出
  LLMCopyGenerator      : CopyGenerator を継承し LLM 生成 + フォールバック実装
  LLMCopyGeneratorFactory: 環境変数 / 設定からジェネレーターを構築するファクトリ

設計方針:
  - 既存 CopyGenerator との完全後方互換（API シグネチャ変更なし）
  - LLM が利用不能 / エラー時はルールベースに自動フォールバック
  - LLM レスポンスのパースはルーズマッチ (JSON / Markdown 混在対応)
  - プロバイダー選択は llm_providers.MultiProviderRouter に委ねる
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from open_mythos.skills.campaign_manager import (
    AdChannel,
    AdCopy,
    AdObjective,
    CopyGenerator,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設定・結果モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CopyGenerationConfig:
    """LLM コピー生成設定"""
    temperature:    float = 0.7
    max_tokens:     int   = 512
    max_retries:    int   = 2
    fallback_on_error: bool = True   # LLM 失敗時にルールベースに切り替えるか
    preferred_provider: Optional[str] = None  # "claude" / "openai" / None


@dataclass
class CopyGenerationResult:
    """コピー生成結果（メタデータ付き）"""
    copy:             AdCopy
    provider_used:    str        # "llm:claude" / "llm:openai" / "rule-based"
    latency_ms:       float      = 0.0
    fallback_used:    bool       = False
    raw_response:     str        = ""
    prompt_tokens:    int        = 0
    completion_tokens: int       = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy":               self.copy.to_dict(),
            "provider_used":      self.provider_used,
            "latency_ms":         round(self.latency_ms, 2),
            "fallback_used":      self.fallback_used,
            "prompt_tokens":      self.prompt_tokens,
            "completion_tokens":  self.completion_tokens,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyPromptBuilder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_OBJECTIVE_JA: Dict[AdObjective, str] = {
    AdObjective.AWARENESS:     "認知拡大（ブランドを知ってもらう）",
    AdObjective.CONSIDERATION: "比較検討（競合との差別化）",
    AdObjective.CONVERSION:    "コンバージョン（購入・申込み）",
    AdObjective.RETENTION:     "顧客維持（リピート促進）",
}

_CHANNEL_JA: Dict[AdChannel, str] = {
    AdChannel.SEARCH:  "検索広告（Google / Bing）",
    AdChannel.SOCIAL:  "SNS 広告（Twitter / Instagram 等）",
    AdChannel.DISPLAY: "ディスプレイ広告",
    AdChannel.EMAIL:   "メール広告",
    AdChannel.VIDEO:   "動画広告",
}

_SYSTEM_PROMPT = """\
あなたはプロの広告コピーライターです。
与えられた条件に従って、日本語の広告コピーを 1 件生成してください。

出力は必ず以下の JSON 形式のみで返してください:
{
  "headline": "見出し（30文字以内）",
  "body": "本文（20〜100文字）",
  "cta": "CTA（行動喚起文、10文字以内）",
  "tags": ["キーワード1", "キーワード2"]
}

JSON 以外のテキスト（前置き・説明・コードブロック）は出力しないでください。"""


class LLMCopyPromptBuilder:
    """CEP シナリオ + 目標 + チャネルから LLM プロンプトを構築する"""

    def build_user_prompt(
        self,
        scenario: str,
        objective: AdObjective,
        channel: AdChannel,
        brand: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> str:
        """ユーザー側プロンプトを組み立てる"""
        lines = [
            f"ブランド名: {brand}",
            f"CEP シナリオ: {scenario}",
            f"広告目標: {_OBJECTIVE_JA.get(objective, objective.value)}",
            f"配信チャネル: {_CHANNEL_JA.get(channel, channel.value)}",
        ]
        if extra:
            for k, v in extra.items():
                lines.append(f"{k}: {v}")
        lines += [
            "",
            "上記の条件に合った広告コピーを JSON 形式で 1 件生成してください。",
        ]
        return "\n".join(lines)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyParser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMCopyParser:
    """
    LLM レスポンスから広告コピーフィールドを抽出する。

    優先順位:
      1. JSON として parse
      2. コードブロック内の JSON を抽出して parse
      3. 正規表現でフィールドを個別抽出（フォールバック）
    """

    def parse(
        self,
        raw: str,
        channel: AdChannel,
        objective: AdObjective,
        fallback_headline: str = "",
        fallback_body: str = "",
        fallback_cta: str = "詳しく見る",
    ) -> Dict[str, Any]:
        """
        raw レスポンスから headline / body / cta / tags を返す。
        抽出失敗時は fallback 値を使用する。
        """
        # 1. JSON直接 parse
        data = self._try_json(raw)
        if data:
            return self._normalize(data, fallback_headline, fallback_body, fallback_cta)

        # 2. コードブロック内 JSON
        data = self._try_code_block(raw)
        if data:
            return self._normalize(data, fallback_headline, fallback_body, fallback_cta)

        # 3. 正規表現フォールバック
        return self._regex_extract(raw, fallback_headline, fallback_body, fallback_cta)

    # ---- 内部ヘルパー ----

    def _try_json(self, text: str) -> Optional[Dict]:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_code_block(self, text: str) -> Optional[Dict]:
        # ```json ... ``` または ``` ... ``` を探す
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if m:
            return self._try_json(m.group(1))
        return None

    def _regex_extract(
        self,
        text: str,
        fallback_headline: str,
        fallback_body: str,
        fallback_cta: str,
    ) -> Dict[str, Any]:
        def find(key: str) -> str:
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
            return m.group(1) if m else ""

        headline = find("headline") or fallback_headline
        body = find("body") or fallback_body
        cta = find("cta") or fallback_cta

        tags_m = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', text)
        tags: List[str] = []
        if tags_m:
            tags = [t.strip().strip('"') for t in tags_m.group(1).split(",") if t.strip()]

        return {"headline": headline, "body": body, "cta": cta, "tags": tags}

    def _normalize(
        self,
        data: Dict,
        fallback_headline: str,
        fallback_body: str,
        fallback_cta: str,
    ) -> Dict[str, Any]:
        headline = str(data.get("headline", "") or fallback_headline)[:40]
        body     = str(data.get("body", "")     or fallback_body)
        cta      = str(data.get("cta", "")      or fallback_cta)[:20]
        tags_raw = data.get("tags", [])
        tags = tags_raw if isinstance(tags_raw, list) else []
        return {"headline": headline, "body": body, "cta": cta, "tags": tags}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMCopyGenerator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMCopyGenerator(CopyGenerator):
    """
    LLM API を使って広告コピーを生成する。

    - 既存 CopyGenerator との完全後方互換（`generate_from_scenario` / `generate_batch`）
    - LLM が利用不能・エラー時はルールベース生成にフォールバック
    - 詳細な生成メタデータを取得するには `generate_with_meta` を使用

    Usage:
        gen = LLMCopyGenerator.from_env(brand="MyBrand")
        copy = gen.generate_from_scenario(
            "コスト削減したい企業担当者",
            objective=AdObjective.CONVERSION,
        )
        # メタデータ付き生成
        result = gen.generate_with_meta("シナリオ", objective=AdObjective.AWARENESS)
        print(result.provider_used, result.latency_ms)
    """

    def __init__(
        self,
        brand: str = "OpenMythos",
        config: Optional[CopyGenerationConfig] = None,
        router: Optional[Any] = None,   # MultiProviderRouter | None
    ) -> None:
        super().__init__(brand=brand)
        self._config  = config or CopyGenerationConfig()
        self._router  = router
        self._builder = LLMCopyPromptBuilder()
        self._parser  = LLMCopyParser()

    # ---- 公開 API ----

    def generate_from_scenario(
        self,
        scenario: str,
        objective: AdObjective = AdObjective.AWARENESS,
        channel: AdChannel = AdChannel.SEARCH,
        extra: Optional[Dict[str, str]] = None,
    ) -> AdCopy:
        """CopyGenerator 互換シグネチャ。LLM 生成 → フォールバックの順で試行する。"""
        result = self.generate_with_meta(scenario, objective, channel, extra)
        return result.copy

    def generate_with_meta(
        self,
        scenario: str,
        objective: AdObjective = AdObjective.AWARENESS,
        channel: AdChannel = AdChannel.SEARCH,
        extra: Optional[Dict[str, str]] = None,
    ) -> CopyGenerationResult:
        """
        LLM 生成を試み、結果とメタデータを返す。
        LLM が利用不能な場合はルールベースにフォールバックする。
        """
        if self._router is not None and self._is_llm_available():
            return self._generate_via_llm(scenario, objective, channel, extra)

        # LLM なし → ルールベース
        return self._generate_rule_based(scenario, objective, channel, extra, fallback=False)

    def generate_batch_with_meta(
        self,
        scenario: str,
        objective: AdObjective = AdObjective.AWARENESS,
        channels: Optional[List[AdChannel]] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> List[CopyGenerationResult]:
        """複数チャネル向けにバッチ生成（メタデータ付き）"""
        if channels is None:
            channels = [AdChannel.SEARCH, AdChannel.SOCIAL]
        return [
            self.generate_with_meta(scenario, objective, ch, extra)
            for ch in channels
        ]

    @property
    def has_llm(self) -> bool:
        """LLM プロバイダーが設定されているか"""
        return self._router is not None and self._is_llm_available()

    # ---- 内部実装 ----

    def _is_llm_available(self) -> bool:
        try:
            available = self._router.available_providers()
            return len(available) > 0
        except Exception:
            return False

    def _generate_via_llm(
        self,
        scenario: str,
        objective: AdObjective,
        channel: AdChannel,
        extra: Optional[Dict[str, str]],
    ) -> CopyGenerationResult:
        from open_mythos.skills.llm_providers import LLMRequest, ProviderType

        user_prompt = self._builder.build_user_prompt(
            scenario, objective, channel, self.brand, extra
        )

        preferred = None
        if self._config.preferred_provider:
            try:
                preferred = ProviderType(self._config.preferred_provider)
            except ValueError:
                pass

        req = LLMRequest(
            prompt=user_prompt,
            system=self._builder.system_prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        last_error: Optional[Exception] = None
        for attempt in range(max(1, self._config.max_retries)):
            try:
                t0 = time.time()
                resp = self._router.complete(req, preferred=preferred)
                latency_ms = (time.time() - t0) * 1000

                # ルールベースの fallback コピーを準備（パース失敗時用）
                rule_copy = super().generate_from_scenario(scenario, objective, channel, extra)
                parsed = self._parser.parse(
                    resp.text,
                    channel=channel,
                    objective=objective,
                    fallback_headline=rule_copy.headline,
                    fallback_body=rule_copy.body,
                    fallback_cta=rule_copy.cta,
                )

                copy = AdCopy(
                    id=str(uuid.uuid4()),
                    headline=parsed["headline"],
                    body=parsed["body"],
                    cta=parsed["cta"],
                    channel=channel,
                    objective=objective,
                    tags=parsed.get("tags", []),
                )

                return CopyGenerationResult(
                    copy=copy,
                    provider_used=f"llm:{resp.provider_used}",
                    latency_ms=latency_ms,
                    fallback_used=False,
                    raw_response=resp.text,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                )
            except Exception as e:
                last_error = e
                continue

        # 全リトライ失敗 → フォールバック
        if self._config.fallback_on_error:
            result = self._generate_rule_based(
                scenario, objective, channel, extra, fallback=True
            )
            result.raw_response = f"LLM error: {last_error}"
            return result

        raise RuntimeError(f"LLM 生成失敗 (retries={self._config.max_retries}): {last_error}")

    def _generate_rule_based(
        self,
        scenario: str,
        objective: AdObjective,
        channel: AdChannel,
        extra: Optional[Dict[str, str]],
        fallback: bool,
    ) -> CopyGenerationResult:
        t0 = time.time()
        copy = super().generate_from_scenario(scenario, objective, channel, extra)
        latency_ms = (time.time() - t0) * 1000
        return CopyGenerationResult(
            copy=copy,
            provider_used="rule-based",
            latency_ms=latency_ms,
            fallback_used=fallback,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ファクトリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMCopyGeneratorFactory:
    """
    環境変数 / 設定から LLMCopyGenerator を構築するファクトリ。

    API キーが設定されていないプロバイダーは自動スキップ。
    全て未設定の場合は LLM なし（ルールベース専用）で構築する。

    Usage:
        gen = LLMCopyGeneratorFactory.from_env(brand="MyBrand")
        gen = LLMCopyGeneratorFactory.from_mock(responses=["..."])  # テスト用
    """

    @classmethod
    def from_env(
        cls,
        brand: str = "OpenMythos",
        config: Optional[CopyGenerationConfig] = None,
        llm: Any = None,
    ) -> LLMCopyGenerator:
        """環境変数から API キーを読み込んでジェネレーターを構築する"""
        from open_mythos.skills.llm_providers import MultiProviderRouter
        router = MultiProviderRouter.from_env(llm=llm)
        return LLMCopyGenerator(brand=brand, config=config, router=router)

    @classmethod
    def from_mock(
        cls,
        responses: List[str],
        brand: str = "OpenMythos",
        config: Optional[CopyGenerationConfig] = None,
    ) -> LLMCopyGenerator:
        """
        テスト用のモックプロバイダーでジェネレーターを構築する。

        responses: LLM が順に返すレスポンス文字列リスト
        """
        router = _MockRouter(responses)
        return LLMCopyGenerator(brand=brand, config=config, router=router)

    @classmethod
    def rule_based(
        cls,
        brand: str = "OpenMythos",
        config: Optional[CopyGenerationConfig] = None,
    ) -> LLMCopyGenerator:
        """LLM なし（ルールベース専用）でジェネレーターを構築する"""
        return LLMCopyGenerator(brand=brand, config=config, router=None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# テスト用モックルーター（内部使用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _MockProvider:
    """テスト用プロバイダー"""

    def __init__(self, responses: List[str]) -> None:
        from open_mythos.skills.llm_providers import ProviderConfig, ProviderType
        self.config = ProviderConfig(provider=ProviderType.OPENMYTHOS)
        self._responses = responses
        self._idx = 0

    def is_available(self) -> bool:
        return True

    def complete(self, req: Any) -> Any:
        from open_mythos.skills.llm_providers import LLMResponse
        resp_text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return LLMResponse(
            text=resp_text,
            provider_used="mock",
            model="mock-model",
        )


class _MockRouter:
    """テスト用ルーター"""

    def __init__(self, responses: List[str]) -> None:
        self._provider = _MockProvider(responses)

    def available_providers(self) -> List[str]:
        return ["mock"]

    def complete(self, req: Any, preferred: Any = None) -> Any:
        return self._provider.complete(req)
