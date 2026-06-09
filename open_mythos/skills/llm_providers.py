"""
Sprint 56 — マルチプロバイダー LLM 統合

Claude / OpenAI / OpenMythos の 3 プロバイダーを共通インターフェースで扱う。

オブジェクト:
  ProviderType      : プロバイダー識別 enum (claude / openai / openmythos)
  LLMRequest        : 共通リクエスト形式
  LLMResponse       : 共通レスポンス形式（プロバイダー名・使用トークンを含む）
  ProviderConfig    : プロバイダー設定（API キー・モデル名・タイムアウト）
  BaseLLMProvider   : 抽象基底クラス（.complete / .stream インターフェース）
  ClaudeProvider    : Anthropic claude-haiku-4-5
  OpenAIProvider    : gpt-4o-mini
  OpenMythosProvider: ローカル OpenMythos モデル
  MultiProviderRouter: 優先順位・フォールバック付きルーター

使用例::
    router = MultiProviderRouter.from_env()
    resp = router.complete(LLMRequest(prompt="夏の広告コピーを1案生成してください"))
    print(resp.text)          # 生成テキスト
    print(resp.provider_used) # "claude" | "openai" | "openmythos"
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 型定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProviderType(str, Enum):
    CLAUDE      = "claude"
    OPENAI      = "openai"
    OPENMYTHOS  = "openmythos"


@dataclass
class LLMRequest:
    """プロバイダー共通リクエスト"""
    prompt:      str
    system:      Optional[str]  = None
    max_tokens:  int            = 256
    temperature: float          = 0.7
    # 追加メッセージ (OpenAI chat 形式)
    messages:    List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LLMResponse:
    """プロバイダー共通レスポンス"""
    text:          str
    provider_used: str                    # "claude" | "openai" | "openmythos"
    model:         str
    latency_ms:    float                  = 0.0
    prompt_tokens: int                    = 0
    completion_tokens: int                = 0
    total_tokens:  int                    = 0
    metadata:      Dict[str, Any]         = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.text)


@dataclass
class ProviderConfig:
    """プロバイダー設定"""
    provider:    ProviderType
    api_key:     Optional[str] = None
    model:       Optional[str] = None     # None → プロバイダーのデフォルト
    timeout:     int           = 30       # 秒
    max_retries: int           = 2

    # デフォルトモデル
    _DEFAULT_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "claude":     "claude-haiku-4-5",
        "openai":     "gpt-4o-mini",
        "openmythos": "openmythos",
    }, repr=False)

    def resolved_model(self) -> str:
        return self.model or self._DEFAULT_MODELS.get(self.provider.value, "unknown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BaseLLMProvider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseLLMProvider:
    """全プロバイダーの抽象基底クラス"""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def provider_type(self) -> ProviderType:
        return self.config.provider

    def is_available(self) -> bool:
        """API キーが設定されているかどうか"""
        return bool(self.config.api_key)

    def complete(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream(self, req: LLMRequest) -> Iterator[str]:
        """デフォルト実装: complete() を呼んで文字ごとに yield"""
        resp = self.complete(req)
        for char in resp.text:
            yield char

    def _build_response(
        self,
        text:              str,
        model:             str,
        latency_ms:        float,
        prompt_tokens:     int = 0,
        completion_tokens: int = 0,
    ) -> LLMResponse:
        return LLMResponse(
            text=text,
            provider_used=self.config.provider.value,
            model=model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ClaudeProvider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude API プロバイダー"""

    _API_URL = "https://api.anthropic.com/v1/messages"

    def complete(self, req: LLMRequest) -> LLMResponse:
        if not self.config.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY が設定されていません")

        model     = self.config.resolved_model()
        t0        = time.perf_counter()
        messages  = req.messages or [{"role": "user", "content": req.prompt}]

        payload = json.dumps({
            "model":      model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages":   messages,
            **({"system": req.system} if req.system else {}),
        }).encode("utf-8")

        http_req = urllib.request.Request(
            self._API_URL,
            data=payload,
            headers={
                "Content-Type":    "application/json",
                "x-api-key":       self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.config.timeout) as res:
                data = json.loads(res.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Claude API HTTP {e.code}: {e.read().decode()[:200]}")

        text = data["content"][0]["text"].strip()
        usage = data.get("usage", {})
        latency_ms = (time.perf_counter() - t0) * 1000

        return self._build_response(
            text=text,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAIProvider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API プロバイダー"""

    _API_URL = "https://api.openai.com/v1/chat/completions"

    def complete(self, req: LLMRequest) -> LLMResponse:
        if not self.config.api_key:
            raise RuntimeError("OPENAI_API_KEY が設定されていません")

        model    = self.config.resolved_model()
        t0       = time.perf_counter()
        messages = req.messages or []

        if req.system:
            messages = [{"role": "system", "content": req.system}] + messages
        if req.prompt and not req.messages:
            messages.append({"role": "user", "content": req.prompt})

        payload = json.dumps({
            "model":       model,
            "max_tokens":  req.max_tokens,
            "temperature": req.temperature,
            "messages":    messages,
        }).encode("utf-8")

        http_req = urllib.request.Request(
            self._API_URL,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.config.timeout) as res:
                data = json.loads(res.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI API HTTP {e.code}: {e.read().decode()[:200]}")

        text  = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        latency_ms = (time.perf_counter() - t0) * 1000

        return self._build_response(
            text=text,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenMythosProvider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OpenMythosProvider(BaseLLMProvider):
    """ローカル OpenMythos モデルプロバイダー"""

    def __init__(self, config: ProviderConfig, llm: Any = None) -> None:
        super().__init__(config)
        self._llm = llm   # OpenMythosLLM インスタンス（任意）

    def is_available(self) -> bool:
        return self._llm is not None

    def complete(self, req: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()

        if self._llm is not None:
            try:
                full_prompt = f"{req.system}\n\n{req.prompt}" if req.system else req.prompt
                result = self._llm.run(full_prompt)
                # MagicMock 等の非文字列を安全に変換
                text = result if isinstance(result, str) else str(result)
            except Exception as e:
                raise RuntimeError(f"OpenMythos モデルエラー: {e}")
        else:
            # モデル未ロード: プロンプトを折り返してエコー (フォールバック)
            text = f"[OpenMythos: モデル未ロード] {req.prompt[:80]}"

        latency_ms = (time.perf_counter() - t0) * 1000
        return self._build_response(
            text=text,
            model=self.config.resolved_model(),
            latency_ms=latency_ms,
            completion_tokens=len(text.split()),
        )

    def stream(self, req: LLMRequest) -> Iterator[str]:
        if self._llm is not None:
            try:
                full = f"{req.system}\n\n{req.prompt}" if req.system else req.prompt
                yield from self._llm.stream(full)
                return
            except Exception:
                pass
        # フォールバック: word-by-word
        resp = self.complete(req)
        for word in resp.text.split():
            yield word + " "


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MultiProviderRouter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiProviderRouter:
    """
    複数プロバイダーを優先順位付きで試行するルーター。

    優先順位 (デフォルト): claude → openai → openmythos
    API キーが未設定のプロバイダーはスキップされる。
    全プロバイダーが利用不能な場合は最後のエラーを再 raise する。
    """

    def __init__(self, providers: List[BaseLLMProvider]) -> None:
        self._providers = providers

    @classmethod
    def from_env(
        cls,
        llm: Any = None,
        priority: Optional[List[ProviderType]] = None,
    ) -> "MultiProviderRouter":
        """
        環境変数から API キーを読み込んでルーターを構築する。

        priority (デフォルト): [CLAUDE, OPENAI, OPENMYTHOS]
        """
        if priority is None:
            priority = [ProviderType.CLAUDE, ProviderType.OPENAI, ProviderType.OPENMYTHOS]

        providers: List[BaseLLMProvider] = []
        for pt in priority:
            if pt == ProviderType.CLAUDE:
                key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
                providers.append(ClaudeProvider(ProviderConfig(provider=pt, api_key=key)))
            elif pt == ProviderType.OPENAI:
                key = os.getenv("OPENAI_API_KEY")
                providers.append(OpenAIProvider(ProviderConfig(provider=pt, api_key=key)))
            elif pt == ProviderType.OPENMYTHOS:
                providers.append(
                    OpenMythosProvider(ProviderConfig(provider=pt), llm=llm)
                )
        return cls(providers)

    def available_providers(self) -> List[str]:
        return [p.config.provider.value for p in self._providers if p.is_available()]

    def complete(
        self,
        req: LLMRequest,
        preferred: Optional[ProviderType] = None,
    ) -> LLMResponse:
        """
        優先プロバイダーから順に試行し、最初に成功したレスポンスを返す。
        preferred を指定するとそのプロバイダーを最優先にする。
        """
        ordered = list(self._providers)
        if preferred is not None:
            ordered = sorted(ordered, key=lambda p: (p.config.provider != preferred))

        last_err: Optional[Exception] = None
        for provider in ordered:
            if not provider.is_available():
                continue
            try:
                return provider.complete(req)
            except Exception as e:
                last_err = e
                continue

        if last_err:
            raise RuntimeError(
                f"全プロバイダーが失敗: {last_err} "
                f"(available: {self.available_providers()})"
            )
        raise RuntimeError("利用可能なプロバイダーがありません")

    def stream(
        self,
        req: LLMRequest,
        preferred: Optional[ProviderType] = None,
    ) -> Iterator[str]:
        """優先プロバイダーから順に試行してストリーミングを返す。"""
        ordered = list(self._providers)
        if preferred is not None:
            ordered = sorted(ordered, key=lambda p: (p.config.provider != preferred))

        for provider in ordered:
            if not provider.is_available():
                continue
            try:
                yield from provider.stream(req)
                return
            except Exception:
                continue

        raise RuntimeError("利用可能なプロバイダーがありません")
