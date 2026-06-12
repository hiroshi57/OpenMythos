"""
Sprint 56 — マルチプロバイダー LLM 統合
Sprint 61 — Claude Fable 5 / Mythos 5 モデルティア + HuggingFace サイバー統合

Claude / OpenAI / OpenMythos / HuggingFace の 4 プロバイダーを共通インターフェースで扱う。

モデルティア (Sprint 61):
  ClaudeModelTier.HAIKU_5   → claude-haiku-4-5         高速 / 低コスト (現行デフォルト)
  ClaudeModelTier.FABLE_5   → claude-sonnet-4-5         バランス / 一般向け
  ClaudeModelTier.MYTHOS_5  → claude-opus-4             サイバー防衛 / 最高性能

HuggingFace 統合 (Sprint 61):
  HFInferenceProvider       : HuggingFace Inference API
    - Lily-Cybersecurity-7B-v0.2        セキュリティ Q&A / SOC アシスタント
    - Vulnerability_Detection_CodeBERT  コード脆弱性テキスト分類

オブジェクト (Sprint 56):
  ProviderType      : プロバイダー識別 enum (claude / openai / openmythos / hf_cyber)
  LLMRequest        : 共通リクエスト形式
  LLMResponse       : 共通レスポンス形式（プロバイダー名・使用トークンを含む）
  ProviderConfig    : プロバイダー設定（API キー・モデル名・タイムアウト）
  BaseLLMProvider   : 抽象基底クラス（.complete / .stream インターフェース）
  ClaudeProvider    : Anthropic (haiku / fable / mythos ティア対応)
  OpenAIProvider    : gpt-4o-mini
  OpenMythosProvider: ローカル OpenMythos モデル
  HFInferenceProvider: HuggingFace Inference API (サイバー特化)
  MultiProviderRouter: 優先順位・フォールバック付きルーター

使用例::
    router = MultiProviderRouter.from_env()
    resp = router.complete(LLMRequest(prompt="夏の広告コピーを1案生成してください"))
    print(resp.text)          # 生成テキスト
    print(resp.provider_used) # "claude" | "openai" | "openmythos" | "hf_cyber"

    # Claude Mythos 5 でサイバー分析
    from open_mythos.skills.llm_providers import ClaudeModelTier, ClaudeProvider, ProviderConfig, ProviderType
    provider = ClaudeProvider(ProviderConfig(
        provider=ProviderType.CLAUDE,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=ClaudeModelTier.MYTHOS_5,
    ))
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
    HF_CYBER    = "hf_cyber"    # Sprint 61: HuggingFace サイバー特化


class ClaudeModelTier(str, Enum):
    """Sprint 61 — Claude モデルティア定義。

    各ティアは実際の Anthropic API モデル ID にマッピングされる。
    API キーが利用できない場合は ClaudeProvider のフォールバックが動作する。

    Attributes:
        HAIKU_5:   高速・低コスト (既存デフォルト)
        FABLE_5:   バランス・一般向け消費者向けモデル (claude-sonnet-4-5)
        MYTHOS_5:  最高性能・サイバー防衛特化 (claude-opus-4)
    """
    HAIKU_5  = "claude-haiku-4-5"
    FABLE_5  = "claude-sonnet-4-5"
    MYTHOS_5 = "claude-opus-4"

    @property
    def tier_label(self) -> str:
        return {
            "claude-haiku-4-5":  "Haiku 5 (Fast)",
            "claude-sonnet-4-5": "Fable 5 (Balanced)",
            "claude-opus-4":     "Mythos 5 (Cyber Defense)",
        }.get(self.value, self.value)

    @property
    def context_window(self) -> int:
        """各モデルの公開コンテキストウィンドウサイズ（トークン）。"""
        return {
            "claude-haiku-4-5":  200_000,
            "claude-sonnet-4-5": 200_000,
            "claude-opus-4":     200_000,
        }.get(self.value, 100_000)

    @property
    def recommended_for(self) -> str:
        return {
            "claude-haiku-4-5":  "高速タスク・大量処理",
            "claude-sonnet-4-5": "汎用生成・コード・一般分析",
            "claude-opus-4":     "サイバー防衛・高度推論・複雑分析",
        }.get(self.value, "汎用")


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
        "claude":     ClaudeModelTier.HAIKU_5.value,
        "openai":     "gpt-4o-mini",
        "openmythos": "openmythos",
        "hf_cyber":   "segolilylabs/Lily-Cybersecurity-7B-v0.2",
    }, repr=False)

    def resolved_model(self) -> str:
        m = self.model
        # ClaudeModelTier enum を渡された場合は .value を使う
        if isinstance(m, ClaudeModelTier):
            return m.value
        return m or self._DEFAULT_MODELS.get(self.provider.value, "unknown")


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
# HFInferenceProvider  (Sprint 61)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HFInferenceProvider(BaseLLMProvider):
    """HuggingFace Inference API プロバイダー (Sprint 61 — サイバー特化)。

    対応モデル:
      segolilylabs/Lily-Cybersecurity-7B-v0.2  : Apache-2.0 | SOC / セキュリティ Q&A
      RayenLLM/Vulnerability_Detection_Using_CodeBERT : コード脆弱性テキスト分類

    HF_TOKEN 環境変数が設定されていれば認証付きリクエストを行う。
    未設定の場合は匿名アクセス（レート制限あり）にフォールバックする。
    """

    _API_BASE = "https://api-inference.huggingface.co/models"

    # 推奨サイバーモデル一覧 (HuggingFace 調査済み, Apache-2.0)
    CYBER_MODELS: Dict[str, str] = {
        "lily-cyber":   "segolilylabs/Lily-Cybersecurity-7B-v0.2",
        "codebert-vuln": "RayenLLM/Vulnerability_Detection_Using_CodeBERT",
        "titus-cyber":  "AlicanKiraz0/Titus-CybersecurityLLM-v1.0",
    }

    def is_available(self) -> bool:
        # API キー不要（匿名アクセス可）。常に利用可能とする。
        return True

    def complete(self, req: LLMRequest) -> LLMResponse:
        model = self.config.resolved_model()
        t0    = time.perf_counter()

        payload = json.dumps({"inputs": req.prompt}).encode("utf-8")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        url     = f"{self._API_BASE}/{model}"
        http_req = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(http_req, timeout=self.config.timeout) as res:
                data = json.loads(res.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            # モデルロード待ち (503) の場合はフォールバックメッセージを返す
            if e.code == 503:
                latency_ms = (time.perf_counter() - t0) * 1000
                return self._build_response(
                    text=f"[HF model loading: {model}] retry after 20s",
                    model=model, latency_ms=latency_ms,
                )
            raise RuntimeError(f"HuggingFace Inference API HTTP {e.code}: {body}")
        except Exception as e:
            raise RuntimeError(f"HuggingFace Inference API error: {e}")

        # レスポンス形式はモデルにより異なる
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text = first.get("generated_text", first.get("label", str(first)))
            else:
                text = str(first)
        elif isinstance(data, dict):
            text = data.get("generated_text", data.get("label", str(data)))
        else:
            text = str(data)

        latency_ms = (time.perf_counter() - t0) * 1000
        return self._build_response(text=text, model=model, latency_ms=latency_ms)


def list_model_tiers() -> List[Dict[str, str]]:
    """Sprint 61 — 利用可能なモデルティア一覧を返す。

    Returns:
        各ティアの id / label / context_window / recommended_for を含む辞書リスト
    """
    return [
        {
            "id":             tier.value,
            "label":          tier.tier_label,
            "context_window": str(tier.context_window),
            "recommended_for": tier.recommended_for,
            "provider":       "anthropic",
        }
        for tier in ClaudeModelTier
    ] + [
        {
            "id":             model_id,
            "label":          alias,
            "context_window": "4096",
            "recommended_for": "サイバーセキュリティ特化 Q&A / 脆弱性分類",
            "provider":       "huggingface",
        }
        for alias, model_id in HFInferenceProvider.CYBER_MODELS.items()
    ]


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
            elif pt == ProviderType.HF_CYBER:
                # Sprint 61: HuggingFace Inference API (HF_TOKEN は任意)
                key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                providers.append(HFInferenceProvider(ProviderConfig(provider=pt, api_key=key)))
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
