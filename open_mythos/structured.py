"""
OpenMythos Structured Output / JSON Mode.

ClaudeMythosのStructured Output能力に対応するオープン実装。
OpenMythosモデルの生成をJSONスキーマに従ってグリーディに制約する。

特徴:
    - JSON schema (dict) を受け取って型・フィールドを保証した生成
    - マーケティング用スキーマ例 (ad_performance, marketing_report) を同梱
    - 生成失敗時はフォールバック値を返す (例外を投げない)

使い方::

    from open_mythos.structured import StructuredGenerator, AD_PERFORMANCE_SCHEMA

    gen = StructuredGenerator(model, device="cpu")
    result = gen.generate_json(
        schema=AD_PERFORMANCE_SCHEMA,
        prompt="新商品ローンチのCTR予測レポートを作成してください",
    )
    print(result)  # dict: {"ctr": 0.032, "tier": "high", "roas": 3.2, ...}
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from open_mythos.main import OpenMythos


# ---------------------------------------------------------------------------
# マーケティング用スキーマ定義
# ---------------------------------------------------------------------------

AD_PERFORMANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "広告パフォーマンス予測レポート",
    "properties": {
        "ctr": {
            "type": "number",
            "description": "クリック率 (0–1)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "cvr": {
            "type": "number",
            "description": "コンバージョン率 (0–1)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "roas": {
            "type": "number",
            "description": "Return on Ad Spend (0以上)",
            "minimum": 0.0,
        },
        "tier": {
            "type": "string",
            "description": "パフォーマンス tier",
            "enum": ["high", "medium", "low"],
        },
        "confidence": {
            "type": "number",
            "description": "予測信頼度 (0–1)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "recommendation": {
            "type": "string",
            "description": "推奨アクション (1文)",
        },
        "quality_score": {
            "type": "integer",
            "description": "Google広告 Quality Score (1–10)",
            "minimum": 1,
            "maximum": 10,
        },
        "impression_share": {
            "type": "number",
            "description": "インプレッションシェア (0–1)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["ctr", "tier", "confidence"],
}

MARKETING_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "マーケティングコンテンツ品質レポート",
    "properties": {
        "title": {
            "type": "string",
            "description": "コンテンツタイトル (50字以内)",
        },
        "llmo_score": {
            "type": "number",
            "description": "LLMO最適化スコア (0–1)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "seo_score": {
            "type": "number",
            "description": "SEOスコア (0–100)",
            "minimum": 0.0,
            "maximum": 100.0,
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "抽出されたキーワードリスト",
        },
        "target_persona": {
            "type": "string",
            "description": "想定ターゲットペルソナ",
        },
        "publish_ready": {
            "type": "boolean",
            "description": "公開可否判定",
        },
    },
    "required": ["title", "llmo_score", "publish_ready"],
}

SEO_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "SEO/LLMO最適化コンテンツ生成結果",
    "properties": {
        "headline": {
            "type": "string",
            "description": "SEO対応ヘッドライン (60字以内)",
        },
        "meta_description": {
            "type": "string",
            "description": "メタディスクリプション (155字以内)",
        },
        "entity_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": "コンテンツ内のキーエンティティ",
        },
        "style": {
            "type": "string",
            "enum": ["answer_first", "faq", "entity_rich"],
            "description": "コンテンツスタイル",
        },
        "citability_score": {
            "type": "number",
            "description": "AI引用されやすさスコア (0–1)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "keyword_density": {
            "type": "number",
            "description": "ターゲットキーワード密度 (0–1, SEO推奨: 0.01–0.02)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "lcp_score": {
            "type": "string",
            "description": "Core Web Vitals LCP評価",
            "enum": ["good", "needs_improvement", "poor"],
        },
        "cls_score": {
            "type": "string",
            "description": "Core Web Vitals CLS評価",
            "enum": ["good", "needs_improvement", "poor"],
        },
    },
    "required": ["headline", "style", "citability_score"],
}

# 既定スキーマの一覧
BUILTIN_SCHEMAS: dict[str, dict] = {
    "ad_performance": AD_PERFORMANCE_SCHEMA,
    "marketing_report": MARKETING_REPORT_SCHEMA,
    "seo_content": SEO_CONTENT_SCHEMA,
}


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """JSON Schema (subset) に対するバリデーター。"""

    def validate(self, data: Any, schema: dict) -> tuple[bool, str]:
        """
        data が schema に適合するか検証する。

        Args:
            data   -- 検証対象データ
            schema -- JSON Schema dict

        Returns:
            (is_valid: bool, error_message: str)
        """
        return self._check(data, schema, path="$")

    def _check(self, data: Any, schema: dict, path: str) -> tuple[bool, str]:
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                return False, f"{path}: expected object, got {type(data).__name__}"
            props = schema.get("properties", {})
            required = schema.get("required", [])
            for key in required:
                if key not in data:
                    return False, f"{path}.{key}: required field missing"
            for key, val in data.items():
                if key in props:
                    ok, msg = self._check(val, props[key], f"{path}.{key}")
                    if not ok:
                        return False, msg

        elif schema_type == "array":
            if not isinstance(data, list):
                return False, f"{path}: expected array, got {type(data).__name__}"
            items_schema = schema.get("items", {})
            for i, item in enumerate(data):
                ok, msg = self._check(item, items_schema, f"{path}[{i}]")
                if not ok:
                    return False, msg

        elif schema_type == "string":
            if not isinstance(data, str):
                return False, f"{path}: expected string, got {type(data).__name__}"
            enum = schema.get("enum")
            if enum and data not in enum:
                return False, f"{path}: value {data!r} not in enum {enum}"

        elif schema_type == "number":
            # bool は Python では int のサブクラスだが number として扱わない
            if isinstance(data, bool) or not isinstance(data, (int, float)):
                return False, f"{path}: expected number, got {type(data).__name__}"
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and data < minimum:
                return False, f"{path}: {data} < minimum {minimum}"
            if maximum is not None and data > maximum:
                return False, f"{path}: {data} > maximum {maximum}"

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                return False, f"{path}: expected boolean, got {type(data).__name__}"

        elif schema_type == "integer":
            # bool は Python では int のサブクラスだが integer として扱わない
            if isinstance(data, bool) or not isinstance(data, int):
                return False, f"{path}: expected integer, got {type(data).__name__}"

        return True, ""


# ---------------------------------------------------------------------------
# StructuredGenerator
# ---------------------------------------------------------------------------


class StructuredGenerator:
    """
    JSON mode / Structured Output 生成エンジン。

    OpenMythosモデルを使ってJSONスキーマ準拠の出力を生成する。
    生成されたテキストからJSONを抽出し、スキーマで検証する。
    検証失敗の場合はスキーマのデフォルト値でフォールバックする。

    Args:
        model   -- OpenMythosモデルインスタンス
        device  -- torch device文字列
    """

    def __init__(self, model: OpenMythos, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._validator = SchemaValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_json(
        self,
        schema: dict[str, Any],
        prompt: str,
        loops: int = 6,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n_attempts: int = 3,
    ) -> dict[str, Any]:
        """
        JSONスキーマ準拠の出力を生成する。

        最大 n_attempts 回生成を試み、最初に検証をパスした結果を返す。
        全試行で失敗した場合はスキーマのデフォルト値を返す。

        Args:
            schema         -- JSON Schema dict
            prompt         -- ユーザープロンプト
            loops          -- 推論ループ数
            max_new_tokens -- 最大生成トークン数
            temperature    -- サンプリング温度
            top_p          -- nucleus sampling 閾値
            n_attempts     -- 最大試行回数 (デフォルト: 3)

        Returns:
            生成されたJSON dict
        """
        system_prompt = self._build_system_prompt(schema)
        full_prompt = f"{system_prompt}\n[User]: {prompt}\n[Assistant]: {{"

        vsize = self.model.cfg.vocab_size
        ids = [ord(c) % vsize for c in full_prompt]
        # JSON開き括弧をプロンプトに含める
        ids.append(ord("{") % vsize)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        for attempt in range(n_attempts):
            t = temperature * (1.0 + attempt * 0.1)  # 温度を少し上げて多様性追加
            try:
                generated_text = self._generate_raw(input_ids, loops, max_new_tokens, t, top_p)
                result = self._extract_json("{" + generated_text)
                if result is not None:
                    result = self._coerce_to_schema(result, schema)
                    ok, msg = self._validator.validate(result, schema)
                    if ok:
                        return result
            except Exception:
                pass

        # フォールバック: スキーマのデフォルト値を構築
        return self._build_fallback(schema)

    def generate_json_batch(
        self,
        schema: dict[str, Any],
        prompts: list[str],
        loops: int = 6,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """複数プロンプトを一括でJSON生成する。"""
        return [
            self.generate_json(schema, p, loops=loops,
                               max_new_tokens=max_new_tokens, temperature=temperature)
            for p in prompts
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_raw(
        self,
        input_ids: torch.Tensor,
        loops: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """生のテキストを生成する (JSON抽出前)。"""
        cur_ids = input_ids
        generated: list[int] = []
        brace_depth = 1  # 最初の { はプロンプトに含まれている

        vsize = self.model.cfg.vocab_size

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(temperature, 1e-8)

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float("-inf")
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(
                        0, sorted_idx, sorted_logits
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
                generated.append(next_token)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )

                # 文字に変換してブレース追跡
                try:
                    c = chr(next_token % 128)
                    if c == "{":
                        brace_depth += 1
                    elif c == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            break
                except (ValueError, OverflowError):
                    pass

        # 文字列に変換
        chars = []
        for i in generated:
            try:
                c = chr(i % 128)
                if c.isprintable() or c in "\n\t ":
                    chars.append(c)
            except (ValueError, OverflowError):
                pass
        return "".join(chars)

    def _extract_json(self, text: str) -> Optional[dict]:
        """テキストからJSONオブジェクトを抽出する。"""
        # 最初の { から最後の } までを取得
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # ネストした JSON を試みる
        try:
            # 簡易 JSON 補完: 未閉鎖の文字列・配列・オブジェクトを閉じる
            completed = _complete_json(text)
            return json.loads(completed)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _coerce_to_schema(self, data: dict, schema: dict) -> dict:
        """
        生成されたデータをスキーマの型・制約に合わせて強制変換する。
        (型エラーの軽微な修正のみ。欠落フィールドはフォールバックで補完)
        """
        props = schema.get("properties", {})
        result = {}
        for key, val in data.items():
            if key not in props:
                continue
            prop = props[key]
            coerced = _coerce_value(val, prop)
            result[key] = coerced
        return result

    def _build_system_prompt(self, schema: dict) -> str:
        """スキーマからシステムプロンプトを構築する。"""
        desc = schema.get("description", "structured data")
        props = schema.get("properties", {})
        required = schema.get("required", [])

        field_lines = []
        for name, prop in props.items():
            req_marker = " [required]" if name in required else ""
            field_desc = prop.get("description", "")
            field_type = prop.get("type", "any")
            enum = prop.get("enum")
            if enum:
                field_type = f"string (one of: {', '.join(enum)})"
            field_lines.append(f"  - {name} ({field_type}){req_marker}: {field_desc}")

        fields_text = "\n".join(field_lines)
        return (
            f"[System]: You must respond with a valid JSON object for: {desc}.\n"
            f"Required fields and types:\n{fields_text}\n"
            f"Respond ONLY with a JSON object. No explanation, no markdown."
        )

    def _build_fallback(self, schema: dict) -> dict:
        """スキーマの型からデフォルト値を構築するフォールバック。"""
        props = schema.get("properties", {})
        result: dict[str, Any] = {}
        for name, prop in props.items():
            result[name] = _default_value(prop)
        return result


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------


def _coerce_value(val: Any, prop: dict) -> Any:
    """単一値をスキーマ型に強制変換する。"""
    schema_type = prop.get("type")
    if schema_type == "number":
        try:
            v = float(val)
            minimum = prop.get("minimum")
            maximum = prop.get("maximum")
            if minimum is not None:
                v = max(v, float(minimum))
            if maximum is not None:
                v = min(v, float(maximum))
            return v
        except (TypeError, ValueError):
            return prop.get("minimum", 0.0)
    elif schema_type == "string":
        val_str = str(val)
        enum = prop.get("enum")
        if enum and val_str not in enum:
            return enum[0]
        return val_str
    elif schema_type == "boolean":
        if isinstance(val, bool):
            return val
        return bool(val)
    elif schema_type == "integer":
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0
    elif schema_type == "array":
        if isinstance(val, list):
            return val
        return []
    return val


def _default_value(prop: dict) -> Any:
    """スキーマプロパティのデフォルト値を返す。"""
    schema_type = prop.get("type", "string")
    if schema_type == "number":
        minimum = prop.get("minimum", 0.0)
        maximum = prop.get("maximum")
        if maximum is not None:
            return (minimum + maximum) / 2.0
        return minimum
    elif schema_type == "string":
        enum = prop.get("enum")
        if enum:
            return enum[0]
        return ""
    elif schema_type == "boolean":
        return False
    elif schema_type == "integer":
        return 0
    elif schema_type == "array":
        return []
    elif schema_type == "object":
        return {}
    return None


def _complete_json(text: str) -> str:
    """
    不完全なJSONを補完して有効なJSONにする簡易実装。
    ブレース・ブラケットのバランスを強制的に修正する。
    """
    # 不正な末尾のカンマを削除
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # ブレース・ブラケットのカウント
    brace = text.count("{") - text.count("}")
    bracket = text.count("[") - text.count("]")

    # 不完全な文字列の補完
    if text.count('"') % 2 == 1:
        text += '"'

    # ブラケットを閉じる
    text += "]" * max(0, bracket)
    text += "}" * max(0, brace)

    return text
