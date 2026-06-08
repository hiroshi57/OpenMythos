"""
Sprint 44 — Instructor: 構造化出力抽出

Hermes Skill: instructor
ref: skills/mlops/instructor-SKILL.md

Pydantic スキーマ定義から LLM レスポンスの JSON を検証・変換する。
`instructor` ライブラリがある場合はそれを使用し、
ない場合は正規表現ベースの JSON 抽出フォールバックを使用する。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# データモデル
# ---------------------------------------------------------------------------

@dataclass
class ExtractionSchema:
    """抽出スキーマ定義。"""
    name: str
    fields: Dict[str, str]          # field_name → type_hint (str)
    description: str = ""
    required: List[str] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        for fname, ftype in self.fields.items():
            props[fname] = {"type": _py_to_json_type(ftype), "description": fname}
        return {
            "type": "object",
            "title": self.name,
            "description": self.description,
            "properties": props,
            "required": self.required or list(self.fields.keys()),
        }


@dataclass
class ExtractionResult:
    """抽出結果。"""
    data: Dict[str, Any]
    schema_name: str
    raw_text: str = ""
    success: bool = True
    error: str = ""
    retries: int = 0


def _py_to_json_type(py_type: str) -> str:
    mapping = {
        "str": "string", "int": "integer", "float": "number",
        "bool": "boolean", "list": "array", "dict": "object",
    }
    return mapping.get(py_type.lower(), "string")


# ---------------------------------------------------------------------------
# JSON 抽出ユーティリティ
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """テキストから最初の JSON オブジェクトを抽出する。"""
    # コードブロック内の JSON
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 裸の JSON
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _validate_against_schema(data: Dict[str, Any], schema: ExtractionSchema) -> Dict[str, Any]:
    """スキーマの required フィールドが揃っているか確認し、型変換を試みる。"""
    result: Dict[str, Any] = {}
    type_casts = {"str": str, "int": int, "float": float, "bool": bool}
    for fname, ftype in schema.fields.items():
        if fname in data:
            cast = type_casts.get(ftype.lower())
            try:
                result[fname] = cast(data[fname]) if cast else data[fname]
            except (ValueError, TypeError):
                result[fname] = data[fname]
        elif fname in schema.required:
            result[fname] = None
    return result


# ---------------------------------------------------------------------------
# InstructorExtractor
# ---------------------------------------------------------------------------

class InstructorExtractor:
    """LLM レスポンスから Pydantic 互換の構造化データを抽出するクラス。

    Parameters
    ----------
    max_retries:
        抽出失敗時の最大リトライ回数。
    strict:
        True の場合、required フィールドが欠落していると例外を送出する。
    """

    def __init__(self, max_retries: int = 3, strict: bool = False) -> None:
        self.max_retries = max_retries
        self.strict = strict
        # instructor ライブラリの有無をチェック
        try:
            import instructor  # type: ignore
            self._instructor = instructor
            self._native = True
        except ImportError:
            self._instructor = None
            self._native = False

    # ---- 主要メソッド ----

    def extract(
        self,
        text: str,
        schema: ExtractionSchema,
    ) -> ExtractionResult:
        """テキストからスキーマに従ってデータを抽出する。"""
        retries = 0
        last_error = ""
        while retries <= self.max_retries:
            parsed = _extract_json(text)
            if parsed is not None:
                validated = _validate_against_schema(parsed, schema)
                missing = [f for f in schema.required if validated.get(f) is None]
                if missing and self.strict:
                    last_error = f"Missing required fields: {missing}"
                    retries += 1
                    continue
                return ExtractionResult(
                    data=validated,
                    schema_name=schema.name,
                    raw_text=text,
                    success=True,
                    retries=retries,
                )
            last_error = "No JSON found in text"
            retries += 1

        return ExtractionResult(
            data={},
            schema_name=schema.name,
            raw_text=text,
            success=False,
            error=last_error,
            retries=retries,
        )

    def extract_batch(
        self,
        texts: List[str],
        schema: ExtractionSchema,
    ) -> List[ExtractionResult]:
        """複数テキストを一括抽出する。"""
        return [self.extract(t, schema) for t in texts]

    def build_prompt(self, schema: ExtractionSchema, instruction: str = "") -> str:
        """スキーマに従った JSON を返すよう LLM に指示するプロンプトを生成する。"""
        js = json.dumps(schema.to_json_schema(), ensure_ascii=False, indent=2)
        base = instruction or "以下のテキストから情報を抽出し、指定の JSON スキーマに従って出力してください。"
        return f"{base}\n\nJSON Schema:\n```json\n{js}\n```\n\n必ず有効な JSON オブジェクトで回答してください。"

    @property
    def is_native(self) -> bool:
        """instructor ライブラリが利用可能かどうか。"""
        return self._native
