"""
OpenMythos Tool Use / Function Calling エンジン。

Claude の Tool Use 機能に対応するオープン実装。
OpenAI 互換の tools/tool_choice API をサポートし、
モデルが自律的にツールを呼び出してタスクを解決できる。

設計:
    ToolDefinition  -- ツール仕様 (name/description/parameters)
    ToolCall        -- モデルが要求したツール呼び出し
    ToolResult      -- ツール実行結果
    ToolRegistry    -- ツールの登録・検索・実行
    @tool デコレータ -- 関数を自動登録

使い方::

    from open_mythos.tools import tool, ToolRegistry, execute_tool_call

    @tool(description="競合他社の広告費を検索する")
    def search_competitor(company: str, metric: str = "ad_spend") -> dict:
        return {"company": company, "ad_spend_usd": 1500000, "metric": metric}

    registry = ToolRegistry.default()
    result = execute_tool_call(
        tool_call=ToolCall(name="search_competitor", arguments={"company": "Jasper AI"}),
        registry=registry,
    )
    print(result.content)  # {"company": "Jasper AI", "ad_spend_usd": ...}
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class ParameterSchema:
    """ツールパラメータの JSON Schema 表現。"""

    type: str
    description: str = ""
    enum: list[str] = field(default_factory=list)
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """
    ツール仕様。OpenAI の function definition と互換。

    Args:
        name        -- ツール名 (snake_case 推奨)
        description -- モデルへの説明文。何をするツールかを明確に
        parameters  -- {param_name: ParameterSchema} の辞書
        fn          -- 実際の実行関数
    """

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    fn: Callable

    def to_openai_schema(self) -> dict:
        """OpenAI tools[] 形式の JSON スキーマを返す。"""
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, pschema in self.parameters.items():
            prop: dict[str, Any] = {
                "type": pschema.type,
                "description": pschema.description,
            }
            if pschema.enum:
                prop["enum"] = pschema.enum
            props[pname] = prop
            if pschema.required:
                required.append(pname)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


@dataclass
class ToolCall:
    """モデルが生成したツール呼び出し要求。"""

    name: str
    """呼び出すツール名。"""

    arguments: dict[str, Any] = field(default_factory=dict)
    """引数の辞書。"""

    call_id: str = ""
    """呼び出しID (OpenAI 互換)。"""

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        """辞書から ToolCall を構築する。"""
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return cls(
            name=data.get("name", ""),
            arguments=args,
            call_id=data.get("id", data.get("call_id", "")),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class ToolResult:
    """ツール実行結果。"""

    tool_name: str
    content: Any
    """実行結果 (dict / str / list など)。"""

    call_id: str = ""
    error: str = ""
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return not self.error

    def to_message(self) -> dict:
        """OpenAI tool message 形式に変換する。"""
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.tool_name,
            "content": (
                json.dumps(self.content, ensure_ascii=False)
                if not isinstance(self.content, str)
                else self.content
            ),
        }


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    ツールの登録・検索・実行レジストリ。

    シングルトンの global registry と、
    独立したインスタンスを使う local registry の両方をサポート。
    """

    _global: "ToolRegistry | None" = None

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # 登録
    # ------------------------------------------------------------------

    def register(self, tool_def: ToolDefinition) -> None:
        """ツールを登録する。同名があれば上書き。"""
        self._tools[tool_def.name] = tool_def

    def register_fn(
        self,
        fn: Callable,
        description: str = "",
        parameters: Optional[dict[str, ParameterSchema]] = None,
    ) -> ToolDefinition:
        """
        関数を ToolDefinition に変換して登録する。

        parameters が None の場合、関数シグネチャから自動推論する。
        """
        name = fn.__name__
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        params = parameters or _infer_parameters(fn)
        tool_def = ToolDefinition(name=name, description=desc, parameters=params, fn=fn)
        self.register(tool_def)
        return tool_def

    # ------------------------------------------------------------------
    # 検索
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict]:
        """OpenAI tools[] 形式のリストを返す。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------

    def call(self, tool_call: ToolCall) -> ToolResult:
        """ToolCall を実行して ToolResult を返す。"""
        return execute_tool_call(tool_call, self)

    # ------------------------------------------------------------------
    # Global / default registry
    # ------------------------------------------------------------------

    @classmethod
    def global_registry(cls) -> "ToolRegistry":
        """グローバルシングルトンレジストリを返す。"""
        if cls._global is None:
            cls._global = cls()
        return cls._global

    @classmethod
    def default(cls) -> "ToolRegistry":
        """マーケ特化ツールを登録済みのデフォルトレジストリを返す。"""
        from open_mythos.tools_marketing import register_marketing_tools

        registry = cls()
        register_marketing_tools(registry)
        return registry


# ---------------------------------------------------------------------------
# @tool デコレータ
# ---------------------------------------------------------------------------


def tool(
    description: str = "",
    parameters: Optional[dict[str, ParameterSchema]] = None,
    registry: Optional[ToolRegistry] = None,
) -> Callable:
    """
    関数を OpenMythos ツールとして登録するデコレータ。

    Args:
        description -- ツールの説明文
        parameters  -- パラメータスキーマ (None なら関数シグネチャから推論)
        registry    -- 登録先レジストリ (None なら global registry)

    使い方::

        @tool(description="競合の広告費を検索")
        def search_competitor(company: str, metric: str = "ad_spend") -> dict:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        reg = registry or ToolRegistry.global_registry()
        reg.register_fn(fn, description=description, parameters=parameters)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# 実行エンジン
# ---------------------------------------------------------------------------


def execute_tool_call(
    tool_call: ToolCall,
    registry: ToolRegistry,
) -> ToolResult:
    """
    ToolCall を実行し ToolResult を返す。

    Args:
        tool_call -- 実行するツール呼び出し
        registry  -- ツールを探すレジストリ

    Returns:
        ToolResult (エラー時も例外を投げず error フィールドに入れる)
    """
    t0 = time.perf_counter()
    tool_def = registry.get(tool_call.name)

    if tool_def is None:
        return ToolResult(
            tool_name=tool_call.name,
            content=None,
            call_id=tool_call.call_id,
            error=f"Tool '{tool_call.name}' not found in registry. "
            f"Available: {registry.names()}",
            latency_ms=0.0,
        )

    try:
        result = tool_def.fn(**tool_call.arguments)
        latency_ms = (time.perf_counter() - t0) * 1000
        return ToolResult(
            tool_name=tool_call.name,
            content=result,
            call_id=tool_call.call_id,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return ToolResult(
            tool_name=tool_call.name,
            content=None,
            call_id=tool_call.call_id,
            error=f"{type(e).__name__}: {e}",
            latency_ms=round(latency_ms, 2),
        )


def execute_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
) -> list[ToolResult]:
    """複数の ToolCall を順次実行する。"""
    return [execute_tool_call(tc, registry) for tc in tool_calls]


# ---------------------------------------------------------------------------
# パラメータ自動推論
# ---------------------------------------------------------------------------

_PYTHON_TO_JSON_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
}


def _infer_parameters(fn: Callable) -> dict[str, ParameterSchema]:
    """
    関数シグネチャから ParameterSchema を自動推論する。

    型アノテーションと docstring から description を取得する。
    """
    sig = inspect.signature(fn)
    hints = fn.__annotations__ if hasattr(fn, "__annotations__") else {}
    doc = fn.__doc__ or ""

    # docstring からパラメータ説明を抽出 (Args: セクション)
    param_docs: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                in_args = False
                continue
            if "--" in stripped:
                pname, pdesc = stripped.split("--", 1)
                param_docs[pname.strip()] = pdesc.strip()
            elif ":" in stripped and not stripped.startswith("#"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    param_docs[parts[0].strip()] = parts[1].strip()

    params: dict[str, ParameterSchema] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue

        annotation = hints.get(pname, None)
        type_name = "string"
        if annotation is not None:
            raw = getattr(annotation, "__name__", str(annotation))
            type_name = _PYTHON_TO_JSON_TYPE.get(raw, "string")

        has_default = param.default is not inspect.Parameter.empty
        default_val = param.default if has_default else None

        params[pname] = ParameterSchema(
            type=type_name,
            description=param_docs.get(pname, ""),
            required=not has_default,
            default=default_val,
        )

    return params


# ---------------------------------------------------------------------------
# ToolCall パーサー (モデル出力テキストから抽出)
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402

_TOOL_CALL_PATTERN = _re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    _re.DOTALL,
)

_FUNC_CALL_PATTERN = _re.compile(
    r'```(?:json)?\s*(\{[^`]*"name"\s*:.*?\})\s*```',
    _re.DOTALL,
)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """
    モデル出力テキストからツール呼び出しを抽出する。

    2つのフォーマットをサポート:
        <tool_call>{"name": "search_competitor", "arguments": {...}}</tool_call>
        ```json {"name": "...", "arguments": {...}} ```

    Args:
        text -- モデルの生成テキスト

    Returns:
        抽出された ToolCall のリスト (なければ空リスト)
    """
    calls: list[ToolCall] = []

    for pattern in (_TOOL_CALL_PATTERN, _FUNC_CALL_PATTERN):
        for m in pattern.finditer(text):
            try:
                data = json.loads(m.group(1))
                if "name" in data:
                    calls.append(ToolCall.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                pass

    return calls


def build_tool_prompt(tools: list[ToolDefinition]) -> str:
    """
    ツール一覧をモデルへのシステムプロンプトとして整形する。

    Args:
        tools -- ツール定義のリスト

    Returns:
        ツール説明を含むシステムプロンプト文字列
    """
    if not tools:
        return ""

    lines = [
        "You have access to the following tools. "
        "To use a tool, output a JSON block wrapped in <tool_call> tags:\n"
        '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>\n',
        "Available tools:",
    ]
    for t in tools:
        param_strs = []
        for pname, pschema in t.parameters.items():
            req = (
                "" if pschema.required else f" (optional, default={pschema.default!r})"
            )
            param_strs.append(
                f"  - {pname} ({pschema.type}){req}: {pschema.description}"
            )
        param_block = "\n".join(param_strs) if param_strs else "  (no parameters)"
        lines.append(f"\n[{t.name}]\n{t.description}\nParameters:\n{param_block}")

    return "\n".join(lines)
