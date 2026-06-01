"""
OpenMythos API Authentication & Rate Limiting.

認証方式
--------
Bearer Token (環境変数 ``API_KEY`` または ``API_KEYS`` カンマ区切り)。
いずれも未設定の場合は開発モードとして認証をスキップする。

レート制限
----------
スライディングウィンドウ方式のインメモリ実装。
外部依存なし。エンドポイント共通で RPM (Requests Per Minute) を制御する。

環境変数
--------
API_KEY          : 単一キー文字列
API_KEYS         : カンマ区切り複数キー文字列 (API_KEY より優先)
RATE_LIMIT_RPM   : 1分あたりのリクエスト上限 (default: 60)
RATE_LIMIT_BURST : バースト許容幅 (default: RATE_LIMIT_RPM の 20%)

使い方 (serve/api.py) ::

    from serve.auth import verify_api_key, RateLimitMiddleware

    app = FastAPI(dependencies=[Depends(verify_api_key)])
    app.add_middleware(RateLimitMiddleware)
"""

from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock
from typing import Deque, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# ---------------------------------------------------------------------------
# API Key 管理
# ---------------------------------------------------------------------------


def _load_api_keys() -> frozenset[str]:
    """環境変数から有効な API キーのセットを構築する。"""
    raw = os.getenv("API_KEYS", os.getenv("API_KEY", ""))
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    return frozenset(keys)


# モジュール読み込み時に確定させる（テスト時は reload して再設定）
_VALID_KEYS: frozenset[str] = _load_api_keys()
_AUTH_DISABLED: bool = not bool(_VALID_KEYS)

_bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """
    Bearer Token を検証する FastAPI dependency。

    ``API_KEY`` / ``API_KEYS`` 環境変数が未設定の場合は認証をスキップし
    空文字列を返す（開発モード）。

    Parameters
    ----------
    credentials:
        FastAPI が ``Authorization`` ヘッダから自動抽出した認証情報。

    Returns
    -------
    str
        検証済みの token 文字列、または開発モード時は ``""``。

    Raises
    ------
    HTTPException(401)
        ``Authorization`` ヘッダが欠如 / 形式不正 / キーが無効な場合。
    """
    if _AUTH_DISABLED:
        return ""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing or invalid. "
            "Provide: Authorization: Bearer <api-key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if token not in _VALID_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ---------------------------------------------------------------------------
# スライディングウィンドウ レートリミッター
# ---------------------------------------------------------------------------


class _SlidingWindow:
    """
    スレッドセーフなスライディングウィンドウ レートリミッター。

    各クライアントキー（IP アドレス等）ごとに独立したカウントを管理する。
    メモリ使用量は O(limit × n_active_clients)。

    Parameters
    ----------
    limit:
        ウィンドウ内の最大リクエスト数。
    window_sec:
        ウィンドウの長さ（秒）。デフォルト 60 秒 = 1 分。
    """

    def __init__(self, limit: int, window_sec: float = 60.0) -> None:
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")
        self.limit = limit
        self.window_sec = window_sec
        self._lock = Lock()
        self._buckets: Dict[str, Deque[float]] = {}

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        ``key`` に対してリクエストを 1 件カウントし、許可するか判定する。

        Parameters
        ----------
        key:
            クライアントを識別する文字列（IP アドレス等）。

        Returns
        -------
        (allowed, remaining)
            ``allowed`` が False の場合はレート超過。
            ``remaining`` は現在のウィンドウで残り何件許可されるか。
        """
        now = time.monotonic()
        cutoff = now - self.window_sec

        with self._lock:
            bucket = self._buckets.setdefault(key, deque())

            # 期限切れエントリを削除
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                return False, 0

            bucket.append(now)
            remaining = self.limit - len(bucket)
            return True, remaining

    def reset(self, key: str) -> None:
        """テスト用: 指定キーのカウントをリセットする。"""
        with self._lock:
            self._buckets.pop(key, None)


# グローバルインスタンス（アプリ全体で共有）
_rpm_limit = int(os.getenv("RATE_LIMIT_RPM", "60"))
_rate_limiter: _SlidingWindow = _SlidingWindow(limit=max(1, _rpm_limit))


# ---------------------------------------------------------------------------
# Rate Limit Middleware (Starlette)
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    クライアント IP ベースの RPM レート制限ミドルウェア。

    ``/health`` エンドポイントはヘルスチェック用途のためスキップする。

    リクエストヘッダ:
        ``X-Forwarded-For`` が存在する場合は最初の IP を使用
        （Cloud Run / リバースプロキシ配下での運用を想定）。

    レスポンスヘッダ:
        - ``X-RateLimit-Limit``    : 上限値
        - ``X-RateLimit-Remaining``: 残りリクエスト数
        - ``Retry-After``          : 超過時のみ付与（秒）

    環境変数:
        ``RATE_LIMIT_RPM`` で上限を変更できる（サーバ再起動が必要）。
    """

    def __init__(self, app: ASGIApp, limiter: Optional[_SlidingWindow] = None) -> None:
        super().__init__(app)
        self._limiter = limiter or _rate_limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        # /health はスキップ（ヘルスチェック・Cloud Run のプローブ用）
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[
            0
        ].strip() or (request.client.host if request.client else "unknown")

        allowed, remaining = self._limiter.is_allowed(client_ip)

        if not allowed:
            return Response(
                content='{"detail":"Rate limit exceeded. Retry after 60 seconds."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self._limiter.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
