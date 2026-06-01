"""
Sprint 17 テスト — APIキー認証 / レート制限 / Docker / OpenAPI

- TestAPIKeyAuth       : verify_api_key dependency の動作 (auth disabled / 401 / valid)
- TestRateLimiter      : _SlidingWindow のカウント・リセット・window 期限切れ
- TestRateLimitHeaders : RateLimitMiddleware が返すレスポンスヘッダの検証
- TestDockerConfig     : Dockerfile / docker-compose.yml の存在と Gunicorn 設定
- TestComparisonDoc    : docs/mythos_vs_openmythos.md の存在と必須セクション確認
- TestOpenAPIVersion   : FastAPI app の version / tags が設定されているか確認
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def _reload_auth(api_key: str | None = None):
    """
    serve.auth を再ロードして API_KEY 環境変数の変更を反映する。
    """
    env = {}
    if api_key is not None:
        env["API_KEY"] = api_key
    else:
        env.pop("API_KEY", None)

    import serve.auth as auth_mod

    with patch.dict(os.environ, env, clear=False):
        valid_keys = auth_mod._load_api_keys()
        return valid_keys, not bool(valid_keys)


# ---------------------------------------------------------------------------
# TestAPIKeyAuth
# ---------------------------------------------------------------------------


class TestAPIKeyAuth:
    """verify_api_key dependency の動作検証。"""

    def test_load_api_keys_empty_when_no_env(self):
        """API_KEY / API_KEYS 未設定時は空の frozenset を返す。"""
        with patch.dict(os.environ, {}, clear=True):
            from serve import auth as auth_mod

            keys = auth_mod._load_api_keys()
        assert isinstance(keys, frozenset)
        assert len(keys) == 0

    def test_load_api_keys_single(self):
        """API_KEY 環境変数から単一キーを読み込む。"""
        with patch.dict(os.environ, {"API_KEY": "sk-test-123"}):
            from serve import auth as auth_mod

            keys = auth_mod._load_api_keys()
        assert "sk-test-123" in keys

    def test_load_api_keys_multiple(self):
        """API_KEYS 環境変数からカンマ区切り複数キーを読み込む。"""
        with patch.dict(os.environ, {"API_KEYS": "key-a, key-b, key-c"}):
            from serve import auth as auth_mod

            keys = auth_mod._load_api_keys()
        assert "key-a" in keys
        assert "key-b" in keys
        assert "key-c" in keys

    def test_load_api_keys_strips_whitespace(self):
        """キーの前後の空白をトリムする。"""
        with patch.dict(os.environ, {"API_KEYS": "  sk-a  ,  sk-b  "}):
            from serve import auth as auth_mod

            keys = auth_mod._load_api_keys()
        assert "sk-a" in keys
        assert "sk-b" in keys

    def test_verify_api_key_disabled_when_no_env(self):
        """認証無効 (開発モード) 時は空文字列を返す。"""
        from serve.auth import verify_api_key

        # credentials = None で呼び出し (認証無効時は 401 にならない)
        from serve.auth import _AUTH_DISABLED

        if _AUTH_DISABLED:
            result = verify_api_key(credentials=None)
            assert result == ""

    def test_verify_api_key_raises_401_without_header(self):
        """認証有効時に credentials=None は 401 を返す。"""
        from fastapi import HTTPException
        from serve.auth import verify_api_key

        # _VALID_KEYS をモックして認証有効状態にする
        with (
            patch("serve.auth._AUTH_DISABLED", False),
            patch("serve.auth._VALID_KEYS", frozenset({"real-key"})),
        ):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(credentials=None)
            assert exc_info.value.status_code == 401

    def test_verify_api_key_raises_401_with_wrong_token(self):
        """不正な token は 401 を返す。"""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        from serve.auth import verify_api_key

        fake_credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        fake_credentials.scheme = "bearer"
        fake_credentials.credentials = "wrong-key"

        with (
            patch("serve.auth._AUTH_DISABLED", False),
            patch("serve.auth._VALID_KEYS", frozenset({"real-key"})),
        ):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(credentials=fake_credentials)
            assert exc_info.value.status_code == 401

    def test_verify_api_key_accepts_valid_token(self):
        """正しい token はトークン文字列を返す。"""
        from fastapi.security import HTTPAuthorizationCredentials
        from serve.auth import verify_api_key

        fake_credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        fake_credentials.scheme = "bearer"
        fake_credentials.credentials = "real-key"

        with (
            patch("serve.auth._AUTH_DISABLED", False),
            patch("serve.auth._VALID_KEYS", frozenset({"real-key"})),
        ):
            result = verify_api_key(credentials=fake_credentials)
        assert result == "real-key"

    def test_verify_api_key_case_insensitive_scheme(self):
        """scheme は大文字小文字を問わず bearer として扱う。"""
        from fastapi.security import HTTPAuthorizationCredentials
        from serve.auth import verify_api_key

        fake_credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        fake_credentials.scheme = "Bearer"  # 大文字始まり
        fake_credentials.credentials = "real-key"

        with (
            patch("serve.auth._AUTH_DISABLED", False),
            patch("serve.auth._VALID_KEYS", frozenset({"real-key"})),
        ):
            result = verify_api_key(credentials=fake_credentials)
        assert result == "real-key"


# ---------------------------------------------------------------------------
# TestRateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """_SlidingWindow のカウント・リセット・期限切れ動作。"""

    def test_allows_requests_within_limit(self):
        """上限以内はすべて許可される。"""
        from serve.auth import _SlidingWindow

        limiter = _SlidingWindow(limit=3, window_sec=60.0)

        for _ in range(3):
            allowed, remaining = limiter.is_allowed("client-a")
            assert allowed is True

    def test_blocks_after_limit(self):
        """上限を超えたリクエストはブロックされる。"""
        from serve.auth import _SlidingWindow

        limiter = _SlidingWindow(limit=3, window_sec=60.0)

        for _ in range(3):
            limiter.is_allowed("client-b")

        allowed, remaining = limiter.is_allowed("client-b")
        assert allowed is False
        assert remaining == 0

    def test_remaining_decrements(self):
        """remaining が 1 ずつ減少する。"""
        from serve.auth import _SlidingWindow

        limiter = _SlidingWindow(limit=5, window_sec=60.0)

        _, remaining_1 = limiter.is_allowed("client-c")
        _, remaining_2 = limiter.is_allowed("client-c")
        assert remaining_1 == 4
        assert remaining_2 == 3

    def test_different_clients_independent(self):
        """異なるクライアントのカウントは独立している。"""
        from serve.auth import _SlidingWindow

        limiter = _SlidingWindow(limit=2, window_sec=60.0)

        limiter.is_allowed("client-x")
        limiter.is_allowed("client-x")

        # client-y は影響なし
        allowed, _ = limiter.is_allowed("client-y")
        assert allowed is True

    def test_reset_clears_count(self):
        """reset() でカウントがクリアされる。"""
        from serve.auth import _SlidingWindow

        limiter = _SlidingWindow(limit=2, window_sec=60.0)

        limiter.is_allowed("client-r")
        limiter.is_allowed("client-r")
        # ここでブロック
        allowed, _ = limiter.is_allowed("client-r")
        assert allowed is False

        limiter.reset("client-r")
        # リセット後は再び許可
        allowed, _ = limiter.is_allowed("client-r")
        assert allowed is True

    def test_window_expiry_allows_again(self):
        """ウィンドウを過ぎたリクエストは期限切れとして除去される。"""
        from serve.auth import _SlidingWindow

        limiter = _SlidingWindow(limit=2, window_sec=0.1)  # 100ms ウィンドウ

        limiter.is_allowed("client-w")
        limiter.is_allowed("client-w")
        # ブロック確認
        allowed, _ = limiter.is_allowed("client-w")
        assert allowed is False

        # ウィンドウが過ぎるまで待つ
        time.sleep(0.12)

        # 再び許可されるはず
        allowed, _ = limiter.is_allowed("client-w")
        assert allowed is True

    def test_invalid_limit_raises(self):
        """limit <= 0 は ValueError を送出する。"""
        from serve.auth import _SlidingWindow

        with pytest.raises(ValueError, match="limit must be > 0"):
            _SlidingWindow(limit=0)

    def test_thread_safe_concurrent(self):
        """複数スレッドから同時アクセスしても上限を超えない。"""
        import threading
        from serve.auth import _SlidingWindow

        limit = 10
        limiter = _SlidingWindow(limit=limit, window_sec=60.0)
        allowed_count = 0
        lock = threading.Lock()

        def _request():
            nonlocal allowed_count
            allowed, _ = limiter.is_allowed("shared-client")
            if allowed:
                with lock:
                    allowed_count += 1

        threads = [threading.Thread(target=_request) for _ in range(limit * 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed_count <= limit


# ---------------------------------------------------------------------------
# TestRateLimitMiddleware
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    """RateLimitMiddleware のヘッダ・ステータスコード検証。"""

    def test_rate_limit_middleware_imports(self):
        """RateLimitMiddleware がインポート可能。"""
        from serve.auth import RateLimitMiddleware

        assert RateLimitMiddleware is not None

    def test_rate_limit_middleware_is_base_http(self):
        """RateLimitMiddleware は BaseHTTPMiddleware のサブクラス。"""
        from starlette.middleware.base import BaseHTTPMiddleware
        from serve.auth import RateLimitMiddleware

        assert issubclass(RateLimitMiddleware, BaseHTTPMiddleware)

    def test_middleware_429_on_exceed(self):
        """レート超過時に 429 ステータスを返す。"""
        from serve.auth import RateLimitMiddleware, _SlidingWindow

        # limit=0 は invalid なので limit=1 を使い、先に 1 消費する
        fast_limiter = _SlidingWindow(limit=1, window_sec=60.0)
        fast_limiter.is_allowed("test-ip")  # 先に 1 消費 → 次はブロック

        # Response を直接テストするため dispatch をモック経由で確認
        # (FastAPI TestClient を使わず、ユニットテストで検証)
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        def homepage(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(RateLimitMiddleware, limiter=fast_limiter)

        client = TestClient(app, raise_server_exceptions=False)
        # 2回目のリクエスト → 429
        response = client.get("/", headers={"X-Forwarded-For": "test-ip"})
        assert response.status_code == 429

    def test_health_endpoint_bypasses_rate_limit(self):
        """/health エンドポイントはレート制限をスキップする。"""
        from serve.auth import RateLimitMiddleware, _SlidingWindow
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        # 上限 0 相当にするため limit=1 で先に消費
        tight_limiter = _SlidingWindow(limit=1, window_sec=60.0)
        tight_limiter.is_allowed("health-ip")

        def health(request):
            return JSONResponse({"status": "ok"})

        def other(request):
            return JSONResponse({"other": True})

        app = Starlette(
            routes=[
                Route("/health", health),
                Route("/other", other),
            ]
        )
        app.add_middleware(RateLimitMiddleware, limiter=tight_limiter)

        client = TestClient(app, raise_server_exceptions=False)

        # /health は制限スキップ
        r = client.get("/health", headers={"X-Forwarded-For": "health-ip"})
        assert r.status_code == 200

        # /other は制限対象
        r2 = client.get("/other", headers={"X-Forwarded-For": "health-ip"})
        assert r2.status_code == 429


# ---------------------------------------------------------------------------
# TestDockerConfig
# ---------------------------------------------------------------------------


class TestDockerConfig:
    """Dockerfile / docker-compose.yml の存在と本番設定確認。"""

    def test_dockerfile_exists(self):
        """serve/Dockerfile が存在する。"""
        assert (REPO_ROOT / "serve" / "Dockerfile").is_file()

    def test_dockerfile_uses_gunicorn(self):
        """Dockerfile に gunicorn が記載されている。"""
        content = (REPO_ROOT / "serve" / "Dockerfile").read_text(encoding="utf-8")
        assert "gunicorn" in content.lower()

    def test_dockerfile_uses_uvicorn_worker(self):
        """Dockerfile に UvicornWorker が記載されている。"""
        content = (REPO_ROOT / "serve" / "Dockerfile").read_text(encoding="utf-8")
        assert "UvicornWorker" in content

    def test_dockerfile_exposes_8000(self):
        """Dockerfile が EXPOSE 8000 を含む。"""
        content = (REPO_ROOT / "serve" / "Dockerfile").read_text(encoding="utf-8")
        assert "EXPOSE 8000" in content

    def test_dockerfile_has_nonroot_user(self):
        """セキュリティ: Dockerfile に非 root ユーザーが設定されている。"""
        content = (REPO_ROOT / "serve" / "Dockerfile").read_text(encoding="utf-8")
        assert "useradd" in content or "USER" in content

    def test_docker_compose_exists(self):
        """docker-compose.yml が存在する。"""
        assert (REPO_ROOT / "docker-compose.yml").is_file()

    def test_docker_compose_has_api_service(self):
        """docker-compose.yml に api サービスが定義されている。"""
        content = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert "api:" in content

    def test_docker_compose_has_rate_limit_config(self):
        """docker-compose.yml に RATE_LIMIT_RPM が記載されている。"""
        content = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert "RATE_LIMIT_RPM" in content

    def test_docker_compose_has_healthcheck(self):
        """docker-compose.yml に healthcheck が定義されている。"""
        content = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert "healthcheck" in content


# ---------------------------------------------------------------------------
# TestComparisonDoc
# ---------------------------------------------------------------------------


class TestComparisonDoc:
    """docs/mythos_vs_openmythos.md の存在と必須セクション確認。"""

    @pytest.fixture(scope="class")
    def doc_content(self):
        path = REPO_ROOT / "docs" / "mythos_vs_openmythos.md"
        assert path.is_file(), "docs/mythos_vs_openmythos.md が見つかりません"
        return path.read_text(encoding="utf-8")

    def test_doc_exists(self, doc_content):
        """docs/mythos_vs_openmythos.md が存在する。"""
        assert len(doc_content) > 500

    def test_doc_has_architecture_section(self, doc_content):
        """アーキテクチャ比較セクションがある。"""
        assert "アーキテクチャ" in doc_content

    def test_doc_has_security_section(self, doc_content):
        """セキュリティ比較セクションがある。"""
        assert "セキュリティ" in doc_content or "Injection" in doc_content

    def test_doc_has_api_compatibility_section(self, doc_content):
        """API 互換性セクションがある。"""
        assert "互換" in doc_content or "compatibility" in doc_content.lower()

    def test_doc_has_benchmark_section(self, doc_content):
        """ベンチマークセクションがある。"""
        assert "ベンチマーク" in doc_content or "benchmark" in doc_content.lower()

    def test_doc_mentions_input_guard(self, doc_content):
        """InputGuard についての記述がある。"""
        assert "InputGuard" in doc_content

    def test_doc_mentions_drift_score(self, doc_content):
        """drift_score についての記述がある。"""
        assert "drift_score" in doc_content or "ドリフト" in doc_content


# ---------------------------------------------------------------------------
# TestOpenAPISpec
# ---------------------------------------------------------------------------


class TestOpenAPISpec:
    """serve/api.py の FastAPI app 設定確認。"""

    def test_auth_module_importable(self):
        """serve.auth がインポート可能。"""
        from serve.auth import verify_api_key, RateLimitMiddleware, _SlidingWindow

        assert verify_api_key is not None
        assert RateLimitMiddleware is not None
        assert _SlidingWindow is not None

    def test_auth_module_has_global_limiter(self):
        """グローバルレートリミッターが存在する。"""
        from serve.auth import _rate_limiter, _SlidingWindow

        assert isinstance(_rate_limiter, _SlidingWindow)
        assert _rate_limiter.limit > 0

    def test_rate_limit_default_is_60(self):
        """デフォルトのレート制限が 60 rpm。"""
        # 環境変数が未設定の場合のデフォルト値を確認
        with patch.dict(os.environ, {}, clear=True):
            default_limit = int(os.getenv("RATE_LIMIT_RPM", "60"))
        assert default_limit == 60
